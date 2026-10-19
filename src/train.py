"""Fine-tune LayoutLMv3 for key information extraction.

    python -m src.train --config configs/cord_layoutlmv3.yaml
    python -m src.train --config configs/cord_layoutlmv3.yaml --train-size 100
    python -m src.train --dataset sroie --epochs 20

Batch size, sequence length and fp16 come from the detected GPU unless the
config pins them, and the effective batch size is held at 16 everywhere via
gradient accumulation so a 4 GB laptop run and a 16 GB cloud run remain
comparable.

Every run writes results/logs/<name>.json with the final metrics, the
resolved config and the environment, so a result can always be traced back
to the machine and settings that produced it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from src.data.loaders import (Collator, build_dataset, build_processor,
                              subsample)
from src.data.schema import (CORD_ABSENT_FROM_TEST, CORD_CATEGORIES,
                             CORD_CORE_FIELDS, CORD_ID2LABEL, CORD_LABELS,
                             CORD_TOO_RARE, SROIE_CATEGORIES,
                             SROIE_CORE_FIELDS, SROIE_ID2LABEL, SROIE_LABELS)
from src.eval.metrics import make_compute_metrics
from src.utils.env import get_paths, recommended_batch_config, summary

DATASETS = {
    "cord": {
        "hf_id": "naver-clova-ix/cord-v2",
        "labels": CORD_LABELS,
        "id2label": CORD_ID2LABEL,
        "categories": CORD_CATEGORIES,
        "core": CORD_CORE_FIELDS,
        # Five categories have no test support and two have almost no train
        # support; both groups are excluded from the macro average only.
        "unscoreable": CORD_ABSENT_FROM_TEST | CORD_TOO_RARE,
        "eval_split": "validation",
    },
    "sroie": {
        "hf_id": "jsdnrs/ICDAR2019-SROIE",
        "labels": SROIE_LABELS,
        "id2label": SROIE_ID2LABEL,
        "categories": SROIE_CATEGORIES,
        "core": SROIE_CORE_FIELDS,
        "unscoreable": frozenset(),
        # SROIE ships train and test only; carve a validation split off
        # train so the test set is never used for model selection.
        "eval_split": None,
    },
}


def load_config(path: str | None) -> dict:
    if not path:
        return {}
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def resolve(cfg: dict, args) -> dict:
    """Merge config file, CLI flags and GPU-derived defaults."""
    auto = recommended_batch_config()
    training = cfg.get("training", {})
    data = cfg.get("data", {})
    model = cfg.get("model", {})
    experiment = cfg.get("experiment", {})

    def pick(cli, cfg_value, fallback):
        if cli is not None:
            return cli
        return fallback if cfg_value is None else cfg_value

    resolved = {
        "dataset": args.dataset,
        "checkpoint": model.get("checkpoint", "microsoft/layoutlmv3-base"),
        "seed": pick(args.seed, experiment.get("seed"), 42),
        "train_size": pick(args.train_size, data.get("train_subset_size"), None),
        "max_seq_length": pick(args.max_seq_length,
                               data.get("max_seq_length"),
                               auto["max_seq_length"]),
        "epochs": pick(args.epochs, training.get("num_train_epochs"), 20),
        "learning_rate": pick(args.learning_rate,
                              training.get("learning_rate"), 1e-5),
        "weight_decay": training.get("weight_decay", 0.01),
        "warmup_ratio": training.get("warmup_ratio", 0.1),
        "batch_size": pick(args.batch_size,
                           training.get("per_device_batch_size"),
                           auto["per_device_batch_size"]),
        "grad_accum": pick(None, training.get("gradient_accumulation_steps"),
                           auto["gradient_accumulation_steps"]),
        "fp16": pick(None, training.get("fp16"), auto["fp16"]),
        "name": args.name or experiment.get("name") or f"{args.dataset}_run",
    }
    # Keep the effective batch size fixed when only one side was overridden.
    if args.batch_size is not None:
        resolved["grad_accum"] = max(1, 16 // args.batch_size)
    resolved["effective_batch_size"] = (
        resolved["batch_size"] * resolved["grad_accum"])
    return resolved


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="cord")
    ap.add_argument("--train-size", type=int,
                    help="low-data study: use only this many training receipts")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--learning-rate", type=float)
    ap.add_argument("--max-seq-length", type=int)
    ap.add_argument("--name", help="run name; results/logs/<name>.json")
    ap.add_argument("--no-save", action="store_true",
                    help="skip checkpoints (sweeps do not need 15 copies)")
    args = ap.parse_args()

    import torch
    from transformers import (LayoutLMv3ForTokenClassification, Trainer,
                              TrainingArguments, set_seed)

    spec = DATASETS[args.dataset]
    cfg = resolve(load_config(args.config), args)
    paths = get_paths()
    set_seed(cfg["seed"])

    print(json.dumps({"env": summary(), "config": cfg}, indent=2, default=str))

    from datasets import load_dataset
    ds = load_dataset(spec["hf_id"], cache_dir=str(paths.cache))

    train_split = ds["train"]
    if spec["eval_split"] and spec["eval_split"] in ds:
        eval_split = ds[spec["eval_split"]]
    else:
        # Hold out 10% of train for model selection, leaving test untouched.
        parts = train_split.train_test_split(test_size=0.1, seed=cfg["seed"])
        train_split, eval_split = parts["train"], parts["test"]

    train_split = subsample(train_split, cfg["train_size"], cfg["seed"])
    print(f"\ntrain={len(train_split)}  eval={len(eval_split)}  "
          f"labels={len(spec['labels'])}")

    processor = build_processor(cfg["checkpoint"], cache_dir=str(paths.cache))
    collator = Collator(processor, max_length=cfg["max_seq_length"])

    model = LayoutLMv3ForTokenClassification.from_pretrained(
        cfg["checkpoint"],
        num_labels=len(spec["labels"]),
        id2label=spec["id2label"],
        label2id={v: k for k, v in spec["id2label"].items()},
        cache_dir=str(paths.cache),
    )

    out_dir = paths.checkpoints / cfg["name"]
    targs = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=cfg["epochs"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        warmup_ratio=cfg["warmup_ratio"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        fp16=cfg["fp16"],
        eval_strategy="epoch",
        save_strategy="no" if args.no_save else "epoch",
        save_total_limit=1,
        load_best_model_at_end=not args.no_save,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=20,
        report_to=[],
        seed=cfg["seed"],
        dataloader_num_workers=0,   # Windows spawns processes; workers cost
                                    # more than they save on datasets this size
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=build_dataset(train_split, args.dataset),
        eval_dataset=build_dataset(eval_split, args.dataset),
        data_collator=collator,
        compute_metrics=make_compute_metrics(
            spec["id2label"],
            unscoreable=spec["unscoreable"],
            core_fields=spec["core"],
        ),
    )

    started = time.time()
    trainer.train()
    elapsed = time.time() - started

    metrics = trainer.evaluate()
    detail = metrics.pop("eval__detail", {})

    record = {
        "name": cfg["name"],
        "dataset": args.dataset,
        "config": cfg,
        "env": summary(),
        "train_size": len(train_split),
        "eval_size": len(eval_split),
        "train_seconds": round(elapsed, 1),
        "metrics": {k: (float(v) if isinstance(v, (int, float, np.floating))
                        else v)
                    for k, v in metrics.items()},
        "detail": detail,
    }
    log_path = paths.logs / f"{cfg['name']}.json"
    log_path.write_text(json.dumps(record, indent=2, default=str))

    print(f"\n{'=' * 60}")
    print(f"f1 (micro, all)      {metrics.get('eval_f1', 0):.4f}")
    print(f"f1 (core fields)     {metrics.get('eval_core_f1', 0):.4f}")
    print(f"f1 (macro scoreable) {metrics.get('eval_macro_f1_scoreable', 0):.4f}")
    print(f"trained in {elapsed / 60:.1f} min")
    print(f"saved -> {log_path}")


if __name__ == "__main__":
    sys.exit(main())
