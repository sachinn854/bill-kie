"""Measure what fine-tuning added over the pretrained backbones.

The hub checkpoints ship no token-classification head, so a fresh classifier is
attached before scoring. A near-zero F1 is the expected result: the pretrained
weights know language and layout, but not this labelling task.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from datasets import load_dataset
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          LayoutLMv3ForTokenClassification, set_seed)

from scripts.final_eval import predict, scalar_metrics
from src.data.loaders import (Collator, TextOnlyCollator, build_dataset,
                              build_processor)
from src.eval.metrics import compute as compute_entity_metrics, decode
from src.train import DATASETS
from src.utils.env import get_paths, summary


def run_pretrained(name, split, spec, hub_id, text_only, paths, batch_size):
    labels = len(spec["labels"])
    if text_only:
        tokenizer = AutoTokenizer.from_pretrained(
            hub_id, cache_dir=str(paths.cache))
        collator = TextOnlyCollator(tokenizer, max_length=256)
        model = AutoModelForTokenClassification.from_pretrained(
            hub_id, num_labels=labels, id2label=spec["id2label"],
            label2id={v: k for k, v in spec["id2label"].items()},
            cache_dir=str(paths.cache))
    else:
        processor = build_processor(hub_id, cache_dir=str(paths.cache))
        collator = Collator(processor, max_length=256)
        model = LayoutLMv3ForTokenClassification.from_pretrained(
            hub_id, num_labels=labels, id2label=spec["id2label"],
            label2id={v: k for k, v in spec["id2label"].items()},
            cache_dir=str(paths.cache))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = build_dataset(split, name)
    logits, gold = predict(model.to(device), dataset, collator, device,
                           batch_size)
    pred_tags, true_tags = decode(logits, gold, spec["id2label"])
    results = compute_entity_metrics(
        pred_tags, true_tags, unscoreable=spec["unscoreable"],
        core_fields=spec["core"])
    return {
        "model": hub_id,
        "name": name,
        "split": "test",
        "n_documents": len(dataset),
        "head": "freshly initialised (no task training)",
        "env": summary(),
        "metrics": scalar_metrics(results),
    }


def main() -> None:
    set_seed(42)
    paths = get_paths()
    ds = load_dataset("naver-clova-ix/cord-v2", cache_dir=str(paths.cache))
    spec = DATASETS["cord"]

    rows = []
    for hub_id, text_only, ours in (
        ("microsoft/layoutlmv3-base", False, "cord_layoutlmv3_base"),
        ("bert-base-uncased", True, "cord_textonly_bert"),
    ):
        print(f"\n{hub_id} + fresh head on CORD test ...")
        record = run_pretrained("cord", ds["test"], spec, hub_id,
                                text_only, paths, batch_size=4)
        out = paths.logs / f"pretrained_{'bert' if text_only else 'layoutlmv3'}.json"
        out.write_text(json.dumps(record, indent=2, default=str))
        trained = json.loads((paths.logs / f"test_{ours}.json").read_text())
        m, tm = record["metrics"], trained["metrics"]
        rows.append({
            "backbone": hub_id,
            "pretrained_f1": m["f1"],
            "pretrained_core_f1": m["core_f1"],
            "finetuned_f1": tm["f1"],
            "finetuned_core_f1": tm["core_f1"],
            "delta_f1": tm["f1"] - m["f1"],
        })
        print(f"  pretrained F1 {m['f1']:.4f}  "
              f"-> fine-tuned F1 {tm['f1']:.4f}")
        print(f"  saved -> {out}")

    table = {
        "dataset": "CORD test (100 receipts)",
        "note": "pretrained rows use a freshly initialised classification head",
        "rows": rows,
    }
    out = paths.logs / "pretrained_vs_finetuned.json"
    out.write_text(json.dumps(table, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
