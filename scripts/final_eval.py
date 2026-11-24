"""Evaluate the saved checkpoints on the test splits training never touched.

Writes token-level entity F1 for CORD, SROIE and the text-only ablation, then
converts SROIE tags into field values and scores them with the same field-level
metric used for the Tesseract rules baseline. That last step is the only fair
way to put the transformer and the rules pipeline in one table.

    python scripts/final_eval.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from datasets import load_dataset
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          LayoutLMv3ForTokenClassification)

from src.assemble import assemble, extract_spans
from src.data.loaders import (Collator, TextOnlyCollator, build_dataset,
                              build_processor)
from src.eval.field_eval import evaluate as field_evaluate, format_report
from src.eval.metrics import compute as compute_entity_metrics, decode
from src.train import DATASETS
from src.utils.env import get_paths, summary

SCORED = ("company", "address", "date", "total")
NUMERIC_FIELDS = ("total",)
DATE_FIELDS = ("date",)


def find_checkpoint(name: str) -> Path:
    root = get_paths().checkpoints / name
    candidates = sorted(root.glob("checkpoint-*"),
                        key=lambda p: int(p.name.split("-")[-1]))
    if not candidates:
        raise FileNotFoundError(f"no checkpoint found under {root}")
    return candidates[-1]


def predict(model, dataset, collator, device: str, batch_size: int = 4):
    model.eval()
    preds, labels = [], []
    for start in range(0, len(dataset), batch_size):
        features = [dataset[i] for i in
                    range(start, min(start + batch_size, len(dataset)))]
        batch = collator(features)
        gold = batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            logits = model(**batch).logits
        preds.append(logits.cpu().numpy())
        labels.append(gold.numpy())
    width = max(p.shape[1] for p in preds)
    padded_preds, padded_labels = [], []
    for p, g in zip(preds, labels):
        pad = width - p.shape[1]
        if pad:
            p = np.pad(p, ((0, 0), (0, pad), (0, 0)))
            g = np.pad(g, ((0, 0), (0, pad)), constant_values=-100)
        padded_preds.append(p)
        padded_labels.append(g)
    return np.concatenate(padded_preds), np.concatenate(padded_labels)


def scalar_metrics(results: dict) -> dict:
    return {k: (float(v) if isinstance(v, (int, float, np.floating))
                else v)
            for k, v in results.items()}


def evaluate_dataset(name: str, split, spec: dict, checkpoint: Path,
                     text_only: bool, paths) -> tuple[dict, object, object]:
    collator_name = "text_only" if text_only else "layoutlmv3"
    if text_only:
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
        collator = TextOnlyCollator(tokenizer,
                                    max_length=spec.get("max_seq_length", 256))
        model = AutoModelForTokenClassification.from_pretrained(str(checkpoint))
    else:
        processor = build_processor("microsoft/layoutlmv3-base",
                                    cache_dir=str(paths.cache))
        collator = Collator(processor,
                            max_length=spec.get("max_seq_length", 256))
        model = LayoutLMv3ForTokenClassification.from_pretrained(
            str(checkpoint), cache_dir=str(paths.cache))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    dataset = build_dataset(split, name)
    logits, gold = predict(model, dataset, collator, device)
    pred_tags, true_tags = decode(logits, gold, spec["id2label"])
    results = compute_entity_metrics(
        pred_tags, true_tags, unscoreable=spec["unscoreable"],
        core_fields=spec["core"])

    record = {
        "name": name,
        "checkpoint": str(checkpoint),
        "split": "test",
        "n_documents": len(dataset),
        "env": summary(),
        "metrics": scalar_metrics(results),
        "detail": results,
    }
    return record, pred_tags, dataset


def sroie_field_scores(pred_tags, dataset, split) -> dict:
    predictions, truths = [], []
    for i, tags in enumerate(pred_tags):
        row = split[i]
        item = dataset[i]
        words, boxes = item["words"], item["boxes"]
        n = min(len(words), len(boxes), len(tags))
        spans = extract_spans(words[:n], boxes[:n], tags[:n])
        document = assemble(sroie_spans=spans)
        predictions.append({
            "company": document["vendor"]["name"],
            "address": document["vendor"]["address"],
            "date": document["transaction"]["date"],
            "total": document["totals"]["grand_total"],
        })
        truths.append({k: (row.get("entities") or {}).get(k) for k in SCORED})

    return field_evaluate(predictions, truths, numeric_fields=NUMERIC_FIELDS,
                          date_fields=DATE_FIELDS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    paths = get_paths()
    ds = load_dataset("naver-clova-ix/cord-v2", cache_dir=str(paths.cache))
    sroie = load_dataset("jsdnrs/ICDAR2019-SROIE", cache_dir=str(paths.cache))

    records = []
    for name, split, checkpoint_name, text_only in (
        ("cord", ds["test"], "cord_layoutlmv3_base", False),
        ("cord", ds["test"], "cord_textonly_bert", True),
        ("sroie", sroie["test"], "sroie_layoutlmv3_base", False),
    ):
        spec = dict(DATASETS[name], max_seq_length=256)
        checkpoint = find_checkpoint(checkpoint_name)
        print(f"\n[{checkpoint_name}] {checkpoint.name} -> test ({len(split)})")
        record, pred_tags, dataset = evaluate_dataset(
            name, split, spec, checkpoint, text_only, paths)
        m = record["metrics"]
        print(f"  f1 {m['f1']:.4f}  core {m['core_f1']:.4f}  "
              f"macro {m['macro_f1_scoreable']:.4f}")
        out = paths.logs / f"test_{checkpoint_name}.json"
        out.write_text(json.dumps(record, indent=2, default=str))
        print(f"  saved -> {out}")
        records.append((checkpoint_name, record))

        if name == "sroie":
            field_results = sroie_field_scores(pred_tags, dataset, split)
            baseline = json.loads(
                (paths.logs / "baseline_rules_sroie.json").read_text())
            field_record = {
                "transformer": field_results,
                "rules_baseline": baseline["results"],
                "n_documents": len(split),
            }
            out = paths.logs / "test_sroie_field_comparison.json"
            out.write_text(json.dumps(field_record, indent=2))
            print("\nfield-level, exact match (transformer):")
            print(format_report(field_results))
            print("\nfield-level, exact match (Tesseract + rules):")
            print(format_report(baseline["results"]))
            print(f"\nsaved -> {out}")

    print(f"\nevaluated {len(records)} checkpoints on held-out test splits")


if __name__ == "__main__":
    main()
