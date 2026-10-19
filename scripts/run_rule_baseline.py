"""Evaluate the Tesseract + rules baseline on SROIE.

    python scripts/run_rule_baseline.py
    python scripts/run_rule_baseline.py --limit 50     # quick check

SROIE is used because it annotates field *values*, which is what a
rule-based pipeline produces. CORD annotates word categories and has no
vendor or date at all, so a value-level comparison there would only cover
the totals.

CPU only - no GPU is touched. Tesseract is the slow part, roughly a
second per receipt.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import load_dataset

from src.baselines.rules import RuleConfig, extract
from src.eval.field_eval import evaluate, format_report
from src.utils.env import configure_tesseract, get_paths

DATASET_ID = "jsdnrs/ICDAR2019-SROIE"
NUMERIC_FIELDS = ("total", "subtotal", "tax", "service_charge", "cash", "change")
DATE_FIELDS = ("date",)
# SROIE annotates only these four; the rest the baseline emits cannot be
# scored here and are reported as extraction coverage only.
SCORED = ("company", "address", "date", "total")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--psm", type=int, default=6)
    ap.add_argument("--name", default="baseline_rules_sroie")
    args = ap.parse_args()

    paths = get_paths()
    tess = configure_tesseract()
    if not tess:
        print("Tesseract not found. Install it, or see env.find_tesseract.")
        sys.exit(1)
    print(f"tesseract {tess}")

    ds = load_dataset(DATASET_ID, cache_dir=str(paths.cache))[args.split]
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"{args.split}: {len(ds)} receipts\n")

    cfg = RuleConfig(psm=args.psm)
    predictions, truths = [], []
    started = time.time()
    for i, row in enumerate(ds, 1):
        predictions.append(extract(row["image"], cfg))
        truths.append({k: (row["entities"] or {}).get(k) for k in SCORED})
        if i % 25 == 0 or i == len(ds):
            rate = i / (time.time() - started)
            print(f"  {i}/{len(ds)}  ({rate:.1f} receipts/s)", end="\r")
    elapsed = time.time() - started
    print(f"\nOCR + rules took {elapsed / 60:.1f} min "
          f"({elapsed / len(ds):.2f} s/receipt)\n")

    scored_predictions = [{k: p.get(k) for k in SCORED} for p in predictions]
    results = evaluate(scored_predictions, truths,
                       numeric_fields=NUMERIC_FIELDS, date_fields=DATE_FIELDS)
    print(format_report(results))

    record = {
        "name": args.name,
        "baseline": "tesseract+rules",
        "dataset": DATASET_ID,
        "split": args.split,
        "n": len(ds),
        "seconds_per_receipt": round(elapsed / len(ds), 3),
        "psm": args.psm,
        "results": results,
    }
    out = paths.logs / f"{args.name}.json"
    out.write_text(json.dumps(record, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
