"""Run the app's inference path over held-out images and tally failure modes.

Uses the same OCR -> two checkpoints -> assembly -> cleanup path as app.py, so
the counts describe what a user sees in the Streamlit table, not just token F1.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import load_dataset

from app import clean_amount, extract
from src.data.cord import parse
from src.eval.field_eval import match
from src.utils.env import get_paths

SROIE_FIELDS = ("company", "address", "date", "total")
NOISY_FIELDS = ("grand_total", "subtotal", "tax", "service_charge",
                "discount", "cash", "card", "e_money", "change")


def check_sroie(rows, limit: int) -> tuple[dict, list[str]]:
    stats = Counter()
    examples = []
    for i in range(min(limit, len(rows))):
        row = rows[i]
        cleaned, raw, words, n_words = extract(row["image"].convert("RGB"))
        if n_words < 20:
            stats["low_ocr_words"] += 1
        for field in SROIE_FIELDS:
            predicted = {"company": cleaned["vendor"]["name"],
                         "address": cleaned["vendor"]["address"],
                         "date": cleaned["transaction"]["date"],
                         "total": cleaned["totals"]["grand_total"]}[field]
            truth = (row.get("entities") or {}).get(field)
            if truth:
                stats[f"{field}_expected"] += 1
                exact, _ = match(predicted, truth,
                                 numeric=(field == "total"),
                                 is_date=(field == "date"))
                if exact:
                    stats[f"{field}_exact"] += 1
                elif len(examples) < 8:
                    examples.append(
                        f"sroie[{i}] {field}: got {predicted!r}, "
                        f"expected {truth!r} (ocr words={n_words})")
        for field in NOISY_FIELDS:
            value = raw["totals"].get(field) or raw["payment"].get(field)
            if value and clean_amount(value) is None:
                stats["non_numeric_noise_fields"] += 1
        stats["raw_junk_items"] += sum(
            1 for item in raw["line_items"]
            if clean_amount(item.get("total_price")) is None)
        stats["items_dropped_by_cleanup"] += (
            len(raw["line_items"]) - len(cleaned["line_items"]))
    return dict(stats), examples


def check_cord(rows, limit: int) -> tuple[dict, list[str]]:
    stats = Counter()
    examples = []
    for i in range(min(limit, len(rows))):
        row = rows[i]
        cleaned, raw, words, n_words = extract(row["image"].convert("RGB"))
        example = parse(row["ground_truth"])
        groups = {example.group_ids[j] for j, tag in enumerate(
            example.labels) if example.group_ids[j] >= 0}
        expected, got = len(groups), len(cleaned["line_items"])
        stats["items_expected"] += expected
        stats["items_extracted"] += got
        if got < expected:
            stats["receipts_with_missing_items"] += 1
            if len(examples) < 8:
                examples.append(
                    f"cord[{i}] expected {expected} items, got {got} "
                    f"(ocr words={n_words})")
        stats["raw_junk_items"] += sum(
            1 for item in raw["line_items"]
            if clean_amount(item.get("total_price")) is None)
        stats["items_dropped_by_cleanup"] += (
            len(raw["line_items"]) - len(cleaned["line_items"]))
        for field in NOISY_FIELDS:
            value = raw["totals"].get(field) or raw["payment"].get(field)
            if value and clean_amount(value) is None:
                stats["non_numeric_noise_fields"] += 1
    return dict(stats), examples


def fmt(stats: dict, keys: tuple[str, ...]) -> str:
    return "\n".join(f"  {k:<32} {stats.get(k, 0)}" for k in keys)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()

    paths = get_paths()
    sroie = load_dataset("jsdnrs/ICDAR2019-SROIE",
                         cache_dir=str(paths.cache))["test"]
    cord = load_dataset("naver-clova-ix/cord-v2",
                        cache_dir=str(paths.cache))["test"]

    print(f"Running app inference on {args.limit} SROIE + {args.limit} "
          f"CORD test receipts ...\n")
    s_stats, s_examples = check_sroie(sroie, args.limit)
    c_stats, c_examples = check_cord(cord, args.limit)

    print("SROIE:")
    print(fmt(s_stats, ("low_ocr_words", "company_expected", "company_exact",
                        "date_expected", "date_exact", "total_expected",
                        "total_exact", "address_expected", "address_exact",
                        "non_numeric_noise_fields", "raw_junk_items",
                        "items_dropped_by_cleanup")))
    print("\nCORD:")
    print(fmt(c_stats, ("items_expected", "items_extracted",
                        "receipts_with_missing_items", "low_ocr_words",
                        "non_numeric_noise_fields", "raw_junk_items",
                        "items_dropped_by_cleanup")))

    print("\nFailure examples:")
    for line in s_examples + c_examples:
        print(f"  {line}")

    report = {"limit": args.limit, "sroie": s_stats, "cord": c_stats,
              "examples": s_examples + c_examples}
    out = paths.logs / "app_diagnostics.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
