"""Measure how well SROIE entity values can be aligned to page words.

SROIE labels values, not words, so every training label is the product of a
string match that may have gone wrong. This reports, per field, which
matching stage succeeded - and how often none did. Those numbers are the
label-noise budget for the SROIE model and should be quoted in the report
rather than left implicit: a field matched only at the fuzzy stage carries
supervision that may be wrong, and an unmatched field is a silent false
negative that caps achievable recall.

    python scripts/validate_sroie.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from datasets import load_dataset

from src.data.cord import validate
from src.data.schema import SROIE_CATEGORIES, SROIE_ID2LABEL
from src.data.sroie import build_example
from src.utils.env import get_paths

DATASET_ID = "jsdnrs/ICDAR2019-SROIE"


def main() -> None:
    paths = get_paths()
    ds = load_dataset(DATASET_ID, cache_dir=str(paths.cache))

    report: dict = {"dataset": DATASET_ID, "splits": {}}

    for split in ds:
        stage_counts: dict[str, Counter] = defaultdict(Counter)
        multi_candidate: Counter = Counter()
        box_problems: Counter = Counter()
        bad_bio = 0
        lengths: list[int] = []

        for row in ds[split]:
            ex, rep = build_example(row)
            lengths.append(len(ex))

            for p in validate(ex):
                box_problems[p.split(":")[0]] += 1

            prev = "O"
            for lid in ex.labels:
                tag = SROIE_ID2LABEL[lid]
                if tag.startswith("I-"):
                    cat = tag[2:]
                    if prev not in (f"B-{cat}", f"I-{cat}"):
                        bad_bio += 1
                prev = tag

            for name in SROIE_CATEGORIES:
                stage_counts[name][rep.stages.get(name) or "UNMATCHED"] += 1
                if rep.candidates.get(name, 0) > 1:
                    multi_candidate[name] += 1

        n = len(ds[split])
        arr = np.array(lengths)
        print()
        print(f"=== {split}  (n={n}) ===")
        print(f"  words/doc: mean {arr.mean():.1f}  median {np.median(arr):.0f}  "
              f"max {arr.max()}")
        print(f"  BIO violations: {bad_bio}   box problems: "
              f"{sum(box_problems.values())}")
        if box_problems:
            for p, c in box_problems.most_common():
                print(f"    PROBLEM {p}: {c}")

        print(f"  {'field':<10}{'exact':>8}{'alnum':>8}{'multi':>8}{'fuzzy':>8}"
              f"{'overlap':>9}{'UNMATCHED':>11}{'ambiguous':>11}")
        split_stats = {}
        for name in SROIE_CATEGORIES:
            c = stage_counts[name]
            overlap = sum(v for k, v in c.items() if k.endswith(":overlap"))
            matched = (c["exact"] + c["alnum"] + c["multipart"]
                       + c["fuzzy"])
            row_stats = {
                "exact": c["exact"], "alnum": c["alnum"],
                "multipart": c["multipart"], "fuzzy": c["fuzzy"],
                "overlap": overlap, "unmatched": c["UNMATCHED"],
                "multi_candidate": multi_candidate[name],
                "matched_pct": round(100 * matched / n, 1),
            }
            split_stats[name] = row_stats
            print(f"  {name:<10}{c['exact']:>8}{c['alnum']:>8}"
                  f"{c['multipart']:>8}{c['fuzzy']:>8}{overlap:>9}"
                  f"{c['UNMATCHED']:>9}{multi_candidate[name]:>7}")

        print("  matched at all: " + "  ".join(
            f"{k}={v['matched_pct']}%" for k, v in split_stats.items()))

        report["splits"][split] = {
            "n": n,
            "words_per_doc_mean": round(float(arr.mean()), 1),
            "bio_violations": bad_bio,
            "box_problems": sum(box_problems.values()),
            "fields": split_stats,
        }

    out = paths.logs / "sroie_alignment.json"
    out.write_text(json.dumps(report, indent=2))
    print()
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
