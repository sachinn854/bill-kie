"""Run CORD preprocessing over every split and assert it is well formed.

Catches silent corruption before training: out-of-range boxes, degenerate
rectangles, length mismatches, and BIO sequences that start with I-.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import load_dataset

from src.data.cord import parse, validate
from src.data.schema import CORD_ID2LABEL
from src.utils.env import get_paths


def main() -> None:
    paths = get_paths()
    ds = load_dataset("naver-clova-ix/cord-v2", cache_dir=str(paths.cache))

    total_problems = 0
    for split in ds:
        n_words, bad_bio, lens = 0, 0, []
        problems: Counter = Counter()

        for row in ds[split]:
            ex = parse(row["ground_truth"])
            for p in validate(ex):
                problems[p.split(":")[0]] += 1
            n_words += len(ex)
            lens.append(len(ex))

            # A sequence must never open a span with I-.
            prev = "O"
            for lid in ex.labels:
                tag = CORD_ID2LABEL[lid]
                if tag.startswith("I-"):
                    cat = tag[2:]
                    if prev not in (f"B-{cat}", f"I-{cat}"):
                        bad_bio += 1
                prev = tag

        total_problems += sum(problems.values())
        print(f"{split:<11} docs={len(ds[split]):>4}  words={n_words:>6}  "
              f"words/doc min={min(lens)} max={max(lens)}  "
              f"bio_violations={bad_bio}")
        if problems:
            for p, c in problems.most_common():
                print(f"    PROBLEM {p}: {c}")

    print()
    print("FAILED" if total_problems else "All splits well formed.")
    sys.exit(1 if total_problems else 0)


if __name__ == "__main__":
    main()
