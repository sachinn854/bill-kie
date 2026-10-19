"""Check whether max_seq_length=256 is safe for CORD, and inventory labels.

Truncating a receipt is not a uniform loss: totals sit at the bottom, so a
length cap silently removes the single most important field first. This
measures the real token-length distribution and reports what each candidate
cap would actually cost, before we commit to one.

    python scripts/analyse_lengths.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.utils.env import get_paths

# Canonical source. The pre-tokenised mirrors (nielsr/cord-layoutlmv3 etc.)
# are script-based datasets, unsupported since datasets 4.x.
DATASET_ID = "naver-clova-ix/cord-v2"
CANDIDATES = (128, 256, 384, 512)
CAP = 256


def iter_words(ground_truth: str):
    """Yield (text, category) per word from a CORD ground_truth record.

    CORD stores one entry per line segment, each holding several words that
    share a category. group_id (unused here) links the segments of a single
    line item, which the JSON assembler will need later.
    """
    for line in json.loads(ground_truth)["valid_line"]:
        cat = line["category"]
        for w in line["words"]:
            text = w["text"].strip()
            if text:
                yield text, cat


def main() -> None:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    paths = get_paths()
    print(f"Loading {DATASET_ID} ...")
    ds = load_dataset(DATASET_ID, cache_dir=str(paths.cache))
    print({k: len(v) for k, v in ds.items()})

    tok = AutoTokenizer.from_pretrained(
        "microsoft/layoutlmv3-base", cache_dir=str(paths.cache)
    )

    report: dict = {"dataset": DATASET_ID, "cap_examined": CAP, "splits": {}}
    all_categories: Counter = Counter()

    for split in ds:
        lengths: list[int] = []
        tail_loss: Counter = Counter()

        for ex in ds[split]:
            pairs = list(iter_words(ex["ground_truth"]))
            all_categories.update(c for _, c in pairs)

            n_sub = [len(tok.tokenize(t, add_special_tokens=False)) or 1
                     for t, _ in pairs]
            lengths.append(sum(n_sub) + 2)      # <s> ... </s>

            cum = 2
            for (_text, cat), n in zip(pairs, n_sub):
                cum += n
                if cum > CAP:
                    tail_loss[cat] += 1

        arr = np.array(lengths)
        stats = {
            "n": int(arr.size),
            "mean": round(float(arr.mean()), 1),
            "median": int(np.median(arr)),
            "p90": int(np.percentile(arr, 90)),
            "p95": int(np.percentile(arr, 95)),
            "p99": int(np.percentile(arr, 99)),
            "max": int(arr.max()),
            "pct_truncated": {str(c): round(float((arr > c).mean() * 100), 1)
                              for c in CANDIDATES},
        }
        report["splits"][split] = stats

        print()
        print(f"--- {split} (n={stats['n']}) ---")
        print(f"  mean {stats['mean']}  median {stats['median']}  "
              f"p90 {stats['p90']}  p95 {stats['p95']}  "
              f"p99 {stats['p99']}  max {stats['max']}")
        print("  truncated at cap:  " + "   ".join(
            f"{c}->{stats['pct_truncated'][str(c)]}%" for c in CANDIDATES))

        if tail_loss:
            stats[f"words_lost_at_{CAP}"] = dict(tail_loss.most_common())
            print(f"  words lost past {CAP}: {sum(tail_loss.values())}"
                  f"   top categories:")
            for name, cnt in tail_loss.most_common(8):
                print(f"    {name:<28} {cnt}")
        else:
            print(f"  nothing lost at cap={CAP}")

    report["categories"] = dict(all_categories.most_common())
    print()
    print(f"=== {len(all_categories)} distinct categories ===")
    for name, cnt in all_categories.most_common():
        print(f"  {name:<30} {cnt}")

    out = paths.logs / "token_length_analysis.json"
    out.write_text(json.dumps(report, indent=2))
    print()
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
