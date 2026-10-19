"""Plot the low-data curve from completed sweep runs.

    python scripts/plot_sweep.py

Draws F1 against training-set size with one point per size, a band of
+/- one standard deviation across seeds, and the individual seeds marked.
Both curves are shown - core fields and the full label set - because the
gap between them is itself a result: it separates "the model needs more
data" from "that category never appeared in a sample this small".
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.utils.env import get_paths

PATTERN = re.compile(r"sweep_(?P<dataset>\w+?)_n(?P<n>\d+)_s(?P<seed>\d+)\.json$")


def collect(logs: Path, dataset: str) -> dict[int, list[dict]]:
    runs: dict[int, list[dict]] = defaultdict(list)
    for path in sorted(logs.glob(f"sweep_{dataset}_n*_s*.json")):
        m = PATTERN.search(path.name)
        if not m:
            continue
        record = json.loads(path.read_text())
        metrics = record.get("metrics", {})
        runs[int(m.group("n"))].append({
            "seed": int(m.group("seed")),
            "f1": metrics.get("eval_f1"),
            "core_f1": metrics.get("eval_core_f1"),
            "macro_f1": metrics.get("eval_macro_f1_scoreable"),
            "minutes": record.get("train_seconds", 0) / 60,
        })
    return dict(sorted(runs.items()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cord")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = get_paths()
    runs = collect(paths.logs, args.dataset)
    if not runs:
        print(f"No sweep results found in {paths.logs}. "
              f"Run scripts/run_sweep.py first.")
        sys.exit(1)

    print(f"{'n':>5} {'seeds':>6} {'core F1':>18} {'all F1':>18} {'min':>6}")
    sizes, summary = [], {}
    for n, entries in runs.items():
        core = np.array([e["core_f1"] for e in entries if e["core_f1"] is not None])
        allf = np.array([e["f1"] for e in entries if e["f1"] is not None])
        if not core.size:
            continue
        sizes.append(n)
        summary[n] = {
            "core_mean": core.mean(), "core_std": core.std(ddof=0),
            "all_mean": allf.mean() if allf.size else np.nan,
            "all_std": allf.std(ddof=0) if allf.size else np.nan,
            "seeds": len(entries),
        }
        print(f"{n:>5} {len(entries):>6} "
              f"{core.mean():>10.4f} +/-{core.std(ddof=0):<5.4f} "
              f"{allf.mean():>10.4f} +/-{allf.std(ddof=0):<5.4f} "
              f"{np.mean([e['minutes'] for e in entries]):>6.1f}")

    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=150)
    for key, label, colour in (("core", "Core fields", "#2563eb"),
                               ("all", "All categories", "#dc2626")):
        mean = np.array([summary[n][f"{key}_mean"] for n in sizes])
        std = np.array([summary[n][f"{key}_std"] for n in sizes])
        ax.plot(sizes, mean, "o-", color=colour, label=label, zorder=3)
        ax.fill_between(sizes, mean - std, mean + std,
                        color=colour, alpha=0.15, zorder=1)

    for n, entries in runs.items():
        for e in entries:
            if e["core_f1"] is not None:
                ax.plot(n, e["core_f1"], ".", color="#2563eb",
                        alpha=0.35, markersize=5, zorder=2)

    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(n) for n in sizes])
    ax.set_xlabel("Training receipts")
    ax.set_ylabel("Entity-level F1")
    ax.set_title(f"How much annotated data does fine-tuning need?\n"
                 f"LayoutLMv3-base on {args.dataset.upper()}, "
                 f"shaded band is +/-1 SD over seeds")
    ax.grid(alpha=0.25, zorder=0)
    ax.legend(frameon=False)
    fig.tight_layout()

    out = Path(args.out) if args.out else paths.figures / f"lowdata_{args.dataset}.png"
    fig.savefig(out)
    (paths.logs / f"sweep_summary_{args.dataset}.json").write_text(
        json.dumps({str(k): v for k, v in summary.items()},
                   indent=2, default=float))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
