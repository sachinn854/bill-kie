"""Run the low-data study: F1 against training-set size, with error bars.

    python scripts/run_sweep.py --dry-run      # show the plan, run nothing
    python scripts/run_sweep.py
    python scripts/run_sweep.py --sizes 50 100 --seeds 42

Each (size, seed) is a separate training run launched as a subprocess, so
one crash cannot take the sweep down with it and a partial sweep can be
resumed - completed runs are skipped unless --force is given.

Two choices worth knowing about:

* Small training sets get more epochs (see EPOCHS_BY_SIZE). Holding epochs
  constant would give n=50 a sixteenth of the gradient updates that n=800
  gets, so the curve would be measuring training budget as much as data,
  and would overstate how much data is actually needed.
* The curve is plotted on core-field F1 rather than on the full label set.
  At n=50 several of CORD's 29 categories simply do not occur, so their
  zero contribution would read as "needs more data" when it really means
  "never appeared in the sample".
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.env import get_paths

SIZES = (50, 100, 200, 400, 800)
SEEDS = (42, 1337, 2024)

# More epochs at small n, so every point gets a comparable number of
# gradient updates rather than a budget proportional to its data.
EPOCHS_BY_SIZE = {50: 60, 100: 50, 200: 40, 400: 30, 800: 20}
DEFAULT_EPOCHS = 20


def run_name(dataset: str, size: int, seed: int) -> str:
    return f"sweep_{dataset}_n{size}_s{seed}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="cord", choices=("cord", "sroie"))
    ap.add_argument("--config", default="configs/cord_layoutlmv3.yaml")
    ap.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--group-by-length", action="store_true", default=True)
    ap.add_argument("--force", action="store_true",
                    help="re-run points that already have a result file")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    paths = get_paths()
    jobs = [(n, s) for n in sorted(args.sizes) for s in args.seeds]

    pending = []
    for n, seed in jobs:
        out = paths.logs / f"{run_name(args.dataset, n, seed)}.json"
        if out.exists() and not args.force:
            continue
        pending.append((n, seed))

    print(f"sweep over {len(jobs)} points "
          f"({len(args.sizes)} sizes x {len(args.seeds)} seeds)")
    print(f"already done: {len(jobs) - len(pending)}   to run: {len(pending)}")
    for n, seed in pending:
        print(f"  n={n:<4} seed={seed:<5} epochs={EPOCHS_BY_SIZE.get(n, DEFAULT_EPOCHS)}")

    if args.dry_run:
        print("\n--dry-run: nothing launched.")
        return
    if not pending:
        print("\nNothing to do.")
        return

    started = time.time()
    failures = []
    for i, (n, seed) in enumerate(pending, 1):
        name = run_name(args.dataset, n, seed)
        cmd = [
            sys.executable, "-m", "src.train",
            "--config", args.config,
            "--dataset", args.dataset,
            "--train-size", str(n),
            "--seed", str(seed),
            "--epochs", str(EPOCHS_BY_SIZE.get(n, DEFAULT_EPOCHS)),
            "--name", name,
            "--num-workers", str(args.num_workers),
            # Checkpoints are not needed: the sweep keeps metrics, not models.
            "--no-save",
        ]
        if args.group_by_length:
            cmd.append("--group-by-length")

        print(f"\n[{i}/{len(pending)}] {name}")
        result = subprocess.run(cmd, cwd=str(paths.root))
        if result.returncode != 0:
            print(f"  FAILED (exit {result.returncode})")
            failures.append(name)

    elapsed = (time.time() - started) / 60
    print(f"\nsweep finished in {elapsed:.0f} min")
    if failures:
        print(f"{len(failures)} runs failed: {failures}")
        print("Re-run this script to retry only those.")
    else:
        print("Now run: python scripts/plot_sweep.py")


if __name__ == "__main__":
    main()
