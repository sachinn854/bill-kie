"""Measure real peak VRAM and step time for LayoutLMv3 fine-tuning.

Runs actual forward+backward+optimizer steps with synthetic batches shaped
like CORD samples, then reports what fits. Paper estimates are not good
enough to plan around: run this on the machine you will actually train on.

Two things this script exists to catch, both of which bit us on a 4 GB
laptop GPU:

1. LayoutLMv3 does NOT support gradient checkpointing (HF sets
   supports_gradient_checkpointing = False). The usual "just enable
   checkpointing" escape hatch for small GPUs is unavailable here.

2. On Windows (WDDM driver), CUDA silently oversubscribes VRAM into system
   RAM instead of raising OutOfMemoryError. The run "succeeds" while
   thrashing over PCIe - we measured an 11x slowdown. A config is therefore
   only usable if peak memory stays under the physical limit; "it ran" is
   not the test. SPILL below marks these.

    python scripts/benchmark_vram.py
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from src.utils.env import gpu_memory_gb, summary

MODEL_ID = "microsoft/layoutlmv3-base"
NUM_LABELS = 30
WARMUP_STEPS = 2
MEASURE_STEPS = 5

# Fraction of physical VRAM above which we treat a run as spilling into
# system RAM rather than genuinely fitting.
SPILL_FRACTION = 0.92


def make_batch(bs: int, seq_len: int, device: str) -> dict:
    return {
        "input_ids": torch.randint(0, 1000, (bs, seq_len), device=device),
        "attention_mask": torch.ones(bs, seq_len, dtype=torch.long, device=device),
        "bbox": torch.randint(0, 1000, (bs, seq_len, 4), device=device),
        "pixel_values": torch.randn(bs, 3, 224, 224, device=device),
        "labels": torch.randint(0, NUM_LABELS, (bs, seq_len), device=device),
    }


def _cleanup(model=None) -> None:
    """Free GPU state, tolerating a wedged CUDA context after an OOM."""
    del model
    gc.collect()
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    except Exception:  # noqa: BLE001 - empty_cache itself can raise post-OOM
        pass


def try_config(bs: int, seq_len: int, fp16: bool, device: str,
               total_vram: float) -> dict:
    from transformers import LayoutLMv3ForTokenClassification

    _cleanup()
    result = {"batch_size": bs, "seq_len": seq_len, "fp16": fp16}
    model = None
    try:
        model = LayoutLMv3ForTokenClassification.from_pretrained(
            MODEL_ID, num_labels=NUM_LABELS
        ).to(device)
        model.train()

        opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
        scaler = torch.amp.GradScaler("cuda", enabled=fp16)
        batch = make_batch(bs, seq_len, device)

        def step():
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=fp16):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        for _ in range(WARMUP_STEPS):
            step()
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(MEASURE_STEPS):
            step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        spilling = peak > SPILL_FRACTION * total_vram
        result.update(
            ran=True,
            usable=not spilling,
            spilling=spilling,
            peak_vram_gb=round(peak, 2),
            sec_per_step=round(elapsed / MEASURE_STEPS, 3),
            samples_per_sec=round(bs * MEASURE_STEPS / elapsed, 1),
        )
    except torch.cuda.OutOfMemoryError:
        result.update(ran=False, usable=False, error="OOM")
    except Exception as exc:  # noqa: BLE001
        result.update(ran=False, usable=False,
                      error=f"{type(exc).__name__}: {exc}")
    finally:
        _cleanup(model)

    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-lens", type=int, nargs="+", default=[256, 512])
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--out", default="results/logs/vram_benchmark.json")
    args = ap.parse_args()

    env = summary()
    print(json.dumps(env, indent=2))
    if not torch.cuda.is_available():
        print("\nNo CUDA GPU detected - nothing to benchmark.")
        sys.exit(1)

    total = gpu_memory_gb()
    print(f"\nTotal VRAM: {total:.2f} GB")
    print("LayoutLMv3 does not support gradient checkpointing - not tested.\n")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for seq_len in args.seq_lens:
        for bs in args.batch_sizes:
            print(f"  seq={seq_len:<4} bs={bs:<2} ... ", end="", flush=True)
            r = try_config(bs, seq_len, True, "cuda", total)
            rows.append(r)
            if r["ran"] and r["usable"]:
                print(f"OK     peak {r['peak_vram_gb']:>4.2f} GB  "
                      f"{r['sec_per_step']:>6.3f} s/step  "
                      f"{r['samples_per_sec']:>5.1f} samp/s")
            elif r["ran"]:
                print(f"SPILL  peak {r['peak_vram_gb']:>4.2f} GB  "
                      f"{r['sec_per_step']:>6.3f} s/step  "
                      f"{r['samples_per_sec']:>5.1f} samp/s  "
                      f"(exceeds VRAM, thrashing to system RAM)")
            else:
                print(f"FAIL   {r['error'][:60]}")

            # Write after every config: an OOM later must not lose earlier data.
            out.write_text(json.dumps(
                {"env": env, "total_vram_gb": round(total, 2), "runs": rows},
                indent=2, default=str))

    print(f"\nSaved -> {out}")

    usable = [r for r in rows if r.get("usable")]
    if not usable:
        print("\nNothing fits natively. Training needs a bigger GPU.")
        return

    best = max(usable, key=lambda r: r["samples_per_sec"])
    print(f"\nBest usable config: seq={best['seq_len']}, bs={best['batch_size']}, "
          f"peak {best['peak_vram_gb']} GB, {best['samples_per_sec']} samples/s")

    print("\nProjected wall-clock on this machine:")
    for label, passes in (
        ("one full CORD run (800 x 20 epochs)", 800 * 20),
        ("low-data sweep (5 sizes x 3 seeds)", 132_000),
    ):
        for r in sorted(usable, key=lambda x: -x["samples_per_sec"])[:2]:
            mins = passes / r["samples_per_sec"] / 60
            unit = f"{mins:.0f} min" if mins < 90 else f"{mins/60:.1f} hr"
            print(f"  {label:38s} @ seq={r['seq_len']:<4} bs={r['batch_size']:<2} "
                  f"-> {unit}")


if __name__ == "__main__":
    main()
