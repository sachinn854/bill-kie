"""Measure real peak VRAM and step time for LayoutLMv3 fine-tuning.

Runs actual forward+backward+optimizer steps with synthetic batches shaped
exactly like CORD samples, then reports what fits. Estimates on paper are
not good enough to plan a project around: run this on the machine you will
actually train on.

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
NUM_LABELS = 30           # CORD has 30 field tags -> 2n+1 BIO labels in practice
WARMUP_STEPS = 2
MEASURE_STEPS = 5


def make_batch(bs: int, seq_len: int, device: str) -> dict:
    """Synthetic batch matching LayoutLMv3's expected inputs."""
    return {
        "input_ids": torch.randint(0, 1000, (bs, seq_len), device=device),
        "attention_mask": torch.ones(bs, seq_len, dtype=torch.long, device=device),
        # bbox coords are on a 0-1000 normalised scale
        "bbox": torch.randint(0, 1000, (bs, seq_len, 4), device=device),
        "pixel_values": torch.randn(bs, 3, 224, 224, device=device),
        "labels": torch.randint(0, NUM_LABELS, (bs, seq_len), device=device),
    }


def try_config(bs: int, seq_len: int, grad_ckpt: bool, fp16: bool,
               device: str) -> dict:
    """Run a few real training steps. Returns measurements or an OOM marker."""
    from transformers import LayoutLMv3ForTokenClassification

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    result = {"batch_size": bs, "seq_len": seq_len,
              "grad_checkpointing": grad_ckpt, "fp16": fp16}
    model = None
    try:
        model = LayoutLMv3ForTokenClassification.from_pretrained(
            MODEL_ID, num_labels=NUM_LABELS
        ).to(device)
        if grad_ckpt:
            model.gradient_checkpointing_enable()
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

        result.update(
            ok=True,
            peak_vram_gb=round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2),
            reserved_vram_gb=round(torch.cuda.max_memory_reserved() / 1024 ** 3, 2),
            sec_per_step=round(elapsed / MEASURE_STEPS, 3),
            samples_per_sec=round(bs * MEASURE_STEPS / elapsed, 1),
        )
    except torch.cuda.OutOfMemoryError:
        result.update(ok=False, error="OOM")
    except Exception as exc:  # noqa: BLE001 - we want the message, not a trace
        result.update(ok=False, error=f"{type(exc).__name__}: {exc}")
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()

    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-lens", type=int, nargs="+", default=[256, 512],
                    help="256 ~ typical CORD receipt, 512 = worst case")
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--out", default="results/logs/vram_benchmark.json")
    args = ap.parse_args()

    print(json.dumps(summary(), indent=2))
    if not torch.cuda.is_available():
        print("\nNo CUDA GPU detected - nothing to benchmark.")
        sys.exit(1)

    total = gpu_memory_gb()
    print(f"\nTotal VRAM: {total:.2f} GB\n")

    rows = []
    for seq_len in args.seq_lens:
        for grad_ckpt in (False, True):
            for bs in args.batch_sizes:
                label = (f"seq={seq_len:<4} bs={bs:<2} "
                         f"grad_ckpt={str(grad_ckpt):<5} fp16=True")
                print(f"  testing  {label} ... ", end="", flush=True)
                r = try_config(bs, seq_len, grad_ckpt, True, "cuda")
                rows.append(r)
                if r["ok"]:
                    print(f"OK   peak {r['peak_vram_gb']:>4.2f} GB   "
                          f"{r['sec_per_step']:>5.3f} s/step   "
                          f"{r['samples_per_sec']:>5.1f} samp/s")
                else:
                    print(f"FAIL {r['error']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"env": summary(), "total_vram_gb": round(total, 2), "runs": rows},
        indent=2, default=str))
    print(f"\nSaved -> {out}")

    ok = [r for r in rows if r["ok"]]
    if ok:
        best = max(ok, key=lambda r: (r["seq_len"], r["samples_per_sec"]))
        print(f"\nFastest config that fits: bs={best['batch_size']}, "
              f"seq={best['seq_len']}, grad_ckpt={best['grad_checkpointing']}, "
              f"peak {best['peak_vram_gb']} GB, "
              f"{best['samples_per_sec']} samples/s")
        # 800 CORD training samples x 20 epochs
        est_min = 800 * 20 / best["samples_per_sec"] / 60
        print(f"Estimated full CORD run (800 samples x 20 epochs): "
              f"{est_min:.0f} min")
    else:
        print("\nNothing fit. Training will need a bigger GPU.")


if __name__ == "__main__":
    main()
