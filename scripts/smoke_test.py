"""Exercise the full encoding path on CPU, without training.

The processor/collator interface is the most error-prone part of a
LayoutLMv3 pipeline - wrong box scaling, labels misaligned against
subwords, or apply_ocr silently throwing away the annotated words. All of
that fails identically on CPU, so it is worth catching here rather than
ten minutes into a GPU run.

    python scripts/smoke_test.py
    python scripts/smoke_test.py --dataset sroie
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from datasets import load_dataset

from src.data.loaders import Collator, build_dataset, build_processor
from src.eval.metrics import decode
from src.train import DATASETS
from src.utils.env import get_paths

CHECKPOINT = "microsoft/layoutlmv3-base"
IGNORE_INDEX = -100


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="cord")
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    spec = DATASETS[args.dataset]
    paths = get_paths()
    failures: list[str] = []

    print(f"dataset   {args.dataset}  ({spec['hf_id']})")
    ds = load_dataset(spec["hf_id"], cache_dir=str(paths.cache))
    data = build_dataset(ds["train"].select(range(args.batch_size)),
                         args.dataset)

    processor = build_processor(CHECKPOINT, cache_dir=str(paths.cache))
    collator = Collator(processor, max_length=256)

    features = [data[i] for i in range(len(data))]
    print(f"words/doc {[len(f['words']) for f in features]}")

    batch = collator(features)
    print("\nencoded tensors:")
    for k, v in batch.items():
        print(f"  {k:<16} {tuple(v.shape)}  {v.dtype}")

    expected = {"input_ids", "attention_mask", "bbox", "pixel_values", "labels"}
    missing = expected - set(batch)
    if missing:
        failures.append(f"missing keys from the encoding: {sorted(missing)}")

    # Boxes must sit on the 0-1000 grid; anything else means a scaling bug.
    if "bbox" in batch:
        lo, hi = int(batch["bbox"].min()), int(batch["bbox"].max())
        print(f"\nbbox range        {lo}..{hi}  (must be within 0..1000)")
        if lo < 0 or hi > 1000:
            failures.append(f"bbox outside 0..1000: {lo}..{hi}")

    if "pixel_values" in batch:
        pv = batch["pixel_values"]
        print(f"image tensor      {tuple(pv.shape)}  "
              f"range {pv.min():.2f}..{pv.max():.2f}")
        if pv.shape[-2:] != (224, 224):
            failures.append(f"unexpected image size {tuple(pv.shape[-2:])}")

    # Labels: continuations must be ignored, and real labels must survive.
    labels = batch["labels"]
    n_ignored = int((labels == IGNORE_INDEX).sum())
    n_real = int(((labels != IGNORE_INDEX) & (labels != 0)).sum())
    n_o = int((labels == 0).sum())
    print(f"\nlabels            {n_real} tagged, {n_o} O, "
          f"{n_ignored} ignored(-100)")
    if n_real == 0:
        failures.append("no tagged labels survived encoding - apply_ocr may "
                        "be discarding the annotated words")

    # Round-trip the labels back to tag strings and confirm they decode to
    # real categories rather than to padding artefacts.
    fake_logits = np.zeros((*labels.shape, len(spec["labels"])),
                           dtype=np.float32)
    _, true_tags = decode(fake_logits, labels.numpy(), spec["id2label"])
    recovered = {t for row in true_tags for t in row if t != "O"}
    print(f"decoded tags      {len(recovered)} distinct, e.g. "
          f"{sorted(recovered)[:5]}")
    if not recovered:
        failures.append("no entity tags recovered after decoding")

    # A forward pass on CPU proves the shapes line up with the model, which
    # is the last thing that can differ between here and a GPU run.
    print("\nrunning one CPU forward pass ...")
    from transformers import LayoutLMv3ForTokenClassification

    model = LayoutLMv3ForTokenClassification.from_pretrained(
        CHECKPOINT, num_labels=len(spec["labels"]),
        cache_dir=str(paths.cache))
    model.eval()
    with torch.no_grad():
        out = model(**batch)
    print(f"  logits {tuple(out.logits.shape)}   loss {out.loss.item():.4f}")
    if not np.isfinite(out.loss.item()):
        failures.append("loss is not finite")

    print()
    if failures:
        for f in failures:
            print(f"FAIL  {f}")
        sys.exit(1)
    print("All checks passed - the pipeline is ready to train.")


if __name__ == "__main__":
    main()
