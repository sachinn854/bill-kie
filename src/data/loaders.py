"""Torch datasets and the collator that feeds LayoutLMv3.

Encoding happens in the collator rather than up front. Receipts are short
(CORD median 48 tokens) but the image tower is a fixed 224x224, so the
memory cost is dominated by batch size, not sequence length - padding each
batch to its own longest member rather than to a global 256 is most of the
speed difference measured in scripts/benchmark_vram.py.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from torch.utils.data import Dataset

from src.data.cord import Example, parse as parse_cord
from src.data.schema import (CORD_LABEL2ID, SROIE_LABEL2ID)
from src.data.sroie import build_example as build_sroie


class ReceiptDataset(Dataset):
    """Holds decoded examples plus their images, ready for the collator."""

    def __init__(self, rows: Sequence[Any],
                 builder: Callable[[Any], tuple[Example, Any]]):
        self.rows = rows
        self.builder = builder

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        example, _ = self.builder(row)
        image = row["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        return {
            "image": image,
            "words": example.words,
            "boxes": example.boxes,
            "labels": example.labels,
        }


def _cord_builder(row) -> tuple[Example, None]:
    return parse_cord(row["ground_truth"]), None


def build_dataset(hf_split, dataset: str) -> ReceiptDataset:
    if dataset == "cord":
        return ReceiptDataset(hf_split, _cord_builder)
    if dataset == "sroie":
        return ReceiptDataset(hf_split, build_sroie)
    raise ValueError(f"unknown dataset {dataset!r}")


def label_map(dataset: str) -> dict[str, int]:
    return CORD_LABEL2ID if dataset == "cord" else SROIE_LABEL2ID


def subsample(hf_split, n: int | None, seed: int):
    """Take n examples for the low-data study, reproducibly.

    Sampling without replacement from a shuffled index, rather than taking
    the first n, because CORD is not shuffled on disk and a prefix would
    correlate with whatever order the dataset was built in.
    """
    if n is None or n >= len(hf_split):
        return hf_split
    idx = list(range(len(hf_split)))
    random.Random(seed).shuffle(idx)
    return hf_split.select(sorted(idx[:n]))


@dataclass
class Collator:
    """Encode a batch of receipts for LayoutLMv3.

    The processor aligns word-level labels onto subword tokens, labelling
    the first subword of each word and marking continuations with -100 so
    they are ignored by both the loss and the metrics.
    """

    processor: Any
    max_length: int = 256
    pad_to_multiple_of: int | None = 8   # keeps tensor cores busy under fp16

    def __call__(self, features: list[dict]) -> dict:
        encoding = self.processor(
            [f["image"] for f in features],
            [f["words"] for f in features],
            boxes=[f["boxes"] for f in features],
            word_labels=[f["labels"] for f in features],
            truncation=True,
            padding=True,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        return dict(encoding)


def build_processor(checkpoint: str, cache_dir: str | None = None):
    """LayoutLMv3 processor with OCR disabled - we supply words and boxes.

    apply_ocr defaults to True, which would silently discard our annotated
    words and boxes and run Tesseract on the image instead, destroying the
    alignment between labels and tokens.
    """
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(
        checkpoint, apply_ocr=False, cache_dir=cache_dir
    )
