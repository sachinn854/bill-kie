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
                 builder: Callable[[Any], tuple[Example, Any]],
                 lengths: Sequence[int] | None = None):
        self.rows = rows
        self.builder = builder
        # Word counts, for length-grouped batching. Approximate (words, not
        # subwords) and cheap, which is all the sampler needs.
        self.lengths = list(lengths) if lengths is not None else None

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


def _cheap_lengths(hf_split, dataset: str) -> list[int]:
    """Word count per document, without running the full builder.

    Used only to group similar-length receipts into the same batch. SROIE's
    builder runs fuzzy string alignment per document, which is far too slow
    to invoke just to measure a length, so count the raw segments instead.
    """
    if dataset == "cord":
        import json
        out = []
        for gt in hf_split["ground_truth"]:
            lines = json.loads(gt)["valid_line"]
            out.append(sum(len(line["words"]) for line in lines))
        return out
    return [sum(len(seg.split()) for seg in words)
            for words in hf_split["words"]]


def build_dataset(hf_split, dataset: str,
                  with_lengths: bool = False) -> ReceiptDataset:
    lengths = _cheap_lengths(hf_split, dataset) if with_lengths else None
    if dataset == "cord":
        return ReceiptDataset(hf_split, _cord_builder, lengths)
    if dataset == "sroie":
        return ReceiptDataset(hf_split, build_sroie, lengths)
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


@dataclass
class TextOnlyCollator:
    """Encode the same receipts for a plain text model - no boxes, no image.

    This is the ablation that makes the main result interpretable. It sees
    exactly the same words in exactly the same reading order as
    LayoutLMv3, and differs only in being denied the layout and the
    picture, so the gap between the two is attributable to those and not
    to a different tokenisation, split or label set.

    Labels are aligned by word id, tagging the first subword of each word
    and marking continuations -100, which is what the LayoutLMv3 processor
    does internally.
    """

    tokenizer: Any
    max_length: int = 256
    pad_to_multiple_of: int | None = 8
    ignore_index: int = -100

    def __call__(self, features: list[dict]) -> dict:
        encoding = self.tokenizer(
            [f["words"] for f in features],
            is_split_into_words=True,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        import torch

        all_labels = []
        for i, feature in enumerate(features):
            word_ids = encoding.word_ids(batch_index=i)
            labels, previous = [], None
            for word_id in word_ids:
                if word_id is None:
                    labels.append(self.ignore_index)
                elif word_id != previous:
                    labels.append(feature["labels"][word_id])
                else:
                    labels.append(self.ignore_index)
                previous = word_id
            all_labels.append(labels)

        batch = {k: v for k, v in encoding.items()}
        batch["labels"] = torch.tensor(all_labels, dtype=torch.long)
        return batch


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
