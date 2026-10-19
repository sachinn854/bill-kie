"""Convert CORD-v2 into LayoutLMv3 token-classification features.

CORD stores annotations as line segments, each with a category and a list of
words carrying quadrilateral coordinates. LayoutLMv3 wants per-word axis-
aligned boxes normalised to a 0-1000 grid, plus BIO labels.

Receipt photos are not always axis-aligned, so a quad is not a rectangle.
We take its bounding rectangle, which is what the layout embedding expects.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Iterator

from src.data.schema import CORD_LABEL2ID

BOX_SCALE = 1000


@dataclass
class Example:
    words: list[str] = field(default_factory=list)
    boxes: list[list[int]] = field(default_factory=list)   # 0-1000, xyxy
    labels: list[int] = field(default_factory=list)
    group_ids: list[int] = field(default_factory=list)      # line-item grouping
    image_size: tuple[int, int] = (0, 0)

    def __len__(self) -> int:
        return len(self.words)


def quad_to_box(quad: dict, width: int, height: int) -> list[int]:
    """Bounding rectangle of a CORD quad, normalised to the 0-1000 grid.

    Coordinates are clamped: a few CORD annotations extend a pixel or two
    past the image edge, and LayoutLMv3 raises on boxes outside 0-1000.
    """
    xs = [quad[f"x{i}"] for i in (1, 2, 3, 4)]
    ys = [quad[f"y{i}"] for i in (1, 2, 3, 4)]

    def norm(v: float, size: int) -> int:
        return max(0, min(BOX_SCALE, int(BOX_SCALE * v / size)))

    x0, x1 = norm(min(xs), width), norm(max(xs), width)
    y0, y1 = norm(min(ys), height), norm(max(ys), height)
    # Degenerate boxes (zero width/height) confuse the layout embedding;
    # give them a single-unit extent rather than dropping the word.
    if x1 <= x0:
        x1 = min(BOX_SCALE, x0 + 1)
    if y1 <= y0:
        y1 = min(BOX_SCALE, y0 + 1)
    return [x0, y0, x1, y1]


def parse(ground_truth: str, label2id: dict[str, int] | None = None) -> Example:
    """Turn one CORD ground_truth record into an Example."""
    label2id = label2id or CORD_LABEL2ID
    gt = json.loads(ground_truth)
    width = gt["meta"]["image_size"]["width"]
    height = gt["meta"]["image_size"]["height"]

    ex = Example(image_size=(width, height))
    for line in gt["valid_line"]:
        category = line["category"]
        gid = line.get("group_id", -1)
        first = True
        for w in line["words"]:
            text = w["text"].strip()
            if not text:
                continue
            prefix = "B" if first else "I"
            tag = f"{prefix}-{category}"
            # An unseen category would otherwise be silently mislabelled as O.
            if tag not in label2id:
                raise KeyError(f"category {category!r} missing from schema")
            ex.words.append(text)
            ex.boxes.append(quad_to_box(w["quad"], width, height))
            ex.labels.append(label2id[tag])
            ex.group_ids.append(gid)
            first = False
    return ex


def iter_examples(split_ds) -> Iterator[Example]:
    for row in split_ds:
        yield parse(row["ground_truth"])


def validate(ex: Example) -> list[str]:
    """Return a list of problems; empty means the example is well formed."""
    problems = []
    n = len(ex.words)
    if not (n == len(ex.boxes) == len(ex.labels) == len(ex.group_ids)):
        problems.append("field lengths disagree")
    for b in ex.boxes:
        if len(b) != 4:
            problems.append(f"box has {len(b)} coords")
            break
        if not all(0 <= v <= BOX_SCALE for v in b):
            problems.append(f"box out of 0-{BOX_SCALE} range: {b}")
            break
        if b[2] <= b[0] or b[3] <= b[1]:
            problems.append(f"degenerate box: {b}")
            break
    return problems
