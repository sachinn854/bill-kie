"""Convert SROIE into LayoutLMv3 token-classification features.

SROIE is harder to prepare than CORD because it never labels words. It gives
OCR text segments with boxes, plus the four entity *values* as free strings:

    {"company": "BOOK TA .K (TAMAN DAYA) SDN BHD",
     "date": "25/12/2018",
     "address": "NO.53 55,57 & 59, JALAN SAGU 18, ... JOHOR.",
     "total": "9.00"}

Turning those into per-word BIO tags means locating each value on the page,
and every field fails a naive exact match for a different reason:

* company - the annotation is the true company name, the page is OCR of a
            noisy scan. One real example reads "SDN BND" on the page against
            "SDN BHD" in the annotation, and the spacing differs too.
* address - routinely spans several segments, so no single segment contains
            it and matching must run across segment boundaries.
* total   - the literal string occurs several times (a line item, the
            subtotal, the cash tendered), so finding it is not enough; the
            right occurrence still has to be chosen.
* date    - sits inside a longer segment ("25/12/2018 8:13:39 PM"), so
            tagging whole segments would drag the time in with it.

Matching therefore runs in stages, from strict to lenient, and every example
records which stage succeeded. Those counts are the label-noise budget for
the SROIE model and belong in the report: a field matched only at the fuzzy
stage is a field whose supervision may be wrong.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from src.data.cord import BOX_SCALE, Example
from src.data.schema import SROIE_CATEGORIES, SROIE_LABEL2ID

STAGES = ("exact", "alnum", "multipart", "fuzzy")
FUZZY_THRESHOLD = 85

# Stricter bar for matching one piece of a multi-part value: a short
# fragment such as a street number matches unrelated text far too easily
# at the whole-value threshold.
PIECE_FUZZY_THRESHOLD = 90

# Fields whose value is a comma-joined concatenation of several printed
# lines, which must be located piece by piece. See _multipart_span.
MULTIPART_FIELDS = frozenset({"address"})

# When two matched pieces of a multi-part value sit close together, the words
# between them belong to the same block and are tagged too. Kept small so a
# stray match cannot swallow unrelated text.
MAX_GAP_WORDS = 3

# Words printed next to a receipt's grand total. Used only to break ties when
# a value such as "9.00" occurs in several places on the page.
TOTAL_CUES = ("TOTAL", "AMOUNT", "DUE", "NETT", "NET")

# An address can run long; cap the span search so it stays linear in practice.
MAX_SPAN_WORDS = 40


@dataclass
class MatchReport:
    """Which stage matched each field, or None if nothing did."""

    stages: dict[str, str | None] = field(default_factory=dict)
    candidates: dict[str, int] = field(default_factory=dict)


def normalise(s: str, *, alnum_only: bool = False) -> str:
    s = unicodedata.normalize("NFKC", s).upper()
    if alnum_only:
        return re.sub(r"[^A-Z0-9]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def split_segment(text: str, box: list[int]) -> list[tuple[str, list[int]]]:
    """Split an OCR segment into words, interpolating a box for each.

    Boxes are apportioned horizontally by character offset. Receipt segments
    are single-line and close to monospaced, so this approximates well, and
    it is far better than giving every word of a segment the same box - which
    would leave 'date' indistinguishable from the time printed beside it.
    """
    words = text.split()
    if len(words) <= 1:
        stripped = text.strip()
        return [(stripped, list(box))] if stripped else []

    x0, y0, x1, y1 = box
    span = max(1, x1 - x0)
    n_chars = max(1, len(text))
    out: list[tuple[str, list[int]]] = []
    cursor = 0
    for w in words:
        start = text.index(w, cursor)
        end = start + len(w)
        cursor = end
        wx0 = x0 + int(span * start / n_chars)
        wx1 = x0 + int(span * end / n_chars)
        out.append((w, [wx0, y0, max(wx0 + 1, wx1), y1]))
    return out


def _norm_boxes(boxes: list[list[int]], width: int, height: int) -> list[list[int]]:
    """Scale pixel boxes onto the 0-1000 grid LayoutLMv3 expects."""

    def n(v: int, size: int) -> int:
        return max(0, min(BOX_SCALE, int(BOX_SCALE * v / max(1, size))))

    out = []
    for x0, y0, x1, y1 in boxes:
        bx0, bx1 = n(x0, width), n(x1, width)
        by0, by1 = n(y0, height), n(y1, height)
        out.append([bx0, by0, max(bx0 + 1, bx1), max(by0 + 1, by1)])
    return out


def _spans(words: list[str], target: str, *, alnum: bool) -> list[tuple[int, int]]:
    """Every contiguous word span whose concatenation equals the target."""
    want = normalise(target, alnum_only=alnum)
    if not want:
        return []
    joiner = "" if alnum else " "
    found: list[tuple[int, int]] = []
    n = len(words)
    for i in range(n):
        acc: list[str] = []
        for j in range(i, min(n, i + MAX_SPAN_WORDS)):
            acc.append(normalise(words[j], alnum_only=alnum))
            joined = joiner.join(acc)
            if joined == want:
                found.append((i, j + 1))
                break
            if len(joined) > len(want):
                break
    return found


def _fuzzy_span(words: list[str], target: str, *, start_from: int = 0,
                threshold: int = FUZZY_THRESHOLD) -> tuple[int, int] | None:
    """Best approximate span, for values the OCR got slightly wrong."""
    from rapidfuzz import fuzz

    want = normalise(target, alnum_only=True)
    if not want:
        return None
    best: tuple[int, int] | None = None
    best_score = threshold
    n = len(words)
    for i in range(start_from, n):
        acc = ""
        for j in range(i, min(n, i + MAX_SPAN_WORDS)):
            acc += normalise(words[j], alnum_only=True)
            if len(acc) > len(want) * 1.6:
                break
            score = fuzz.ratio(acc, want)
            if score > best_score:
                best_score, best = score, (i, j + 1)
    return best


def _multipart_span(words: list[str], target: str) -> tuple[int, int] | None:
    """Locate a value that the page prints across several lines.

    Addresses fail whole-string matching almost always, and not because the
    matcher is too strict: SROIE's OCR word list is frequently incomplete
    relative to the Task 3 annotation, which was transcribed from the image.
    One real example annotates "NO.53 55,57 & 59, JALAN SAGU 18, TAMAN DAYA,
    81100 JOHOR BAHRU, JOHOR." while the page words jump straight from
    "NO.53 55" to "TAMAN DAYA" - the middle of the address is simply absent
    from the transcription, so no contiguous span can ever equal the value.

    So match each comma-separated piece on its own, walking left to right and
    keeping each match after the previous one, then tag from the first hit to
    the last. Pieces the OCR never captured are skipped instead of failing
    the whole field, and short gaps between hits are included because the
    address is one printed block.
    """
    pieces = [p.strip() for p in target.split(",") if p.strip()]
    if len(pieces) < 2:
        return None

    hits: list[tuple[int, int]] = []
    cursor = 0
    for piece in pieces:
        cands = [s for s in _spans(words, piece, alnum=True) if s[0] >= cursor]
        chosen = min(cands, key=lambda se: se[0]) if cands else None
        if chosen is None:
            # Most pieces that fail here are present but misread, not absent:
            # the page says "JALAN SAGO 18" where the annotation says "JALAN
            # SAGU 18". Fall back to an approximate match for the piece, held
            # to a stricter threshold than whole-value matching because a
            # short piece is easy to match against the wrong text by chance.
            chosen = _fuzzy_span(words, piece, start_from=cursor,
                                 threshold=PIECE_FUZZY_THRESHOLD)
        if chosen is None:
            continue
        hits.append(chosen)
        cursor = chosen[1]

    if len(hits) < 2:
        return None

    # Reject a scatter: real address pieces sit in one block, so consecutive
    # hits should not be far apart.
    for (_, prev_end), (next_start, _) in zip(hits, hits[1:]):
        if next_start - prev_end > MAX_GAP_WORDS:
            return None

    return hits[0][0], hits[-1][1]


def _pick(spans: list[tuple[int, int]], words: list[str],
          boxes: list[list[int]], field_name: str) -> tuple[int, int]:
    """Choose among several matching spans.

    Only 'total' is badly ambiguous: the same number can appear as a line
    item, the subtotal, the cash tendered and the grand total. Prefer a span
    printed next to a total cue word, then the lowest on the page, since the
    grand total comes last.
    """
    if len(spans) == 1:
        return spans[0]

    if field_name == "total":
        cued = []
        for s, e in spans:
            window = " ".join(words[max(0, s - 4):s]).upper()
            if any(cue in window for cue in TOTAL_CUES):
                cued.append((s, e))
        if cued:
            spans = cued
        return max(spans, key=lambda se: boxes[se[0]][1])

    if field_name == "company":
        # The company name is printed in the header.
        return min(spans, key=lambda se: boxes[se[0]][1])

    return spans[0]


def build_example(row: dict) -> tuple[Example, MatchReport]:
    """Build a labelled Example from one SROIE row."""
    width = row["image_size"]["width"]
    height = row["image_size"]["height"]

    words: list[str] = []
    raw_boxes: list[list[int]] = []
    for seg_text, seg_box in zip(row["words"], row["bboxes"]):
        for w, b in split_segment(seg_text, seg_box):
            words.append(w)
            raw_boxes.append(b)

    boxes = _norm_boxes(raw_boxes, width, height)
    labels = [SROIE_LABEL2ID["O"]] * len(words)
    report = MatchReport()

    for name in SROIE_CATEGORIES:
        value = (row.get("entities") or {}).get(name)
        if not value:
            report.stages[name] = None
            continue

        span: tuple[int, int] | None = None
        stage: str | None = None
        for st in ("exact", "alnum"):
            cands = _spans(words, value, alnum=(st == "alnum"))
            report.candidates[name] = max(report.candidates.get(name, 0),
                                          len(cands))
            if cands:
                span = _pick(cands, words, boxes, name)
                stage = st
                break
        if span is None and name in MULTIPART_FIELDS:
            span = _multipart_span(words, value)
            stage = "multipart"
        if span is None:
            span = _fuzzy_span(words, value)
            stage = "fuzzy"

        if span is None:
            report.stages[name] = None
            continue

        # Never overwrite an already-tagged word: an address overlapping the
        # company line would otherwise silently clobber it.
        s, e = span
        if any(labels[k] != 0 for k in range(s, e)):
            report.stages[name] = f"{stage}:overlap"
            continue

        labels[s] = SROIE_LABEL2ID[f"B-{name}"]
        for k in range(s + 1, e):
            labels[k] = SROIE_LABEL2ID[f"I-{name}"]
        report.stages[name] = stage

    ex = Example(words=words, boxes=boxes, labels=labels,
                 group_ids=[-1] * len(words), image_size=(width, height))
    return ex, report
