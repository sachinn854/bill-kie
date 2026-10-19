"""Turn token-level predictions into the final receipt JSON.

The model tags words; the deliverable is a structured document. Two steps
sit in between:

1. **Spans.** Consecutive B-/I- tags of one category become one field
   occurrence, carrying its text and its box.
2. **Line items.** CORD's training labels group menu words with a
   group_id, but that is annotation metadata and does not exist at
   inference time. Items have to be recovered from layout instead, by
   clustering spans on their vertical position: a name, a quantity and a
   price printed on the same row belong to the same item. This is the one
   genuinely inferential step in assembly, and the place to look first
   when line items come out scrambled.

Scalar fields (totals, tax, vendor) take the highest-confidence single
occurrence; receipts print each of those once, and a second prediction is
usually a misfire rather than a second value.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

# Two spans belong to the same printed row if their vertical centres are
# within this fraction of the 0-1000 box grid.
#
# Chosen by sweeping 0.008-0.020 against CORD's own parsed line items over
# 200 receipts, feeding in ground-truth tags so the measurement isolates
# assembly from model error. The curve is flat across that range (74-78%
# of receipts get an exactly correct item count) so this is not finely
# tuned, and should not be: fitting it harder would be fitting noise.
#
# That flatness is itself the result worth reporting. Even with perfect
# token predictions, recovering line items from vertical position alone
# gets the count exactly right on about 78% of receipts and recovers 96%
# of items. Grouping, not tagging, is the ceiling on line-item output -
# because CORD's group_id is annotation metadata that does not exist at
# inference time, so the structure has to be inferred from layout.
ROW_TOLERANCE = 0.015

# A single amount: 9, 9.00, 75,000, 1,591,600. Matching amounts rather
# than stripping non-digits matters, because a span can hold more than one
# number - a tax line often prints its rate beside its value ("11%
# 144,695"), and stripping would concatenate the two into 11144695.
AMOUNT = re.compile(r"\d[\d,.]*\d|\d")


@dataclass
class Span:
    category: str
    text: str
    box: list[int]          # 0-1000 grid, xyxy
    score: float = 1.0

    @property
    def y_centre(self) -> float:
        return (self.box[1] + self.box[3]) / 2.0


def extract_spans(words: Sequence[str], boxes: Sequence[Sequence[int]],
                  tags: Sequence[str],
                  scores: Sequence[float] | None = None) -> list[Span]:
    """Merge BIO tags into field occurrences.

    An I- tag that opens a span is treated as a B-: the model is free to
    emit invalid BIO, and dropping those tokens would silently lose
    predictions rather than scoring them.
    """
    spans: list[Span] = []
    current: list[int] = []
    current_cat: str | None = None

    def flush() -> None:
        if not current or current_cat is None:
            return
        text = " ".join(words[i] for i in current)
        xs0 = min(boxes[i][0] for i in current)
        ys0 = min(boxes[i][1] for i in current)
        xs1 = max(boxes[i][2] for i in current)
        ys1 = max(boxes[i][3] for i in current)
        score = (sum(scores[i] for i in current) / len(current)
                 if scores is not None else 1.0)
        spans.append(Span(current_cat, text, [xs0, ys0, xs1, ys1], score))

    for i, tag in enumerate(tags):
        if tag == "O":
            flush()
            current, current_cat = [], None
            continue
        prefix, category = tag[:1], tag[2:]
        if prefix == "B" or category != current_cat:
            flush()
            current, current_cat = [i], category
        else:
            current.append(i)
    flush()
    return spans


def _clean_number(text: str) -> str | None:
    """Pull the amount out of a span, ignoring currency marks and rates.

    The last match wins: where a span carries two numbers they are
    conventionally printed rate-then-value ("11% 144,695", "1 x 75,000"),
    so the value is the one at the end.
    """
    matches = AMOUNT.findall(text)
    return matches[-1].strip(",.") if matches else None


def group_line_items(spans: Iterable[Span]) -> list[dict]:
    """Cluster menu spans into items by the row they are printed on."""
    menu = [s for s in spans if s.category.startswith("menu.")]
    if not menu:
        return []

    menu.sort(key=lambda s: s.y_centre)
    rows: list[list[Span]] = [[menu[0]]]
    row_anchor = menu[0].y_centre
    for span in menu[1:]:
        # Compare against the row's anchor, not its last member. Chaining
        # off the last member lets densely printed items link into one
        # unbounded run - on a 22-item receipt that collapsed them to 13.
        if abs(span.y_centre - row_anchor) <= ROW_TOLERANCE * 1000:
            rows[-1].append(span)
        else:
            rows.append([span])
            row_anchor = span.y_centre

    field_of = {
        "menu.nm": "description", "menu.sub.nm": "description",
        "menu.cnt": "quantity", "menu.sub.cnt": "quantity",
        "menu.unitprice": "unit_price", "menu.sub.unitprice": "unit_price",
        "menu.price": "total_price", "menu.sub.price": "total_price",
        "menu.discountprice": "discount",
    }

    items: list[dict] = []
    for row in rows:
        item: dict = {}
        for span in row:
            key = field_of.get(span.category)
            if key is None or key in item:
                continue
            item[key] = (span.text if key == "description"
                         else _clean_number(span.text) or span.text)
        if not item.get("description"):
            continue
        # A row carrying only a name, with no quantity and no price, is a
        # wrapped continuation of the item above rather than a new item.
        # Long dish names routinely wrap, and treating each line as its own
        # item inflates the count.
        bare = set(item) == {"description"}
        if bare and items:
            items[-1]["description"] += " " + item["description"]
        else:
            items.append(item)
    return items


def _best(spans: Iterable[Span], category: str) -> Span | None:
    candidates = [s for s in spans if s.category == category]
    if not candidates:
        return None
    return max(candidates, key=lambda s: s.score)


def _scalar(spans: Iterable[Span], category: str,
            numeric: bool = True) -> str | None:
    span = _best(spans, category)
    if span is None:
        return None
    return (_clean_number(span.text) or span.text) if numeric else span.text


def assemble(cord_spans: Sequence[Span] = (),
             sroie_spans: Sequence[Span] = ()) -> dict:
    """Merge both models' predictions into one receipt document.

    CORD supplies line items and the totals breakdown; SROIE supplies the
    vendor block. Both predict a grand total, so SROIE's is kept alongside
    as a cross-check rather than silently discarded - a disagreement
    between them is a useful signal that one model misread the receipt.
    """
    cord = list(cord_spans)
    sroie = list(sroie_spans)

    cord_total = _scalar(cord, "total.total_price")
    sroie_total = _scalar(sroie, "total")

    document = {
        "vendor": {
            "name": _scalar(sroie, "company", numeric=False),
            "address": _scalar(sroie, "address", numeric=False),
        },
        "transaction": {
            "date": _scalar(sroie, "date", numeric=False),
        },
        "line_items": group_line_items(cord),
        "totals": {
            "subtotal": _scalar(cord, "sub_total.subtotal_price"),
            "tax": _scalar(cord, "sub_total.tax_price"),
            "service_charge": _scalar(cord, "sub_total.service_price"),
            "discount": _scalar(cord, "sub_total.discount_price"),
            "grand_total": cord_total or sroie_total,
        },
        "payment": {
            "cash": _scalar(cord, "total.cashprice"),
            "card": _scalar(cord, "total.creditcardprice"),
            "e_money": _scalar(cord, "total.emoneyprice"),
            "change": _scalar(cord, "total.changeprice"),
        },
    }

    if cord_total and sroie_total and cord_total != sroie_total:
        document["warnings"] = [
            f"models disagree on the total: CORD read {cord_total!r}, "
            f"SROIE read {sroie_total!r}"
        ]
    return document


def to_json(document: dict, indent: int = 2) -> str:
    return json.dumps(document, indent=indent, ensure_ascii=False)
