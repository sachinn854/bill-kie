"""Compare extracted field *values* against the ground truth.

Token-level F1 cannot score a rule-based OCR pipeline, which never emits
tokens - it emits values. So the transformer and the baseline are compared
here instead, on the thing the project actually delivers: did the system
read the right total off the receipt?

Both exact and fuzzy matching are reported, and the gap between them is
the point. Exact match punishes "RM 9.00" against "9.00" and "BOOK TA .K
SDN BND" against "BOOK TA .K SDN BHD" as though the field had been missed
entirely, when in fact it was found and the OCR or the formatting differs.
Published receipt-extraction numbers vary enormously depending on which
of the two is quoted, so quoting only one is not meaningful.
"""
from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from typing import Any, Mapping, Sequence

FUZZY_THRESHOLD = 90

_PUNCT = re.compile(r"[^\w\s.]", re.UNICODE)
_SPACE = re.compile(r"\s+")
_CURRENCY = re.compile(r"^(rm|rp|usd|inr|idr|\$|₹)\s*", re.IGNORECASE)
_DATE_SEP = re.compile(r"[./\-]")


def normalise(value: Any, *, numeric: bool = False) -> str:
    """Canonical form for comparison.

    Deliberately conservative: it removes formatting noise (currency
    prefixes, thousands separators, case, punctuation) but never reorders
    or drops words, so a genuinely wrong value cannot be normalised into a
    right one.
    """
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    text = _CURRENCY.sub("", text)
    if numeric:
        text = re.sub(r"[^\d.,\-]", "", text)
        # 1,591,600 and 1591600 are the same amount; 9.00 and 9 are not
        # necessarily, so only separators are removed, never the decimal.
        if text.count(",") and text.count("."):
            text = text.replace(",", "")
        elif text.count(",") > 1:
            text = text.replace(",", "")
        elif re.match(r"^\d{1,3}(,\d{3})+$", text):
            text = text.replace(",", "")
        text = text.rstrip(".,")
        # Trailing zeros after a decimal point carry no value.
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text
    text = _PUNCT.sub(" ", text)
    return _SPACE.sub(" ", text).strip()


def normalise_date(value: Any) -> str:
    """Reduce a date to its digits, so 25/12/2018 == 25-12-2018."""
    if value is None:
        return ""
    return _DATE_SEP.sub("/", str(value).strip())


def match(predicted: Any, truth: Any, *, numeric: bool = False,
          is_date: bool = False, threshold: int = FUZZY_THRESHOLD
          ) -> tuple[bool, bool]:
    """Return (exact_match, fuzzy_match) for one field."""
    from rapidfuzz import fuzz

    if is_date:
        p, t = normalise_date(predicted), normalise_date(truth)
    else:
        p, t = normalise(predicted, numeric=numeric), normalise(truth, numeric=numeric)

    if not t:
        return (not p, not p)      # nothing expected: predicting nothing is right
    if not p:
        return (False, False)

    exact = p == t
    fuzzy = exact or fuzz.ratio(p, t) >= threshold
    return exact, fuzzy


def evaluate(predictions: Sequence[Mapping[str, Any]],
             truths: Sequence[Mapping[str, Any]],
             *,
             numeric_fields: Sequence[str] = (),
             date_fields: Sequence[str] = (),
             threshold: int = FUZZY_THRESHOLD) -> dict:
    """Per-field precision, recall and F1 under exact and fuzzy matching.

    A field counts as predicted when the system emitted a non-empty value
    and as expected when the ground truth has one, so a system that
    declines to guess is penalised on recall rather than on precision -
    which is the right way round for data entry, where a wrong total is
    worse than a blank one.
    """
    if len(predictions) != len(truths):
        raise ValueError(f"{len(predictions)} predictions vs {len(truths)} truths")

    numeric_fields = set(numeric_fields)
    date_fields = set(date_fields)
    fields = sorted({k for row in truths for k in row}
                    | {k for row in predictions for k in row})

    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tp_exact": 0, "tp_fuzzy": 0, "predicted": 0, "expected": 0})

    for pred, truth in zip(predictions, truths):
        for name in fields:
            p, t = pred.get(name), truth.get(name)
            c = counts[name]
            has_p, has_t = bool(str(p or "").strip()), bool(str(t or "").strip())
            c["predicted"] += has_p
            c["expected"] += has_t
            if not (has_p and has_t):
                continue
            exact, fuzzy = match(p, t,
                                 numeric=name in numeric_fields,
                                 is_date=name in date_fields,
                                 threshold=threshold)
            c["tp_exact"] += exact
            c["tp_fuzzy"] += fuzzy

    def prf(tp: int, predicted: int, expected: int) -> dict:
        precision = tp / predicted if predicted else 0.0
        recall = tp / expected if expected else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall else 0.0)
        return {"precision": round(precision, 4), "recall": round(recall, 4),
                "f1": round(f1, 4)}

    per_field, totals = {}, {"tp_exact": 0, "tp_fuzzy": 0,
                             "predicted": 0, "expected": 0}
    for name in fields:
        c = counts[name]
        per_field[name] = {
            "exact": prf(c["tp_exact"], c["predicted"], c["expected"]),
            "fuzzy": prf(c["tp_fuzzy"], c["predicted"], c["expected"]),
            "support": c["expected"],
        }
        for k in totals:
            totals[k] += c[k]

    return {
        "micro": {
            "exact": prf(totals["tp_exact"], totals["predicted"],
                         totals["expected"]),
            "fuzzy": prf(totals["tp_fuzzy"], totals["predicted"],
                         totals["expected"]),
        },
        "per_field": per_field,
        "n_documents": len(truths),
    }


def format_report(results: Mapping[str, Any]) -> str:
    """Readable table, for dropping straight into the report."""
    lines = [
        f"{'field':<22}{'exact P':>9}{'exact R':>9}{'exact F1':>10}"
        f"{'fuzzy F1':>10}{'n':>6}",
        "-" * 66,
    ]
    for name, row in sorted(results["per_field"].items()):
        lines.append(
            f"{name:<22}{row['exact']['precision']:>9.4f}"
            f"{row['exact']['recall']:>9.4f}{row['exact']['f1']:>10.4f}"
            f"{row['fuzzy']['f1']:>10.4f}{row['support']:>6}")
    micro = results["micro"]
    lines += [
        "-" * 66,
        f"{'MICRO':<22}{micro['exact']['precision']:>9.4f}"
        f"{micro['exact']['recall']:>9.4f}{micro['exact']['f1']:>10.4f}"
        f"{micro['fuzzy']['f1']:>10.4f}{results['n_documents']:>6}",
    ]
    return "\n".join(lines)
