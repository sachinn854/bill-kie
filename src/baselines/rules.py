"""Baseline 1: Tesseract OCR plus hand-written rules.

This is the "conventional OCR-based extraction pipeline" the project is
benchmarked against, and it is deliberately a good-faith effort rather
than a straw man - a baseline that loses because it was written badly
proves nothing. It uses the same cues a person would write down: keywords
("TOTAL", "DATE"), position on the page, number formatting, and the fact
that the grand total is usually the last and largest amount printed.

What it cannot do is the point of the comparison. It has no notion of
which column a number sits in, so a quantity, a unit price and a line
total on the same row are only distinguishable by their format; it cannot
tell a subtotal from a grand total except by keyword; and every new
receipt layout needs another rule. Those are exactly the relationships a
layout-aware transformer learns instead of being told.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from src.utils.env import configure_tesseract

# Amounts, with or without thousands separators: 9.00, 1,591,600, 75000
AMOUNT = re.compile(r"\d[\d,.]*\d|\d")

DATE_PATTERNS = (
    re.compile(r"\b(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4})\b"),
    re.compile(r"\b(\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2})\b"),
    re.compile(r"\b(\d{1,2}\s+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)"
               r"[A-Z]*\s+\d{2,4})\b", re.IGNORECASE),
)

# Ordered most specific first: "SUB TOTAL" must not be read as "TOTAL".
TOTAL_CUES = ("GRAND TOTAL", "ROUNDED TOTAL", "NET TOTAL", "TOTAL AMOUNT",
              "AMOUNT DUE", "TOTAL")
SUBTOTAL_CUES = ("SUB TOTAL", "SUBTOTAL", "SUB-TOTAL")
TAX_CUES = ("GST", "VAT", "TAX", "PPN", "SST")
CASH_CUES = ("CASH", "TENDERED", "PAID")
CHANGE_CUES = ("CHANGE", "KEMBALI")
SERVICE_CUES = ("SERVICE", "SVC CHARGE", "SERV CHG")

# Lines that look like a company name sit at the top and are mostly letters.
COMPANY_STOPWORDS = ("RECEIPT", "INVOICE", "TAX INVOICE", "CASH BILL",
                     "BILL", "ORDER", "TEL", "FAX", "GST ID", "WELCOME")
COMPANY_SUFFIXES = ("SDN BHD", "BHD", "PTE LTD", "LTD", "LLC", "ENTERPRISE",
                    "TRADING", "RESTAURANT", "CAFE", "PT ", "CV ")


@dataclass
class OcrLine:
    text: str
    top: int
    height: int
    conf: float = 0.0

    @property
    def upper(self) -> str:
        return self.text.upper()


@dataclass
class RuleConfig:
    lang: str = "eng"
    # Tesseract page segmentation 6 = "a single uniform block of text",
    # which suits a receipt far better than the default 3 (full page with
    # column detection) - column detection shuffles receipt lines.
    psm: int = 6
    min_conf: float = 30.0


def ocr_lines(image, cfg: RuleConfig | None = None) -> list[OcrLine]:
    """Run Tesseract and regroup words into printed lines."""
    import pytesseract

    cfg = cfg or RuleConfig()
    configure_tesseract()

    if image.mode != "RGB":
        image = image.convert("RGB")
    data = pytesseract.image_to_data(
        image, lang=cfg.lang, config=f"--psm {cfg.psm}",
        output_type=pytesseract.Output.DICT)

    grouped: dict[tuple, list[int]] = {}
    for i, text in enumerate(data["text"]):
        if not text.strip():
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        if conf < cfg.min_conf:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        grouped.setdefault(key, []).append(i)

    lines = []
    for idxs in grouped.values():
        text = " ".join(data["text"][i].strip() for i in idxs)
        confs = [float(data["conf"][i]) for i in idxs]
        lines.append(OcrLine(
            text=text,
            top=min(data["top"][i] for i in idxs),
            height=max(data["height"][i] for i in idxs),
            conf=sum(confs) / len(confs),
        ))
    lines.sort(key=lambda l: l.top)
    return lines


def _amounts(text: str) -> list[str]:
    return [m.group(0) for m in AMOUNT.finditer(text)]


def _amount_value(raw: str) -> float | None:
    cleaned = raw.replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _keyed_amount(lines: list[OcrLine], cues: tuple[str, ...],
                  exclude: tuple[str, ...] = ()) -> str | None:
    """Last amount on the lowest line mentioning one of the cues.

    Searching bottom-up because receipts print the summary block last, and
    a cue word appearing earlier is usually a column header rather than
    the value itself.
    """
    for line in reversed(lines):
        upper = line.upper
        if any(bad in upper for bad in exclude):
            continue
        if not any(cue in upper for cue in cues):
            continue
        found = _amounts(line.text)
        if found:
            return found[-1]
    return None


def extract_date(lines: list[OcrLine]) -> str | None:
    for line in lines:
        for pattern in DATE_PATTERNS:
            m = pattern.search(line.text)
            if m:
                return m.group(1)
    return None


def extract_company(lines: list[OcrLine]) -> str | None:
    """Company name: a legal suffix if one is printed, else the top line.

    The header is where it lives, but the topmost line is often a slogan
    or a logo transcription, so a line carrying a company suffix wins.
    """
    header = lines[:8]
    for line in header:
        upper = line.upper
        if any(suffix in upper for suffix in COMPANY_SUFFIXES):
            return line.text.strip()
    for line in header:
        upper = line.upper
        if any(stop in upper for stop in COMPANY_STOPWORDS):
            continue
        letters = sum(c.isalpha() for c in line.text)
        if letters >= 4 and letters >= len(line.text) * 0.6:
            return line.text.strip()
    return None


def extract_total(lines: list[OcrLine]) -> str | None:
    """Grand total, by keyword; falling back to the largest amount.

    The fallback is what a rule-based system is reduced to when no cue is
    legible, and it is wrong whenever the cash tendered exceeds the total
    - a failure mode worth keeping visible rather than special-casing.
    """
    keyed = _keyed_amount(lines, TOTAL_CUES, exclude=SUBTOTAL_CUES)
    if keyed:
        return keyed

    best, best_value = None, float("-inf")
    for line in lines:
        for raw in _amounts(line.text):
            value = _amount_value(raw)
            if value is not None and value > best_value:
                best, best_value = raw, value
    return best


def extract_address(lines: list[OcrLine], company: str | None) -> str | None:
    """Header lines after the company name, before the first bill field."""
    out: list[str] = []
    seen_company = company is None
    for line in lines[:12]:
        upper = line.upper
        if company and line.text.strip() == company:
            seen_company = True
            continue
        if not seen_company:
            continue
        if any(stop in upper for stop in
               ("TEL", "FAX", "GST", "DOCUMENT", "DATE", "INVOICE",
                "RECEIPT", "CASHIER", "BILL")):
            break
        if any(ch.isalpha() for ch in line.text):
            out.append(line.text.strip())
    return ", ".join(out) if out else None


def extract(image, cfg: RuleConfig | None = None) -> dict[str, Any]:
    """Extract the receipt fields from an image using OCR and rules."""
    lines = ocr_lines(image, cfg)
    if not lines:
        return {}

    company = extract_company(lines)
    return {
        "company": company,
        "address": extract_address(lines, company),
        "date": extract_date(lines),
        "total": extract_total(lines),
        "subtotal": _keyed_amount(lines, SUBTOTAL_CUES),
        "tax": _keyed_amount(lines, TAX_CUES),
        "service_charge": _keyed_amount(lines, SERVICE_CUES),
        "cash": _keyed_amount(lines, CASH_CUES),
        "change": _keyed_amount(lines, CHANGE_CUES),
    }
