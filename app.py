"""Streamlit app for Receipt Key Information Extraction.

Upload a receipt photo. The app runs Tesseract OCR to get words and boxes,
feeds them to both fine-tuned LayoutLMv3 checkpoints, assembles the predicted
tags into the unified receipt schema and shows the values as a table.
"""
from __future__ import annotations

import json
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd
import pytesseract
import streamlit as st
import torch
from PIL import Image, ImageFilter, ImageOps
from transformers import LayoutLMv3ForTokenClassification

from src.assemble import AMOUNT, assemble, extract_spans
from src.data.cord import BOX_SCALE
from src.data.loaders import build_processor
from src.utils.env import configure_tesseract, get_paths

SCORE_THRESHOLD = 30
TESSERACT_PSM = 6
MAX_LENGTH = 256

SOURCE_BY_SECTION = {
    "vendor": "SROIE",
    "transaction": "SROIE",
    "line_items": "CORD",
    "totals": "CORD / SROIE",
    "payment": "CORD",
}

ITEM_NUMERIC_FIELDS = ("description", "quantity", "unit_price",
                      "total_price", "discount")


def checkpoint_for(name: str) -> Path:
    root = get_paths().checkpoints / name
    candidates = sorted(root.glob("checkpoint-*"),
                        key=lambda p: int(p.name.split("-")[-1]))
    if not candidates:
        raise FileNotFoundError(f"no checkpoint found under {root}")
    return candidates[-1]


@st.cache_resource(show_spinner="Loading fine-tuned models...")
def load_models():
    paths = get_paths()
    processor = build_processor("microsoft/layoutlmv3-base",
                                cache_dir=str(paths.cache))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = {}
    for name in ("cord_layoutlmv3_base", "sroie_layoutlmv3_base"):
        model = LayoutLMv3ForTokenClassification.from_pretrained(
            str(checkpoint_for(name)), cache_dir=str(paths.cache))
        model.to(device)
        model.eval()
        models[name] = model
    return processor, models, device


def ocr_variants(image: Image.Image) -> list[Image.Image]:
    enlarged = image.resize((image.width * 3, image.height * 3),
                            Image.Resampling.LANCZOS)
    contrasted = ImageOps.autocontrast(enlarged.convert("L")).filter(
        ImageFilter.SHARPEN).convert("RGB")
    return [image, enlarged, contrasted]


def ocr_words(image: Image.Image) -> tuple[list[str], list[list[int]]]:
    configure_tesseract()
    best = _ocr_once(image)
    if len(best[0]) >= 20:
        return best
    for variant in ocr_variants(image)[1:]:
        words, boxes = _ocr_once(variant)
        if len(words) > len(best[0]):
            best = (words, boxes)
    return best


def _ocr_once(image: Image.Image) -> tuple[list[str], list[list[int]]]:
    data = pytesseract.image_to_data(
        image, lang="eng", config=f"--psm {TESSERACT_PSM}",
        output_type=pytesseract.Output.DICT)
    width, height = image.size

    def scale(value: int, size: int) -> int:
        return max(0, min(BOX_SCALE, int(BOX_SCALE * value / max(1, size))))

    words, boxes = [], []
    for i, text in enumerate(data["text"]):
        text = text.strip()
        if not text:
            continue
        try:
            confidence = float(data["conf"][i])
        except (TypeError, ValueError):
            continue
        if confidence < SCORE_THRESHOLD:
            continue
        x0 = scale(data["left"][i], width)
        y0 = scale(data["top"][i], height)
        x1 = scale(data["left"][i] + data["width"][i], width)
        y1 = scale(data["top"][i] + data["height"][i], height)
        words.append(text)
        boxes.append([x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)])
    return words, boxes


def predict_tags(model, processor, device, image, words,
                 boxes) -> list[str]:
    encoding = processor(images=[image], text=[words], boxes=[boxes],
                         truncation=True, padding=True,
                         max_length=MAX_LENGTH, return_tensors="pt")
    word_ids = encoding.word_ids(0)
    inputs = {k: v.to(device) for k, v in encoding.items()}
    with torch.no_grad():
        logits = model(**inputs).logits[0].argmax(-1).cpu().numpy()

    tags = ["O"] * len(words)
    assigned = set()
    for token_index, word_index in enumerate(word_ids):
        if word_index is None or word_index in assigned:
            continue
        if word_index >= len(words):
            continue
        tags[word_index] = model.config.id2label[int(logits[token_index])]
        assigned.add(word_index)
    return tags


def extract(image: Image.Image) -> tuple[dict, dict, list[str], int]:
    words, boxes = ocr_words(image)
    if not words:
        return {}, [], 0

    processor, models, device = load_models()
    cord_tags = predict_tags(models["cord_layoutlmv3_base"], processor,
                             device, image, words, boxes)
    sroie_tags = predict_tags(models["sroie_layoutlmv3_base"], processor,
                              device, image, words, boxes)
    cord_spans = extract_spans(words, boxes, cord_tags)
    sroie_spans = extract_spans(words, boxes, sroie_tags)
    raw = assemble(cord_spans=cord_spans, sroie_spans=sroie_spans)
    return clean_document(raw, cord_spans, sroie_spans), raw, words, len(words)


def to_rows(document: dict) -> pd.DataFrame:
    rows = []
    for section, values in document.items():
        if section == "warnings":
            continue
        if isinstance(values, list):
            for index, item in enumerate(values, 1):
                for key, entry in item.items():
                    rows.append((section, f"item #{index} {key}",
                                 entry if entry is not None else "-",
                                 SOURCE_BY_SECTION.get(section, "")))
            continue
        for field, value in values.items():
            if isinstance(value, list):
                for index, item in enumerate(value, 1):
                    for key, entry in item.items():
                        rows.append((section, f"{field} #{index} {key}",
                                     entry if entry is not None else "-",
                                     SOURCE_BY_SECTION.get(section, "")))
            else:
                rows.append((section, field,
                             value if value is not None else "-",
                             SOURCE_BY_SECTION.get(section, "")))
    return pd.DataFrame(rows, columns=["Section", "Field", "Value", "Source"])


def clean_amount(value) -> str | None:
    if value is None:
        return None
    matches = AMOUNT.findall(str(value))
    if not matches:
        return None
    try:
        amount = Decimal(matches[-1].replace(",", "").strip(",."))
    except InvalidOperation:
        return None
    if amount <= 0:
        return None
    return f"{amount:.2f}"


def clean_quantity(value) -> str | None:
    matches = AMOUNT.findall(str(value or ""))
    if not matches:
        return None
    try:
        return str(int(Decimal(matches[0].replace(",", ""))))
    except (InvalidOperation, ValueError):
        return None


def span_amount(spans, category: str) -> str | None:
    for span in spans:
        if span.category == category:
            return clean_amount(span.text)
    return None


def clean_document(raw: dict, cord_spans, sroie_spans) -> dict:
    document = json.loads(json.dumps(raw))

    for field in ("subtotal", "tax", "service_charge", "discount"):
        document["totals"][field] = clean_amount(document["totals"].get(field))

    address_spans = sorted(
        (s for s in sroie_spans if s.category == "address"),
        key=lambda s: (s.box[1], s.box[0]))
    if address_spans:
        document["vendor"]["address"] = ", ".join(
            s.text for s in address_spans) or None

    cord_total = span_amount(cord_spans, "total.total_price")
    sroie_total = span_amount(sroie_spans, "total")
    document["totals"]["grand_total"] = cord_total or sroie_total
    warnings = list(document.get("warnings", []))
    if cord_total and sroie_total and cord_total != sroie_total:
        warnings.append(
            f"total cross-check: CORD {cord_total}, SROIE {sroie_total}")
    if warnings:
        document["warnings"] = warnings
    else:
        document.pop("warnings", None)

    for field in ("cash", "card", "e_money", "change"):
        document["payment"][field] = clean_amount(
            document["payment"].get(field))

    items = []
    for item in document.get("line_items", []):
        cleaned = {}
        for key in ITEM_NUMERIC_FIELDS:
            if key == "description":
                cleaned[key] = item.get(key)
            elif key == "quantity":
                cleaned[key] = clean_quantity(item.get(key))
            else:
                cleaned[key] = clean_amount(item.get(key))
        if not cleaned["description"]:
            continue
        if not any(cleaned[k] for k in ("quantity", "unit_price",
                                       "total_price", "discount")):
            continue
        items.append(cleaned)
    document["line_items"] = items
    return document


def main() -> None:
    st.set_page_config(page_title="Receipt KIE", layout="wide")
    st.title("Receipt Key Information Extraction")
    st.caption("Tesseract OCR + two fine-tuned LayoutLMv3 models, merged into "
               "one structured receipt.")

    uploaded = st.file_uploader("Upload a bill or receipt photo",
                                type=["png", "jpg", "jpeg", "webp"])
    if uploaded is None:
        st.info("Upload an image to extract vendor, date, line items and totals.")
        return

    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded receipt", use_container_width=True)

    if not st.button("Extract fields", type="primary"):
        return

    with st.spinner("Running OCR and both fine-tuned models..."):
        document, raw, words, n_words = extract(image)

    if not document:
        st.warning("No words were detected by OCR. Try a sharper, "
                   "better-lit photo.")
        return

    st.caption(f"OCR read {n_words} words from the image "
               f"(best of three enhanced variants).")
    if n_words < 20:
        st.warning("OCR found very few words. A sharper, better-lit photo "
                   "will give better results.")
    st.subheader("Extracted fields")
    st.dataframe(to_rows(document), use_container_width=True,
                 hide_index=True)

    for warning in document.get("warnings", []):
        st.warning(warning)

    with st.expander("Raw model output (before cleanup)"):
        st.json(raw)

    st.download_button("Download JSON",
                       data=json.dumps(document, indent=2, ensure_ascii=False),
                       file_name="receipt.json", mime="application/json")
    st.download_button("Download CSV",
                       data=to_rows(document).to_csv(index=False),
                       file_name="receipt.csv", mime="text/csv")

    with st.sidebar:
        st.header("Known limitations")
        st.markdown(
            "- CORD annotations have no background class, so the CORD model "
            "assigns every OCR word to some field, including noise.\n"
            "- Multi-line addresses are currently reduced to one span by the "
            "assembler.\n"
            "- Line items are grouped from vertical position alone.\n"
            "- LayoutLMv3 weights are CC-BY-NC-SA 4.0 (academic use only)."
        )


if __name__ == "__main__":
    main()
