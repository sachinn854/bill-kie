# Transformer-Based Key Information Extraction from Bill and Receipt Images

Course project — IIITDM

| Name | Roll No. |
|---|---|
| Sachin Kumar Yadav | CS23B1042 |
| Anurag Chaubey | CS23B1052 |

## Problem

OCR reads the text on a receipt but loses the structure: which string is the
vendor, which numbers belong to the same line item, which total is *the*
total. This project fine-tunes a multimodal transformer that reads text,
layout and image together, and emits structured JSON.

## Approach

| Stage | What |
|---|---|
| Baseline 1 | Tesseract OCR + regex / keyword-proximity rules |
| Baseline 2 | BERT on OCR text only — **no layout** (ablation: does spatial information actually matter?) |
| Main model | LayoutLMv3-base fine-tuned for token classification (BIO) |
| Comparison | Donut (OCR-free), inference-only from the pretrained CORD checkpoint |

Baseline 2 is the one that makes the result interpretable. Without it, a gain
over Baseline 1 cannot be attributed to layout rather than to simply using a
transformer.

## Datasets

| Dataset | Size | Fields | Role |
|---|---|---|---|
| [CORD-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2) | 800/100/100 | 30 tags incl. line items | primary |
| [SROIE](https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE) | 626/347 | company, date, address, total | cross-dataset generalisation |

SROIE's original annotations contain known errors (dates, addresses, company
names). Use a corrected release and document which one.

## Research question (novelty)

**How much annotated data does transfer learning actually need here?**
F1 is measured at n = 50, 100, 200, 400, 800 training receipts, 3 seeds each.
The practical question behind it: a small business digitising its own receipt
format cannot annotate thousands of samples — where does the curve flatten?

## Evaluation

Field-level precision / recall / F1, reported under **both** exact and fuzzy
matching. The gap between them separates field-identification errors from OCR
and formatting noise — published receipt-KIE numbers vary wildly depending on
which one is reported, so reporting only one is not meaningful.

## Setup

```bash
uv venv --python 3.11 .venv
.venv/Scripts/activate                       # Windows
uv pip install torch==2.9.1 torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt
```

Tesseract must be installed separately (Windows: UB Mannheim build).

Check the machine before training:

```bash
python scripts/benchmark_vram.py            # what fits, and how fast
python -m src.utils.env                     # platform, paths, batch config
```

### Running on Kaggle

The code detects its platform and resolves paths accordingly, so no edits are
needed. In a notebook with **Internet = ON**:

```python
!git clone https://github.com/<user>/bill-kie.git
%cd bill-kie
!pip install -q -r requirements.txt
!python -m src.train --config configs/cord_layoutlmv3.yaml
```

Effective batch size is pinned to 16 on every machine via gradient
accumulation, so a laptop run and a Kaggle run are directly comparable.

## Licensing

LayoutLMv3 weights are CC-BY-NC-SA 4.0 (non-commercial). Donut is MIT.
CORD and SROIE are CC-BY-4.0.
