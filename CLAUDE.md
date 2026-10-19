# CLAUDE.md

Working notes for this repository. Read this before changing anything.

## What this is

A course project (IIITDM): transformer-based key information extraction
from bill and receipt images. A pre-trained multimodal transformer is
fine-tuned to read a receipt photo and emit structured JSON, and is
benchmarked against a conventional OCR pipeline.

Team: Sachin Kumar Yadav (CS23B1042), Anurag Chaubey (CS23B1052).

The abstract promises four fields: **vendor name, date, itemised entries,
and total amount**. Everything else is a bonus and should not be allowed
to delay those four.

## Conventions

- **Everything in the repo is written in English** - code, comments,
  docstrings, README, commit messages. The user is spoken to in Hinglish
  in chat, but none of that belongs in a file.
- Measure before deciding. Several planning assumptions in this project
  turned out to be wrong, and each was caught by running something rather
  than by reasoning about it. Benchmarks and dataset profiles live in
  `scripts/` and write their results to `results/logs/*.json`.
- Nothing hardcodes a path or a device. `src/utils/env.py` resolves both.

## Current status

Data layer complete and validated. Model layer not yet written.

| Written | Not yet written |
|---|---|
| `src/utils/env.py` | `src/baselines/rules.py` |
| `src/data/schema.py` | `src/baselines/text_only.py` |
| `src/data/cord.py` | `src/assemble.py` |
| `src/data/sroie.py` | `scripts/run_sweep.py` |
| `src/data/loaders.py` | |
| `src/eval/metrics.py` | |
| `src/train.py` | |
| `scripts/benchmark_vram.py` | |
| `scripts/analyse_lengths.py` | |
| `scripts/validate_preprocessing.py` | |
| `scripts/validate_sroie.py` | |
| `scripts/smoke_test.py` | |

The training path is verified end to end on CPU by
`scripts/smoke_test.py`, for both datasets: boxes land on the 0-1000
grid, labels survive subword alignment, and a forward pass produces a
finite loss.

**No model has been trained yet.** No checkpoints exist.

## Settled decisions

These were decided with the user. Do not silently revisit them.

1. **Two models, one JSON.** CORD and SROIE are complementary, not
   overlapping. CORD has line items and the totals breakdown but no
   vendor, date or address; SROIE has exactly those and no line items.
   Their label sets intersect only on the total, so one model is trained
   per dataset and the outputs are merged (`schema.UNIFIED_SCHEMA`).
2. **Novelty is a low-data study**: F1 at n = 50/100/200/400/800 training
   receipts, three seeds each. Cross-dataset generalisation was
   considered and rejected; it would not have worked anyway, given the
   label sets barely overlap.
3. **Evaluation**: keep all 29 CORD categories and report micro-F1 as the
   headline number, for comparability with published CORD results. Plot
   the low-data curve on `CORD_CORE_FIELDS` only.
4. **Baselines**: a Tesseract + regex pipeline, and a text-only BERT. The
   second is the ablation that shows layout is what helps - without it, a
   gain over the rule-based baseline cannot be attributed to layout
   rather than to merely using a transformer.

## Open question

**SROIE address aligns at only ~65%.** This is a defect in the dataset,
not in the matcher: the mirror's OCR word list is frequently missing the
middle of the address. Confirmed by opening the image - the text is
plainly printed on the receipt, and running Tesseract on it recovers the
missing span. So re-OCR would fix it, at the cost of using our own boxes
instead of the dataset's.

Not urgent: address is not one of the four promised fields, and the three
fields SROIE does supply (company, date, total) align at 98-100%.

## Hardware budget (measured, not estimated)

RTX 3050 Laptop, 4 GB VRAM; 16 GB system RAM.

| Config | Peak VRAM | Throughput |
|---|---|---|
| seq 96, batch 4 | 2.65 GB | 19.0 samples/s |
| seq 256, batch 4 | 3.65 GB | 9.4 samples/s |
| batch 8 (any length) | >4 GB | spills, unusable |

A full CORD run is ~14 min; the whole low-data sweep ~2 hr. **Local
training is comfortable.** Kaggle is a fallback, not a requirement.

### Two traps specific to this setup

1. **LayoutLMv3 does not support gradient checkpointing.**
   `supports_gradient_checkpointing` is `False`, so the usual
   activation-memory escape hatch for small GPUs does not exist here.
2. **Exceeding VRAM does not raise OutOfMemoryError on Windows.** The
   WDDM driver spills into system RAM and the run continues, silently:
   batch 8 reported a 5.68 GB peak on a 4 GB card and ran 11x slower per
   step. This looks like success, which makes it worse than an OOM.
   `benchmark_vram.py` flags any config above 92% of physical VRAM.

Memory is dominated by the fixed 224x224 image tower rather than by text
length, so **batch size, not sequence length, is the binding constraint**.

## Working with the user

**Do not start training runs, sweeps, or other GPU/RAM-heavy jobs without
an explicit go-ahead**, even when the plan that includes them is already
approved. The user frees RAM before letting anything heavy run. Short
CPU-only work (downloads, parsing, analysis) is fine unannounced.

## Datasets

Both are public on Hugging Face; no login or registration needed. They
download on first use into `.cache/` (gitignored, ~2.8 GB).

| Dataset | HF id | Splits | Licence |
|---|---|---|---|
| CORD-v2 | `naver-clova-ix/cord-v2` | 800 / 100 / 100 | CC-BY-4.0 |
| SROIE | `jsdnrs/ICDAR2019-SROIE` | 626 / 361 | CC-BY-4.0 |

The pre-tokenised CORD mirrors (`nielsr/cord-layoutlmv3` and friends) are
script-based datasets and **no longer load under `datasets` 4.x**. Use the
canonical source and `src/data/cord.py`.

SROIE is a community mirror; the official release is behind registration
at the ICDAR RRC portal.

### What the profiling found

- CORD receipts are short: median 48 tokens, p99 223, max 296. A 256 cap
  truncates 0.1% of train and nothing in val/test.
- Truncation is **not** a uniform loss. The words a cap drops are
  `total.total_price`, `total.creditcardprice`, `tax_price`,
  `changeprice` - totals sit at the bottom of a receipt, so a length cap
  removes the most important fields first. A 128 cap would truncate 4.8%.
- CORD has a severe head/tail: `menu.nm` is 27.8% of all training words
  while the bottom ten categories together are 1.62%.
- **Five CORD categories are absent from the test split**
  (`CORD_ABSENT_FROM_TEST`), so their F1 is undefined and a naive macro
  average silently scores 5 of 29 classes at zero, understating the model
  by ~17%. `menu.itemsubtotal` has 1 training and 6 test instances.
- **CORD has no `O` class.** Its `valid_line` lists only annotated
  words, so every token the CORD model ever sees belongs to some field
  and it never learns to reject anything. The smoke test shows this
  directly: a CORD batch decodes to 235 tagged tokens and 0 `O`, where
  the same-sized SROIE batch has 319 `O`. On a real receipt, where OCR
  also returns headers, footers and barcode junk, the CORD model will
  assign those to some field. State this as a limitation; it is a
  property of the dataset, not a bug in the pipeline.
- SROIE labels entity *values*, not words, so labels must be recovered by
  string matching. Match rates are in `results/logs/sroie_alignment.json`
  and are the label-noise budget for that model - quote them in the
  report rather than leaving them implicit.

## Environment

Python 3.11 (matches Kaggle), torch 2.9.1+cu128, transformers 4.57.6.

```bash
uv venv --python 3.11 .venv
uv pip install torch==2.9.1 torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt
```

Tesseract 5.4.0 is installed at `C:\Program Files\Tesseract-OCR\`, with
`eng` and `osd`. The Windows installer does not add it to PATH, so call
`env.configure_tesseract()` before using pytesseract rather than relying
on `shutil.which`.

Run `python -m src.utils.env` to print the resolved platform, paths,
device, batch config and Tesseract location.

### Kaggle

`src/utils/env.py` detects the platform, so no edits are needed. In a
notebook with **Internet = ON**:

```python
!git clone https://github.com/<user>/bill-kie.git
%cd bill-kie
!pip install -q -r requirements.txt
```

Effective batch size is pinned to 16 on every machine via gradient
accumulation, so a 4 GB laptop run and a 16 GB cloud run stay directly
comparable. Changing it would change the optimisation trajectory and
invalidate the comparison.

## Licensing

LayoutLMv3 weights are **CC-BY-NC-SA 4.0 (non-commercial)** - acceptable
for an academic project, but it must be disclosed in the report. Donut is
MIT. Both datasets are CC-BY-4.0.
