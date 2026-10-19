"""Entity-level metrics for receipt key-information extraction.

Three numbers are reported, and they answer different questions:

* **micro F1 over all categories** - the headline. Published CORD results
  are entity-level micro F1 over the full label set, so this is the only
  figure comparable with the literature.
* **macro F1 excluding unscoreable categories** - the per-class view. Five
  CORD categories never appear in the test split, so their F1 is undefined
  rather than zero; averaging them in would silently score 5 of 29 classes
  at 0 and understate the model by roughly 17%. They are excluded and
  listed, not quietly dropped.
* **micro F1 over core fields** - used for the low-data curve. On the full
  label set a drop at n=50 confounds "needs more data" with "this class
  never appeared in the sample", which is not a learning result.

Everything is computed with seqeval in IOB2 mode, so a prediction counts
only when the whole span and its type are right - partial overlap scores
nothing, which is the correct standard for field extraction.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

IGNORE_INDEX = -100


def decode(predictions: np.ndarray, label_ids: np.ndarray,
           id2label: dict[int, str]) -> tuple[list[list[str]], list[list[str]]]:
    """Turn logits and label ids into aligned tag sequences.

    Subword positions carry IGNORE_INDEX: LayoutLMv3 labels only the first
    subword of each word, and scoring the continuations would count a single
    word several times and inflate the result.
    """
    if predictions.ndim == 3:
        predictions = predictions.argmax(-1)

    pred_tags, true_tags = [], []
    for pred_row, label_row in zip(predictions, label_ids):
        keep = label_row != IGNORE_INDEX
        pred_tags.append([id2label[int(p)] for p in pred_row[keep]])
        true_tags.append([id2label[int(t)] for t in label_row[keep]])
    return pred_tags, true_tags


def _filter_to(tags: list[list[str]], keep: frozenset[str]) -> list[list[str]]:
    """Blank out every tag whose category is not in keep."""
    out = []
    for row in tags:
        out.append([t if (t != "O" and t[2:] in keep) else "O" for t in row])
    return out


def compute(pred_tags: Sequence[Sequence[str]],
            true_tags: Sequence[Sequence[str]],
            *,
            unscoreable: Iterable[str] = (),
            core_fields: Iterable[str] = ()) -> dict:
    """Entity-level metrics. Keys are flat so they drop straight into logs."""
    from seqeval.metrics import (classification_report, f1_score,
                                 precision_score, recall_score)
    from seqeval.scheme import IOB2

    pred_tags = [list(r) for r in pred_tags]
    true_tags = [list(r) for r in true_tags]
    unscoreable = frozenset(unscoreable)
    core_fields = frozenset(core_fields)

    kw = {"mode": "strict", "scheme": IOB2, "zero_division": 0}
    results = {
        "precision": precision_score(true_tags, pred_tags, average="micro", **kw),
        "recall": recall_score(true_tags, pred_tags, average="micro", **kw),
        "f1": f1_score(true_tags, pred_tags, average="micro", **kw),
    }

    report = classification_report(true_tags, pred_tags, output_dict=True, **kw)
    per_field = {k: v for k, v in report.items()
                 if k not in ("micro avg", "macro avg", "weighted avg")}
    results["per_field"] = {
        k: {"precision": round(v["precision"], 4),
            "recall": round(v["recall"], 4),
            "f1": round(v["f1-score"], 4),
            "support": int(v["support"])}
        for k, v in per_field.items()
    }

    # Macro over categories that can actually be scored. A category with no
    # support in the reference is undefined, not zero.
    scoreable = [v["f1-score"] for k, v in per_field.items()
                 if k not in unscoreable and v["support"] > 0]
    results["macro_f1_scoreable"] = (
        float(np.mean(scoreable)) if scoreable else 0.0)
    results["n_scoreable_categories"] = len(scoreable)
    results["excluded_categories"] = sorted(
        k for k in per_field
        if k in unscoreable or per_field[k]["support"] == 0)

    if core_fields:
        core_pred = _filter_to(pred_tags, core_fields)
        core_true = _filter_to(true_tags, core_fields)
        results["core_f1"] = f1_score(core_true, core_pred,
                                      average="micro", **kw)
        results["core_precision"] = precision_score(core_true, core_pred,
                                                    average="micro", **kw)
        results["core_recall"] = recall_score(core_true, core_pred,
                                              average="micro", **kw)

    return results


def make_compute_metrics(id2label: dict[int, str], *,
                         unscoreable: Iterable[str] = (),
                         core_fields: Iterable[str] = ()):
    """Build the compute_metrics callable HuggingFace Trainer expects."""
    unscoreable = frozenset(unscoreable)
    core_fields = frozenset(core_fields)

    def compute_metrics(eval_pred) -> dict:
        predictions, label_ids = eval_pred
        pred_tags, true_tags = decode(predictions, label_ids, id2label)
        out = compute(pred_tags, true_tags,
                      unscoreable=unscoreable, core_fields=core_fields)
        # Trainer wants scalars for logging and best-model selection; the
        # nested breakdown is kept for the final report only.
        scalars = {k: v for k, v in out.items()
                   if isinstance(v, (int, float))}
        scalars["_detail"] = out
        return scalars

    return compute_metrics
