"""Corpus-level measures for text-prompted binary segmentation: per-record IoU and Dice at a stated threshold,
their means (macro), the pixel-pooled IoU (micro), pixel precision and recall, and two non-adapted baselines
(empty mask, full-image mask).

Every rate is computed over the supplied records at the supplied threshold; the sigmoid is uncalibrated and no
dispersion is estimated (one seeded split of one sample gives one number)."""
# ruff: noqa: E501  -- adaptation-contract lines are kept at the fleet width

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

METRIC_DEFINITIONS = {
    "miou": "mean over records of the intersection-over-union between the thresholded mask and the reference mask (macro IoU; an empty prediction against a non-empty reference scores 0)",
    "iou_micro": "total intersection pixels over total union pixels across all records (micro IoU; large masks weigh more)",
    "dice": "mean over records of 2·|A∩B| / (|A| + |B|)",
    "pixel_precision": "total true-positive pixels over total predicted pixels",
    "pixel_recall": "total true-positive pixels over total reference pixels",
    "baselines": "empty = no pixel predicted (IoU 0 by construction); full = every pixel predicted (IoU = the reference's area fraction)",
}


def mask_metrics(prediction: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """IoU, Dice, precision and recall of one boolean prediction against one boolean reference."""
    a = np.asarray(prediction, dtype=bool)
    b = np.asarray(reference, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"prediction shape {a.shape} != reference shape {b.shape}")
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    pred, ref = int(a.sum()), int(b.sum())
    return {
        "iou": inter / union if union else 0.0,
        "dice": 2 * inter / (pred + ref) if (pred + ref) else 0.0,
        "precision": inter / pred if pred else 0.0,
        "recall": inter / ref if ref else 0.0,
        "intersection": inter,
        "union": union,
        "predicted": pred,
        "reference": ref,
    }


def segmentation_metrics(predictions: Sequence[np.ndarray], records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Score one boolean mask per record against the record's `mask`."""
    if len(predictions) != len(records):
        raise ValueError(f"{len(predictions)} predictions for {len(records)} records")
    if not records:
        raise ValueError("no records to score")
    rows = []
    for prediction, record in zip(predictions, records, strict=True):
        row = mask_metrics(prediction, record["mask"])
        row["id"] = record["id"]
        row["prompt"] = record["prompt"]
        rows.append(row)
    inter = sum(r["intersection"] for r in rows)
    union = sum(r["union"] for r in rows)
    pred = sum(r["predicted"] for r in rows)
    ref = sum(r["reference"] for r in rows)
    return {
        "n": len(rows),
        "miou": sum(r["iou"] for r in rows) / len(rows),
        "iou_micro": inter / union if union else 0.0,
        "dice": sum(r["dice"] for r in rows) / len(rows),
        "pixel_precision": inter / pred if pred else 0.0,
        "pixel_recall": inter / ref if ref else 0.0,
        "predicted_pixels": pred,
        "reference_pixels": ref,
        "rows": rows,
        "definitions": dict(METRIC_DEFINITIONS),
    }


def empty_baseline(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Predict no pixel for every record: IoU, Dice and recall 0 by construction."""
    out = segmentation_metrics([np.zeros(np.asarray(r["mask"]).shape, dtype=bool) for r in records], records)
    out["baseline"] = "empty mask (no pixel predicted)"
    return out


def full_baseline(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Predict every pixel for every record: IoU equals the reference's area fraction, recall 1."""
    out = segmentation_metrics([np.ones(np.asarray(r["mask"]).shape, dtype=bool) for r in records], records)
    out["baseline"] = "full-image mask (every pixel predicted)"
    return out
