
"""Segmentation metrics for SAM2.1."""

from __future__ import annotations

import numpy as np


def iou_and_dice(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    iou = intersection / union if union > 0 else 1.0
    dice = (2 * intersection) / (pred.sum() + target.sum()) if (pred.sum() + target.sum()) > 0 else 1.0
    return float(iou), float(dice)
