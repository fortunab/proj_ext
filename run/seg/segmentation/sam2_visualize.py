
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from project.config import OUTPUT_ROOT


def _save(fig, name: str):
    path = OUTPUT_ROOT / "images" / name
    fig.savefig(path, bbox_inches="tight", dpi=180)
    plt.close(fig)
    return path


def save_mask_figure(image, pred_mask: np.ndarray, gt_mask: np.ndarray, name: str = "sam2_masks.png"):
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(image)
    ax[0].set_title("Image")
    ax[1].imshow(gt_mask, cmap="gray")
    ax[1].set_title("Ground truth")
    ax[2].imshow(pred_mask, cmap="gray")
    ax[2].set_title("Predicted mask")
    for a in ax:
        a.axis("off")
    return _save(fig, name)


def save_overlay_figure(image, pred_mask: np.ndarray, gt_mask: np.ndarray, name: str = "sam2_overlay.png"):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow(image)
    ax[0].imshow(gt_mask, alpha=0.45)
    ax[0].set_title("GT overlay")
    ax[1].imshow(image)
    ax[1].imshow(pred_mask, alpha=0.45)
    ax[1].set_title("Prediction overlay")
    for a in ax:
        a.axis("off")
    return _save(fig, name)


def save_metric_bar_chart(metrics: dict, name: str = "sam2_metric_comparison.png"):
    labels = ["mIoU", "Dice"]
    base = [metrics["base_sam2"]["miou"], metrics["base_sam2"]["dice"]]
    tuned = [metrics["sam2_lora"]["miou"], metrics["sam2_lora"]["dice"]]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width / 2, base, width, label="Base SAM2")
    ax.bar(x + width / 2, tuned, width, label="SAM2.1 LoRA")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_title("SAM2.1 segmentation metrics")
    ax.legend()
    return _save(fig, name)
