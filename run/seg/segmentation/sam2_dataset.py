
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
from datasets import load_dataset


@dataclass
class SegmentationBatch:
    pixel_values: Any
    input_boxes: Any
    ground_truth_masks: List[np.ndarray]


def get_bounding_box(mask: np.ndarray) -> list[float]:
    """Extract [xmin, ymin, xmax, ymax] as Python floats."""
    y_indices, x_indices = np.where(mask > 0)
    if len(x_indices) == 0:
        return [0.0, 0.0, 10.0, 10.0]
    return [
        float(np.min(x_indices)),
        float(np.min(y_indices)),
        float(np.max(x_indices)),
        float(np.max(y_indices)),
    ]


def load_kvasir_splits(dataset_id: str):
    dataset = load_dataset(dataset_id)
    return dataset["train"], dataset["validation"], dataset["test"]


def collate_fn(batch, processor):
    images, masks, boxes = [], [], []
    for item in batch:
        img = item["image"].convert("RGB")
        mask = np.array(item["mask"].convert("L"))
        mask = (mask > 128).astype(np.uint8)
        bbox = get_bounding_box(mask)
        images.append(img)
        masks.append(mask)
        boxes.append([bbox])

    inputs = processor(images=images, input_boxes=boxes, return_tensors="pt")
    return {
        "pixel_values": inputs["pixel_values"],
        "input_boxes": inputs["input_boxes"],
        "ground_truth_masks": masks,
    }
