
from __future__ import annotations

import numpy as np
import torch

from project.config import OUTPUT_ROOT, SAM2_BASE_MODEL_ID, SAM2_DATASET_ID, SAM2_MERGED_DIR
from project.segmentation.sam2_dataset import get_bounding_box, load_kvasir_splits
from project.segmentation.sam2_model import get_device, load_processor
from project.utils.io_utils import write_json, write_text


def infer_single_example(model, processor, item, device: str):
    image = item["image"].convert("RGB")
    gt_mask = np.array(item["mask"].convert("L"))
    gt_binary = (gt_mask > 128).astype(np.uint8)
    bbox = get_bounding_box(gt_binary)
    inputs = processor(images=image, input_boxes=[[bbox]], return_tensors="pt").to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)
        pred_mask = torch.sigmoid(outputs.pred_masks.squeeze()).cpu().numpy()
        pred_mask = (pred_mask > 0.5).astype(np.uint8)

    return image, gt_binary, pred_mask, bbox


def save_inference_summary(sample_index: int, bbox: list[float], mask_pixels: int, pred_pixels: int):
    payload = {
        "sample_index": sample_index,
        "bbox": bbox,
        "ground_truth_positive_pixels": int(mask_pixels),
        "predicted_positive_pixels": int(pred_pixels),
        "note": "Summary for a single SAM2.1 segmentation inference.",
    }
    write_json(OUTPUT_ROOT / "reports" / "sam2_inference_summary.json", payload)
    write_text(
        OUTPUT_ROOT / "reports" / "sam2_inference_summary.txt",
        "\n".join([
            "SAM2.1 inference summary",
            f"Sample index: {sample_index}",
            f"Bounding box: {bbox}",
            f"GT positive pixels: {mask_pixels}",
            f"Predicted positive pixels: {pred_pixels}",
        ]),
    )
