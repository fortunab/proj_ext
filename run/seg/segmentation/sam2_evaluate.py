

from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm
from transformers import Sam2Model

from project.config import OUTPUT_ROOT, SAM2_BASE_MODEL_ID, SAM2_DATASET_ID, SAM2_MERGED_DIR
from project.segmentation.sam2_dataset import get_bounding_box, load_kvasir_splits
from project.segmentation.sam2_metrics import iou_and_dice
from project.segmentation.sam2_model import get_device, load_processor
from project.utils.io_utils import write_json, write_text


def evaluate_model(model, dataset, processor, device: str):
    ious, dices = [], []
    with torch.no_grad():
        for item in tqdm(dataset, desc="Evaluating"):
            gt = (np.array(item["mask"].convert("L")) > 128).astype(np.uint8)
            y, x = np.where(gt > 0)
            if len(x) == 0:
                continue
            bbox = [[float(np.min(x)), float(np.min(y)), float(np.max(x)), float(np.max(y))]]
            inputs = processor(images=item["image"].convert("RGB"), input_boxes=[bbox], return_tensors="pt").to(device)
            outputs = model(**inputs, multimask_output=False)
            pred = (torch.sigmoid(outputs.pred_masks.squeeze()) > 0.5).cpu().numpy().astype(np.uint8)
            iou, dice = iou_and_dice(pred, gt)
            ious.append(iou)
            dices.append(dice)
    return float(np.mean(ious)), float(np.mean(dices))


def compare_base_and_finetuned(base_model_path: str | None = None, tuned_model_path: str | None = None):
    device = get_device()
    _, _, test_ds = load_kvasir_splits(SAM2_DATASET_ID)
    processor = load_processor(SAM2_BASE_MODEL_ID)
    base_model = Sam2Model.from_pretrained(base_model_path or SAM2_BASE_MODEL_ID).to(device).eval()
    tuned_model = Sam2Model.from_pretrained(tuned_model_path or SAM2_MERGED_DIR).to(device).eval()

    base_iou, base_dice = evaluate_model(base_model, test_ds, processor, device)
    tuned_iou, tuned_dice = evaluate_model(tuned_model, test_ds, processor, device)

    payload = {
        "base_sam2": {"miou": base_iou, "dice": base_dice},
        "sam2_lora": {"miou": tuned_iou, "dice": tuned_dice},
        "gain_percent": {
            "miou": ((tuned_iou - base_iou) / base_iou) * 100 if base_iou else 0.0,
            "dice": ((tuned_dice - base_dice) / base_dice) * 100 if base_dice else 0.0,
        },
    }
    write_json(OUTPUT_ROOT / "reports" / "sam2_eval_metrics.json", payload)
    write_text(
        OUTPUT_ROOT / "reports" / "sam2_eval_metrics.txt",
        "\n".join([
            "SAM2.1 evaluation summary",
            f"Base SAM2 mIoU: {base_iou:.4f}",
            f"Base SAM2 Dice: {base_dice:.4f}",
            f"LoRA SAM2 mIoU: {tuned_iou:.4f}",
            f"LoRA SAM2 Dice: {tuned_dice:.4f}",
            f"mIoU gain (%): {payload['gain_percent']['miou']:+.2f}",
            f"Dice gain (%): {payload['gain_percent']['dice']:+.2f}",
        ]),
    )
    return payload
