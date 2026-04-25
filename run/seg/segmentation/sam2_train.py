
from __future__ import annotations

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from project.config import (
    OUTPUT_ROOT,
    SAM2_BASE_MODEL_ID,
    SAM2_BATCH_SIZE,
    SAM2_DATASET_ID,
    SAM2_EPOCHS,
    SAM2_LEARNING_RATE,
)
from project.segmentation.sam2_dataset import load_kvasir_splits, collate_fn
from project.segmentation.sam2_model import apply_lora, get_device, load_base_model, load_processor
from project.utils.io_utils import write_json, write_text


def train():
    device = get_device()
    train_ds, val_ds, test_ds = load_kvasir_splits(SAM2_DATASET_ID)
    processor = load_processor(SAM2_BASE_MODEL_ID)
    model = apply_lora(load_base_model(SAM2_BASE_MODEL_ID, device=device))
    model.train()

    train_loader = DataLoader(train_ds, batch_size=SAM2_BATCH_SIZE, shuffle=True, collate_fn=lambda b: collate_fn(b, processor))
    optimizer = torch.optim.AdamW(model.parameters(), lr=SAM2_LEARNING_RATE)

    epoch_losses = []
    for epoch in range(SAM2_EPOCHS):
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{SAM2_EPOCHS}"):
            pixel_values = batch["pixel_values"].to(device)
            input_boxes = batch["input_boxes"].to(device)
            gt_masks = torch.tensor(np.stack(batch["ground_truth_masks"]), dtype=torch.float32, device=device)

            outputs = model(pixel_values=pixel_values, input_boxes=input_boxes, multimask_output=False)
            pred_masks = outputs.pred_masks.squeeze(1).squeeze(1)
            if pred_masks.ndim == 4:
                pred_masks = pred_masks[:, 0]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_masks, gt_masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())

        epoch_loss = running_loss / max(len(train_loader), 1)
        epoch_losses.append(epoch_loss)

    report = {
        "model": "SAM2.1 LoRA",
        "dataset": SAM2_DATASET_ID,
        "epochs": SAM2_EPOCHS,
        "batch_size": SAM2_BATCH_SIZE,
        "learning_rate": SAM2_LEARNING_RATE,
        "epoch_losses": epoch_losses,
        "device": device,
    }
    write_json(OUTPUT_ROOT / "reports" / "sam2_training_report.json", report)
    write_text(OUTPUT_ROOT / "reports" / "sam2_training_report.txt", "\n".join([
        "SAM2.1 training summary",
        f"Dataset: {SAM2_DATASET_ID}",
        f"Epochs: {SAM2_EPOCHS}",
        f"Batch size: {SAM2_BATCH_SIZE}",
        f"Learning rate: {SAM2_LEARNING_RATE}",
        f"Epoch losses: {epoch_losses}",
        f"Device: {device}",
    ]))
    return report


if __name__ == "__main__":
    train()
