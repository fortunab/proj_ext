from __future__ import annotations

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BlipForQuestionAnswering, BlipProcessor

from bionnica.config import OUTPUTS_DIR, VQA_BATCH_SIZE, VQA_EPOCHS, DEVICE
from bionnica.datasets.vqa_dataset import KvasirVQADataset
from bionnica.metrics.vqa_metrics import evaluate_exact_match
from bionnica.utils.io_utils import ensure_dir, save_text_report


def main() -> None:
    output_dir = ensure_dir(OUTPUTS_DIR / "vqa")

    dataset = load_dataset("SimulaMet/Kvasir-VQA-x1")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(DEVICE)

    small_train = dataset["train"].select(range(min(2000, len(dataset["train"]))))
    small_test = dataset["test"].select(range(min(300, len(dataset["test"]))))

    train_ds = KvasirVQADataset(small_train, processor)
    test_ds = KvasirVQADataset(small_test, processor)

    train_loader = DataLoader(train_ds, batch_size=VQA_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=VQA_BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    epoch_logs = []

    for epoch in range(VQA_EPOCHS):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(train_loader), 1)
        epoch_logs.append(f"Epoch {epoch + 1}: loss = {avg_loss:.4f}")

    em = evaluate_exact_match(model, processor, dataset["test"], device=DEVICE, n_samples=100)

    lines = [
        f"DEVICE = {DEVICE}",
        f"Train samples used: {len(train_ds)}",
        f"Test samples used: {len(test_ds)}",
        f"Train batches: {len(train_loader)}",
        f"Test batches: {len(test_loader)}",
        f"Epochs: {VQA_EPOCHS}",
        "",
        "Training log:",
        *epoch_logs,
        "",
        f"Exact Match (first 100 test samples): {em:.4f}",
    ]

    save_text_report(output_dir / "vqa_training_report.txt", lines)
    print(f"Saved: {output_dir / 'vqa_training_report.txt'}")


if __name__ == "__main__":
    main()
