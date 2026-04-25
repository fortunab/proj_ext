from __future__ import annotations

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BlipForQuestionAnswering, BlipProcessor

from bionnica.config import OUTPUTS_DIR, VQA_BATCH_SIZE, VQA_EPOCHS, DEVICE
from bionnica.datasets.vqa_dataset import KvasirVQADataset
from bionnica.metrics.vqa_metrics import evaluate_exact_match
from bionnica.utils.io_utils import ensure_dir, save_text_report
from bionnica.utils.training_vqa import fine_tune_vqa


def main() -> None:
    output_dir = ensure_dir(OUTPUTS_DIR / "blip_vqa_ft")

    dataset = load_dataset("SimulaMet/Kvasir-VQA-x1")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(DEVICE)

    small_train = dataset["train"].select(range(min(2000, len(dataset["train"]))))
    small_test = dataset["test"].select(range(min(300, len(dataset["test"]))))

    train_ds = KvasirVQADataset(small_train, processor)
    test_loader = DataLoader(KvasirVQADataset(small_test, processor), batch_size=VQA_BATCH_SIZE, shuffle=False)
    train_loader = DataLoader(train_ds, batch_size=VQA_BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    epoch_logs = fine_tune_vqa(model, train_loader, optimizer, epochs=VQA_EPOCHS, device=DEVICE)
    em = evaluate_exact_match(model, processor, dataset["test"], device=DEVICE, n_samples=min(300, len(dataset["test"])))

    lines = [
        "Model: BLIP-VQA FT",
        f"DEVICE = {DEVICE}",
        f"Train samples used: {len(train_ds)}",
        f"Test samples used: {len(small_test)}",
        f"Train batches: {len(train_loader)}",
        f"Test batches: {len(test_loader)}",
        f"Epochs: {VQA_EPOCHS}",
        "",
        "Fine-tuning log:",
        *epoch_logs,
        "",
        f"Exact Match after fine-tuning: {em:.4f}",
    ]

    save_text_report(output_dir / "blip_vqa_ft_report.txt", lines)
    print(f"Saved: {output_dir / 'blip_vqa_ft_report.txt'}")


if __name__ == "__main__":
    main()
