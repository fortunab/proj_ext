from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from bionnica.config import HYPERKVASIR_DIR, OUTPUTS_DIR, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, DEVICE
from bionnica.datasets.classification_dataset import ImageFolderFromCSV
from bionnica.metrics.classification_metrics import count_total_params, count_trainable_params
from bionnica.models.transfer_classifier import TransferClassifier
from bionnica.utils.io_utils import ensure_dir, save_text_report, save_json
from bionnica.utils.training_classification import fit_classification_model, predict_labels, confusion_details
from sklearn.metrics import roc_auc_score


def main() -> None:
    output_dir = ensure_dir(OUTPUTS_DIR / "effnet_b0_ft")

    train_csv = HYPERKVASIR_DIR / "train.csv"
    val_csv = HYPERKVASIR_DIR / "val.csv"
    test_csv = HYPERKVASIR_DIR / "test.csv"
    images_dir = HYPERKVASIR_DIR / "images"

    train_ds = ImageFolderFromCSV(train_csv, images_dir, train=True, image_size=IMAGE_SIZE)
    val_ds = ImageFolderFromCSV(val_csv, images_dir, train=False, image_size=IMAGE_SIZE)
    test_ds = ImageFolderFromCSV(test_csv, images_dir, train=False, image_size=IMAGE_SIZE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = TransferClassifier(backbone_name="efficientnet_b0", num_classes=2, freeze_backbone=False).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

    history = fit_classification_model(model, train_loader, val_loader, criterion, optimizer, epochs=5, device=DEVICE)
    y_true, y_pred, y_prob = predict_labels(model, test_loader, device=DEVICE)
    details = confusion_details(y_true, y_pred)
    details["auc"] = roc_auc_score(y_true, y_prob) if len(set(y_true)) == 2 else None

    lines = [
        "Model: EffNet-B0 FT",
        f"DEVICE = {DEVICE}",
        f"Train samples: {len(train_ds)}",
        f"Validation samples: {len(val_ds)}",
        f"Test samples: {len(test_ds)}",
        f"Total parameters: {count_total_params(model)}",
        f"Trainable parameters: {count_trainable_params(model)}",
        "",
        "Best-practice setup:",
        "- EfficientNet-B0 pretrained backbone unfrozen",
        "- Full fine-tuning on train split",
        "- AdamW optimizer over all model parameters",
        "",
        "Test metrics:",
        f"Accuracy: {details['accuracy']:.4f}",
        f"Sensitivity: {details['sensitivity']:.4f}",
        f"Specificity: {details['specificity']:.4f}",
        f"Precision: {details['precision']:.4f}",
        f"F1-score: {details['f1_score']:.4f}",
        f"AUC: {details['auc']:.4f}" if details['auc'] is not None else "AUC: N/A",
        "",
        "Epoch history:",
    ]
    lines.extend(
        f"Epoch {entry['epoch']}: train_acc={entry['train']['accuracy']:.4f}, val_acc={entry['val']['accuracy']:.4f}, train_loss={entry['train']['loss']:.4f}, val_loss={entry['val']['loss']:.4f}"
        for entry in history
    )

    save_text_report(output_dir / "effnet_b0_ft_report.txt", lines)
    save_json(output_dir / "effnet_b0_ft_history.json", {"history": history, "test": details})
    print(f"Saved: {output_dir}")


if __name__ == "__main__":
    main()
