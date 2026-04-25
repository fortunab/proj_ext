from __future__ import annotations

from pathlib import Path
from torch.utils.data import DataLoader

from bionnica.config import (
    HYPERKVASIR_DIR,
    OUTPUTS_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    DEVICE,
)
from bionnica.datasets.classification_dataset import ImageFolderFromCSV
from bionnica.models.bionnica_lite import BionnicaLite
from bionnica.metrics.classification_metrics import count_total_params, count_trainable_params
from bionnica.utils.io_utils import ensure_dir, save_text_report


def main() -> None:
    output_dir = ensure_dir(OUTPUTS_DIR / "classification")

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

    model = BionnicaLite(num_classes=2)

    lines = [
        f"DEVICE = {DEVICE}",
        f"Train samples: {len(train_ds)}",
        f"Validation samples: {len(val_ds)}",
        f"Test samples: {len(test_ds)}",
        f"Train batches: {len(train_loader)}",
        f"Validation batches: {len(val_loader)}",
        f"Test batches: {len(test_loader)}",
        f"Total parameters: {count_total_params(model)}",
        f"Trainable parameters: {count_trainable_params(model)}",
        "",
        "This script prepares the structured classification pipeline.",
        "Add your optimizer, loss, training loop, and final evaluation here.",
    ]

    save_text_report(output_dir / "classification_summary.txt", lines)
    print(f"Saved: {output_dir / 'classification_summary.txt'}")


if __name__ == "__main__":
    main()
