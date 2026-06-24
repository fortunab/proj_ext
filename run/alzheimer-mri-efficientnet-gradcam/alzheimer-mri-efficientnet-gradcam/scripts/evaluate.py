from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf

from alzheimer_xai.config import load_config
from alzheimer_xai.data import load_datasets
from alzheimer_xai.evaluate import (
    collect_predictions,
    save_classification_report,
    plot_confusion_matrix,
    plot_multiclass_roc,
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved Alzheimer MRI classifier.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config YAML.")
    parser.add_argument("--model", default=None, help="Optional custom model path.")
    args = parser.parse_args()

    config = load_config(args.config)
    image_size = tuple(config["image_size"])
    model_path = args.model or config["model_path"]

    _, test_data, class_names, _ = load_datasets(
        train_dir=config["train_dir"],
        test_dir=config["test_dir"],
        image_size=image_size,
        batch_size=config["batch_size"],
    )

    model = tf.keras.models.load_model(model_path)
    loss, accuracy = model.evaluate(test_data)

    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Test Loss: {loss:.4f}")

    y_true, y_pred, y_prob = collect_predictions(model, test_data)

    reports_dir = Path(config["reports_dir"])
    figures_dir = Path(config["figures_dir"])

    report = save_classification_report(
        y_true,
        y_pred,
        class_names,
        reports_dir / "classification_report.txt",
    )
    print(report)

    plot_confusion_matrix(
        y_true,
        y_pred,
        class_names,
        figures_dir / "confusion_matrix.png",
    )

    plot_multiclass_roc(
        y_true,
        y_prob,
        class_names,
        figures_dir / "roc_curve.png",
    )

    print(f"Saved evaluation outputs to: {figures_dir} and {reports_dir}")


if __name__ == "__main__":
    main()
