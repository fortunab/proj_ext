from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from alzheimer_xai.config import load_config
from alzheimer_xai.data import load_datasets
from alzheimer_xai.model import build_efficientnet_model
from alzheimer_xai.evaluate import (
    collect_predictions,
    save_classification_report,
    plot_training_history,
    plot_confusion_matrix,
    plot_multiclass_roc,
)


def main():
    parser = argparse.ArgumentParser(description="Train EfficientNetB0 Alzheimer MRI classifier.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config YAML.")
    args = parser.parse_args()

    config = load_config(args.config)
    image_size = tuple(config["image_size"])

    train_data, test_data, class_names, class_counts = load_datasets(
        train_dir=config["train_dir"],
        test_dir=config["test_dir"],
        image_size=image_size,
        batch_size=config["batch_size"],
    )

    print(f"Class names: {class_names}")
    print(f"Class counts: {class_counts}")

    model = build_efficientnet_model(
        num_classes=config["num_classes"],
        image_size=image_size,
        learning_rate=config["learning_rate"],
        fine_tune_from_layer=config["fine_tune_from_layer"],
    )

    model.summary()

    early = config["early_stopping"]
    reduce = config["reduce_lr"]

    callbacks = [
        EarlyStopping(
            monitor=early["monitor"],
            min_delta=early["min_delta"],
            patience=early["patience"],
            restore_best_weights=early["restore_best_weights"],
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor=reduce["monitor"],
            factor=reduce["factor"],
            patience=reduce["patience"],
            min_lr=reduce["min_lr"],
            verbose=1,
        ),
    ]

    history = model.fit(
        train_data,
        epochs=config["epochs"],
        validation_data=test_data,
        callbacks=callbacks,
    )

    model_path = Path(config["model_path"])
    model.save(model_path)
    print(f"Model saved to: {model_path}")

    loss, accuracy = model.evaluate(test_data)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Test Loss: {loss:.4f}")

    reports_dir = Path(config["reports_dir"])
    figures_dir = Path(config["figures_dir"])

    plot_training_history(history, figures_dir)

    y_true, y_pred, y_prob = collect_predictions(model, test_data)

    report = save_classification_report(
        y_true,
        y_pred,
        class_names,
        reports_dir / "classification_report.txt",
    )
    print(report)

    conf_matrix = plot_confusion_matrix(
        y_true,
        y_pred,
        class_names,
        figures_dir / "confusion_matrix.png",
    )
    print("Confusion Matrix:")
    print(conf_matrix)

    roc_auc = plot_multiclass_roc(
        y_true,
        y_prob,
        class_names,
        figures_dir / "roc_curve.png",
    )
    print(f"ROC AUC: {roc_auc}")


if __name__ == "__main__":
    main()
