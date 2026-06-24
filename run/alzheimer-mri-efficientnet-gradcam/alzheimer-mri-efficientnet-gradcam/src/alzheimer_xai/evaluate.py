from __future__ import annotations

from pathlib import Path
from itertools import cycle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize


def collect_predictions(model, dataset):
    y_true = []
    y_pred = []
    y_prob_batches = []

    for images, labels in dataset:
        y_true.extend(labels.numpy())
        probabilities = model.predict(images, verbose=0)
        y_prob_batches.append(probabilities)
        y_pred.extend(np.argmax(probabilities, axis=1))

    y_prob = np.concatenate(y_prob_batches, axis=0)
    return np.array(y_true), np.array(y_pred), y_prob


def save_classification_report(y_true, y_pred, class_names, output_path: str | Path):
    output_path = Path(output_path)
    report = classification_report(y_true, y_pred, target_names=class_names)
    output_path.write_text(report, encoding="utf-8")
    return report


def plot_training_history(history, figures_dir: str | Path):
    figures_dir = Path(figures_dir)
    history_df = pd.DataFrame(history.history)

    accuracy_path = figures_dir / "training_accuracy.png"
    loss_path = figures_dir / "training_loss.png"

    history_df[["accuracy", "val_accuracy"]].plot()
    plt.title("Model Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(accuracy_path, dpi=300)
    plt.close()

    history_df[["loss", "val_loss"]].plot()
    plt.title("Model Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_path, dpi=300)
    plt.close()

    return accuracy_path, loss_path


def plot_confusion_matrix(y_true, y_pred, class_names, output_path: str | Path):
    conf_matrix = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return conf_matrix


def plot_multiclass_roc(y_true, y_prob, class_names, output_path: str | Path):
    num_classes = len(class_names)
    y_true_one_hot = label_binarize(y_true, classes=range(num_classes))

    fpr = {}
    tpr = {}
    roc_auc = {}

    plt.figure(figsize=(10, 8))

    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_one_hot.ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label=f'Micro-average ROC curve (area = {roc_auc["micro"]:.2f})',
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(["aqua", "darkorange", "cornflowerblue", "green"])
    for i, color in zip(range(num_classes), colors):
        fpr[i], tpr[i], _ = roc_curve(y_true_one_hot[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(
            fpr[i],
            tpr[i],
            lw=2,
            label=f"ROC curve of class {class_names[i]} (area = {roc_auc[i]:.2f})",
        )

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multi-class Receiver Operating Characteristic Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return roc_auc
