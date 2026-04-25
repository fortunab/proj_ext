from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def save_bar_chart(labels, values, ylabel, title, output_path: str | Path, ylim=None):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    plt.bar(labels, values)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=20, ha="right")

    if ylim is not None:
        plt.ylim(*ylim)
    elif np.max(values) <= 1.0:
        plt.ylim(max(0, np.min(values) - 0.05), min(1.0, np.max(values) + 0.05))
    else:
        plt.ylim(max(0, np.min(values) - 1), np.max(values) + 2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path


def save_line_chart(x, series, xlabel, ylabel, title, output_path: str | Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    for label, values in series.items():
        plt.plot(x, values, marker="o", label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path
