from __future__ import annotations

from pathlib import Path
from collections import Counter
from typing import Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.efficientnet import preprocess_input


def load_datasets(
    train_dir: str,
    test_dir: str,
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = 32,
):
    """Load train and test datasets from directory structure."""
    train_dir = Path(train_dir)
    test_dir = Path(test_dir)

    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    train_dataset = keras.utils.image_dataset_from_directory(
        directory=train_dir,
        labels="inferred",
        label_mode="int",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        verbose=True,
    )

    test_dataset = keras.utils.image_dataset_from_directory(
        directory=test_dir,
        labels="inferred",
        label_mode="int",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False,
        verbose=True,
    )

    class_names = train_dataset.class_names

    def preprocess(image, label):
        image = preprocess_input(image)
        return image, label

    train_prepared = train_dataset.map(preprocess).prefetch(tf.data.AUTOTUNE)
    test_prepared = test_dataset.map(preprocess).prefetch(tf.data.AUTOTUNE)

    class_counts = Counter()
    for _, labels in train_dataset:
        class_counts.update(labels.numpy().tolist())

    return train_prepared, test_prepared, class_names, dict(class_counts)
