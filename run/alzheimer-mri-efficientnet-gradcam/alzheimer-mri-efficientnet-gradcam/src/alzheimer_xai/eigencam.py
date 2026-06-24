from __future__ import annotations

from pathlib import Path
import glob
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def find_last_conv_layer(model: tf.keras.Model) -> str:
    """Return the name of the last layer with a 4D output."""
    for layer in reversed(model.layers):
        try:
            if len(layer.output.shape) == 4:
                return layer.name
        except Exception:
            continue
    raise ValueError("No convolution-like 4D layer found in the model.")


def load_image(path: str | Path, image_size: tuple[int, int]):
    img = tf.keras.utils.load_img(path, target_size=image_size)
    img_array = tf.keras.utils.img_to_array(img).astype(np.float32)
    return np.expand_dims(img_array, axis=0), img_array.astype(np.uint8)


def eigen_attention_map(model, image_batch, image_size: tuple[int, int], last_conv_layer_name: str | None = None):
    """Create Eigen-CAM-like attention map using SVD on last convolutional features."""
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)

    feature_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=model.get_layer(last_conv_layer_name).output,
    )

    features = feature_model(image_batch)[0].numpy()
    features = np.maximum(features, 0)

    h, w, c = features.shape
    reshaped = features.reshape((-1, c))
    reshaped = reshaped - reshaped.mean(axis=0)

    _, _, vt = np.linalg.svd(reshaped, full_matrices=False)

    cam = reshaped @ vt[0]
    cam = cam.reshape(h, w)

    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)
    cam = cv2.resize(cam, image_size)
    cam = cv2.GaussianBlur(cam, (9, 9), 0)
    cam = cam / (cam.max() + 1e-8)

    return cam


def full_heatmap_overlay(image, cam, alpha_image=0.55, alpha_heatmap=0.45):
    cam_uint8 = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(
        image,
        alpha_image,
        heatmap,
        alpha_heatmap,
        0,
    )

    return heatmap, overlay


def collect_images_per_class(test_dir: str | Path, images_per_class: int = 3, seed: int = 42):
    test_dir = Path(test_dir)
    class_names = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])

    random.seed(seed)
    selected_images = []

    for class_name in class_names:
        image_files = []
        for extension in ["*.jpg", "*.jpeg", "*.png"]:
            image_files.extend(glob.glob(str(test_dir / class_name / extension)))

        chosen = random.sample(image_files, min(images_per_class, len(image_files)))
        for path in chosen:
            selected_images.append((path, class_name))

    return selected_images, class_names


def save_eigencam_grid(
    model,
    test_dir: str | Path,
    output_path: str | Path,
    image_size: tuple[int, int] = (224, 224),
    images_per_class: int = 3,
    seed: int = 42,
    alpha_image: float = 0.55,
    alpha_heatmap: float = 0.45,
):
    selected_images, class_names = collect_images_per_class(test_dir, images_per_class, seed)
    rows = len(selected_images)

    if rows == 0:
        raise ValueError(f"No images found in {test_dir}")

    last_conv_layer_name = find_last_conv_layer(model)
    plt.figure(figsize=(18, rows * 4))

    for idx, (img_path, true_class) in enumerate(selected_images):
        image_batch, image = load_image(img_path, image_size)

        predictions = model.predict(image_batch, verbose=0)[0]
        pred_index = int(np.argmax(predictions))
        pred_class = class_names[pred_index] if pred_index < len(class_names) else str(pred_index)

        cam = eigen_attention_map(model, image_batch, image_size, last_conv_layer_name)
        heatmap, overlay = full_heatmap_overlay(image, cam, alpha_image, alpha_heatmap)

        plt.subplot(rows, 3, idx * 3 + 1)
        plt.imshow(image)
        plt.title(f"Original\nTrue: {true_class}")
        plt.axis("off")

        plt.subplot(rows, 3, idx * 3 + 2)
        plt.imshow(heatmap)
        plt.title("Full Attention Heatmap")
        plt.axis("off")

        plt.subplot(rows, 3, idx * 3 + 3)
        plt.imshow(overlay)
        plt.title(f"Overlay\nPredicted: {pred_class}")
        plt.axis("off")

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path
