from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf

from alzheimer_xai.config import load_config
from alzheimer_xai.eigencam import save_eigencam_grid


def main():
    parser = argparse.ArgumentParser(description="Generate Eigen-CAM attention heatmaps.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config YAML.")
    parser.add_argument("--model", default=None, help="Optional custom model path.")
    parser.add_argument("--output", default=None, help="Optional output image path.")
    args = parser.parse_args()

    config = load_config(args.config)
    model_path = args.model or config["model_path"]
    output_path = args.output or str(Path(config["figures_dir"]) / "eigencam_grid.png")
    image_size = tuple(config["image_size"])

    model = tf.keras.models.load_model(model_path)

    result = save_eigencam_grid(
        model=model,
        test_dir=config["test_dir"],
        output_path=output_path,
        image_size=image_size,
        images_per_class=config["eigencam"]["images_per_class"],
        seed=config["eigencam"]["seed"],
        alpha_image=config["eigencam"]["overlay_alpha_image"],
        alpha_heatmap=config["eigencam"]["overlay_alpha_heatmap"],
    )

    print(f"Eigen-CAM grid saved to: {result}")


if __name__ == "__main__":
    main()
