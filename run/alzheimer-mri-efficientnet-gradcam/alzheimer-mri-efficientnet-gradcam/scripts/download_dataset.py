from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import kagglehub
import yaml


def copytree_contents(source: Path, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        target = destination / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def main():
    parser = argparse.ArgumentParser(description="Download Alzheimer MRI dataset from KaggleHub.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config YAML.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    dataset_name = config["dataset_name"]
    data_dir = Path(config["data_dir"])

    downloaded_path = Path(kagglehub.dataset_download(dataset_name))
    print(f"Downloaded dataset to: {downloaded_path}")

    # Some Kaggle datasets already contain train/test at root;
    # others may contain one nested folder. Try to locate train/test automatically.
    candidates = [downloaded_path] + [p for p in downloaded_path.iterdir() if p.is_dir()]
    source_root = None
    for candidate in candidates:
        if (candidate / "train").exists() and (candidate / "test").exists():
            source_root = candidate
            break

    if source_root is None:
        raise FileNotFoundError(
            "Could not find train/ and test/ folders in the downloaded dataset. "
            "Please copy them manually into data/train and data/test."
        )

    copytree_contents(source_root, data_dir)
    print(f"Dataset copied to: {data_dir.resolve()}")
    print("Expected folders:")
    print(f"  {Path(config['train_dir']).resolve()}")
    print(f"  {Path(config['test_dir']).resolve()}")


if __name__ == "__main__":
    main()
