from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import yaml


def load_config(config_path: str | Path = "configs/config.yaml") -> Dict[str, Any]:
    """Load YAML config and create output directories."""
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    for key in ["reports_dir", "figures_dir"]:
        if key in config:
            Path(config[key]).mkdir(parents=True, exist_ok=True)

    model_path = Path(config["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)

    return config
