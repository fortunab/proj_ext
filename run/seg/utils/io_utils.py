
import json
from pathlib import Path
from typing import Any, Mapping

def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def write_text(path: str | Path, text: str) -> None:
    p = ensure_parent(path)
    p.write_text(text, encoding="utf-8")

def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    p = ensure_parent(path)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
