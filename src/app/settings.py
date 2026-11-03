from __future__ import annotations

import json
from pathlib import Path


def load_config(root: Path) -> dict[str, object]:
    config_path = root / "config.json"
    if not config_path.is_file():
        config_path = root / "config.example.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a JSON object: {config_path}")
    return config
