from __future__ import annotations

import json
from pathlib import Path


def load_catalog(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
