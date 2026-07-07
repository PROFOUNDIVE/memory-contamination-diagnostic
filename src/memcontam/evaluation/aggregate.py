from __future__ import annotations

from pathlib import Path


def aggregate_run(run_dir: Path) -> dict:
    return {"run_dir": str(run_dir), "status": "aggregate skeleton"}
