from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from memcontam.clients.provider_profile import ProviderProfile


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary_path).replace(path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
    return path


def write_provider_profile_atomic(run_dir: Path | str, profile: ProviderProfile) -> Path:
    return _write_json_atomic(Path(run_dir) / "provider_profile.json", profile.to_dict())


def write_resolved_config_atomic(run_dir: Path | str, resolved_config: dict[str, Any]) -> Path:
    return _write_json_atomic(Path(run_dir) / "resolved_config.json", resolved_config)
