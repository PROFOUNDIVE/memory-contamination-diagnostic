from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

from .phase13_calibration_v2_lifecycle_models import LifecycleReport


def lifecycle_identities(
    config_path: Path, request_path: Path | None, authorization_path: Path | None
) -> dict[str, str | None]:
    def digest(path: Path | None) -> str | None:
        return hashlib.sha256(path.read_bytes()).hexdigest() if path is not None and path.is_file() else None

    try:
        git_identity = subprocess.run(
            ["git", "show", "-s", "--format=%H%n%cI", "HEAD"],
            cwd=config_path.resolve().parents[2], check=True,
            capture_output=True, text=True,
        ).stdout.splitlines()
        commit, commit_timestamp = git_identity
    except (OSError, subprocess.SubprocessError):
        commit = commit_timestamp = None
    return {
        "config_sha256": digest(config_path),
        "request_sha256": digest(request_path),
        "authorization_sha256": digest(authorization_path),
        "implementation_commit": commit,
        "implementation_timestamp": commit_timestamp,
    }


def seal_lifecycle_report(path: Path, report: LifecycleReport) -> LifecycleReport:
    payload = asdict(report)
    payload.pop("report_sha256")
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    sealed = replace(report, report_sha256=digest)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(asdict(sealed), stream, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return sealed


__all__ = ("lifecycle_identities", "seal_lifecycle_report")
