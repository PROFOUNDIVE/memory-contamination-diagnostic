from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from memcontam.manifests.phase13_archive_models import ArchiveAuthorities
from memcontam.readiness.phase13_analysis_contract import load_analysis_registry
from memcontam.readiness.phase13_analysis_models import AnalysisRegistry
from memcontam.readiness.phase13_authority_files import AuthorityFileError, read_regular_nofollow
from memcontam.readiness.phase13_execution_contract import load_execution_registry
from memcontam.readiness.phase13_execution_models import AnalysisWindow, ExecutionRegistry
from memcontam.readiness.phase13_structural_authority import (
    StructuralAuthorityError,
    registered_checkpoints,
)


ROOT: Final = Path(__file__).resolve().parents[3]
ROLE_BINDINGS: Final = {
    "execution": (
        ROOT / "data/phase13/authority/execution_registry_v1.json",
        "7c92189f645b74381f3fedf2d3ffbc8c4768a7019eaeb621edb8c22fb4a34970",
    ),
    "analysis": (
        ROOT / "data/phase13/authority/analysis_registry_v1.json",
        "b58e6aec8acc040fb934e9b25842eb68c702d098a08b41ba0eab9502a198a0f3",
    ),
    "historical": (
        ROOT / "data/phase13/authority/historical_compatibility_v1.json",
        "446e5634d7be2bd049ffd3af733262e72a076d22ec24a0e9c11d7259b60264d4",
    ),
    "checkpoint": (
        ROOT / "data/phase13/authority/structural_checkpoint_registry_v1.json",
        "c2173d1fb5557611050a7e281fcf0613671bda06ffad6e7cc370de568f37ecff",
    ),
}


class ArchiveAuthorityError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class StreamProjection:
    stream_id: str
    task: str
    seed_id: int
    ordered_stream_sha256: str
    checkpoints: dict[str, str]
    checkpoint_hashes: dict[str, str]


@dataclass(frozen=True, slots=True)
class ArchiveProjection:
    execution: ExecutionRegistry
    analysis: AnalysisRegistry
    streams: dict[str, StreamProjection]
    windows: tuple[AnalysisWindow, ...]
    primary_families: dict[str, str]
    historical_run_id: str
    historical_availability: str


def load_archive_projection(bindings: ArchiveAuthorities) -> ArchiveProjection:
    supplied = {
        "execution": bindings.execution,
        "analysis": bindings.analysis,
        "historical": bindings.historical,
        "checkpoint": bindings.checkpoint,
    }
    authenticated: dict[str, bytes] = {}
    for role, (path, digest) in ROLE_BINDINGS.items():
        binding = supplied[role]
        if Path(binding.path).resolve() != path.resolve():
            raise ArchiveAuthorityError("AUTHORITY_ROLE_MISMATCH")
        if binding.sha256 != digest:
            raise ArchiveAuthorityError("AUTHORITY_HASH_MISMATCH")
        try:
            raw = read_regular_nofollow(path)
        except AuthorityFileError as error:
            raise ArchiveAuthorityError("AUTHORITY_HASH_MISMATCH") from error
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ArchiveAuthorityError("AUTHORITY_HASH_MISMATCH")
        authenticated[role] = raw
    try:
        execution = load_execution_registry(ROLE_BINDINGS["execution"][0], ROOT)
        analysis = load_analysis_registry(ROLE_BINDINGS["analysis"][0], ROOT)
    except ValueError as error:
        raise ArchiveAuthorityError("AUTHORITY_CONTENT_INVALID") from error
    streams: dict[str, StreamProjection] = {}
    for task in execution.task_streams:
        for suffix in task.suffixes:
            stream_id = f"{task.task}-seed-{suffix.seed_id}"
            try:
                checkpoints = registered_checkpoints(stream_id, ROOT)
            except StructuralAuthorityError:
                continue
            streams[stream_id] = StreamProjection(
                stream_id,
                task.task,
                suffix.seed_id,
                suffix.source_ordered_stream_sha256,
                {row.baseline: row.checkpoint_id for row in checkpoints},
                {row.baseline: row.sha256 for row in checkpoints},
            )
    families = {row.task: row.family_id for row in analysis.inference.families}
    try:
        historical = json.loads(authenticated["historical"])["historical_execution"]
        run_id = historical["run_id"]
        availability = historical["sealed_archive"]["availability"]
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ArchiveAuthorityError("AUTHORITY_CONTENT_INVALID") from error
    return ArchiveProjection(
        execution,
        analysis,
        streams,
        execution.analysis_windows,
        families,
        run_id,
        availability,
    )


__all__ = ("ArchiveAuthorityError", "ArchiveProjection", "StreamProjection", "load_archive_projection")
