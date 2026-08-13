from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from memcontam.readiness.phase13_authority_files import AuthorityFileError, read_regular_nofollow
from memcontam.readiness.phase13_calibration_v2_runtime_models import TrajectoryRequest
from memcontam.readiness.phase13_execution_contract import Phase13ExecutionError, load_execution_registry
from memcontam.readiness.phase13_execution_models import AnalysisWindow
from memcontam.readiness.phase13_structural_authority import (
    RegisteredCheckpoint,
    StructuralAuthorityError,
    registered_checkpoints,
)


EXECUTION_PATH: Final = Path("data/phase13/authority/execution_registry_v1.json")
EXECUTION_FILE_SHA256: Final = "7c92189f645b74381f3fedf2d3ffbc8c4768a7019eaeb621edb8c22fb4a34970"


class PrefixAuthorityError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class PrefixAuthority:
    execution_registry_hash: str
    execution_contract_id: str
    execution_owner_id: str
    identities: dict[str, str]
    ordered_suffix: tuple[str, ...]
    ordered_stream_sha256: str
    checkpoints: tuple[RegisteredCheckpoint, ...]
    windows: tuple[AnalysisWindow, ...]


def load_prefix_authority(request: TrajectoryRequest) -> PrefixAuthority:
    root = request.verified.root
    try:
        raw = read_regular_nofollow(root / EXECUTION_PATH)
    except AuthorityFileError as error:
        raise PrefixAuthorityError("EXECUTION_AUTHORITY_INVALID") from error
    if hashlib.sha256(raw).hexdigest() != EXECUTION_FILE_SHA256:
        raise PrefixAuthorityError("EXECUTION_AUTHORITY_INVALID")
    try:
        execution = load_execution_registry(root / EXECUTION_PATH, root)
    except Phase13ExecutionError as error:
        raise PrefixAuthorityError(error.code) from error
    try:
        checkpoints = registered_checkpoints(request.stream_id, root)
    except StructuralAuthorityError as error:
        raise PrefixAuthorityError(error.code) from error
    ordered_suffix = request.verified.ordered_suffixes.get((request.task, request.seed_id))
    stream = next(
        (item for item in execution.task_streams if item.task == request.task), None
    )
    suffix = None if stream is None else next(
        (item for item in stream.suffixes if item.seed_id == request.seed_id), None
    )
    if ordered_suffix is None or suffix is None:
        raise PrefixAuthorityError("SUFFIX_ORDER_INVALID")
    return PrefixAuthority(
        execution.registry_hash,
        execution.execution_contract_id,
        execution.execution_owner_id,
        execution.identities.model_dump(),
        ordered_suffix,
        suffix.source_ordered_stream_sha256,
        checkpoints,
        tuple(window for window in execution.analysis_windows if window.window_length < 10),
    )


def canonical_short_windows() -> tuple[AnalysisWindow, ...]:
    definitions = (
        ("accuracy-h2-sensitivity", 2, "verified_accuracy", "prespecified_sensitivity", "descriptive_no_inferential_family"),
        ("recurrence-h2-descriptive", 2, "recurrence", "descriptive", "estimation_only"),
        ("accuracy-h5-primary", 5, "verified_accuracy", "confirmatory_primary", "primary_holm_family"),
        ("recurrence-h5-secondary", 5, "recurrence", "confirmatory_secondary", "estimation_only"),
        ("persistence-h5-secondary", 5, "persistence", "confirmatory_secondary", "estimation_only"),
        ("propagation-h5-conditional", 5, "propagation", "descriptive", "descriptive_no_inferential_family"),
        ("collapse-h5-exploratory", 5, "collapse_like", "exploratory", "descriptive_no_inferential_family"),
    )
    return tuple(
        AnalysisWindow.model_validate(
            {
                "analysis_window_id": identity,
                "source_execution_contract_id": "phase13-main-a-h10-execution-v1",
                "window_length": length,
                "event_time_start": 0,
                "event_time_end": length - 1,
                "outcome_family": outcome,
                "evidence_status": evidence,
                "multiplicity_status": multiplicity,
                "realization_disposition": "prefix_view",
                "provider_execution_multiplicity": 0,
            }
        )
        for identity, length, outcome, evidence, multiplicity in definitions
    )


__all__ = (
    "PrefixAuthority", "PrefixAuthorityError", "canonical_short_windows",
    "load_prefix_authority",
)
