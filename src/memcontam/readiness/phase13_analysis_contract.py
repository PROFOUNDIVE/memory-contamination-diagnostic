from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Final

from pydantic import ValidationError

from memcontam.readiness.phase13_analysis_models import AnalysisRegistry
from memcontam.readiness.phase13_authority_files import AuthorityFileError, read_regular_nofollow


EXECUTION_PATH: Final = "data/phase13/authority/execution_registry_v1.json"
EXECUTION_FILE_SHA256: Final = "7c92189f645b74381f3fedf2d3ffbc8c4768a7019eaeb621edb8c22fb4a34970"
EXECUTION_REGISTRY_HASH: Final = "acb769e1e1adbc3eb69e4302322c8eac81829dc836611519caea2ba960900c38"
TASKS: Final = ("game24", "math_equation_balancer", "word_sorting")
BASELINES: Final = ("fh_bounded", "rag_frozen", "bot_style", "reflexion_style")
PAIRS: Final = (
    ("P01", "fh_bounded", "rag_frozen", "required_confirmatory", True),
    ("P02", "fh_bounded", "bot_style", "required_confirmatory", True),
    ("P03", "fh_bounded", "reflexion_style", "required_confirmatory", True),
    ("P04", "rag_frozen", "bot_style", "planned_secondary", False),
    ("P05", "rag_frozen", "reflexion_style", "planned_secondary", False),
    ("P06", "bot_style", "reflexion_style", "planned_secondary", False),
)
WINDOWS: Final = (
    "accuracy-h2-sensitivity", "recurrence-h2-descriptive", "recurrence-h5-secondary",
    "persistence-h5-secondary", "propagation-h5-conditional", "collapse-h5-exploratory",
    "accuracy-h10-sensitivity", "recurrence-h10-descriptive", "persistence-h10-descriptive",
    "propagation-h10-conditional", "collapse-h10-exploratory",
)
OFFLINE_ROWS: Final = (
    ("prefix_derivation", "phase13-offline-compute-owner-v1"),
    ("paired_seed_bootstrap", "phase13-offline-compute-owner-v1"),
    ("report_rendering", "phase13-offline-compute-owner-v1"),
)


class Phase13AnalysisError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _canonical_hash(payload: dict[str, object]) -> str:
    unsigned = dict(payload)
    unsigned.pop("registry_hash", None)
    return hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _expected_slots() -> tuple[tuple[int, str, str, str | None], ...]:
    estimands = (
        *[f"l1-{baseline}-clean-contam" for baseline in BASELINES],
        *[f"l2-p0{index}-clean-contam-did" for index in range(1, 4)],
    )
    return tuple(
        (index, estimand, "L1" if index <= 4 else "L2", None if index <= 4 else f"P0{index - 4}")
        for index, estimand in enumerate(estimands, start=1)
    )


def _validate_execution(registry: AnalysisRegistry, root: Path) -> None:
    reference = registry.execution_authority
    if (reference.file_sha256, reference.registry_hash) != (
        EXECUTION_FILE_SHA256, EXECUTION_REGISTRY_HASH,
    ):
        raise Phase13AnalysisError("EXECUTION_AUTHORITY_MISMATCH")
    try:
        raw = read_regular_nofollow(root / EXECUTION_PATH)
    except AuthorityFileError as error:
        raise Phase13AnalysisError(str(error)) from error
    if hashlib.sha256(raw).hexdigest() != EXECUTION_FILE_SHA256:
        raise Phase13AnalysisError("EXECUTION_AUTHORITY_MISMATCH")
    payload = json.loads(raw)
    if not isinstance(payload, dict) or (
        payload.get("registry_hash"), payload.get("registry_id"),
        payload.get("analysis_window_registry_id"),
    ) != (EXECUTION_REGISTRY_HASH, reference.registry_id, reference.analysis_window_registry_id):
        raise Phase13AnalysisError("EXECUTION_AUTHORITY_MISMATCH")


def _validate_semantics(registry: AnalysisRegistry) -> None:
    l1 = tuple(
        (row.baseline, row.support_population_id, row.status) for row in registry.support.level_1
    )
    l2 = tuple(
        (row.pair_id, row.left_baseline, row.right_baseline, row.status, row.route_gating)
        for row in registry.support.level_2
    )
    families = tuple(
        (
            family.task, family.family_id,
            tuple(
                (slot.order, slot.estimand_id, slot.support_level, slot.pair_id)
                for slot in family.slots
            ),
        )
        for family in registry.inference.families
    )
    expected_families = tuple(
        (task, f"{task}-h5-primary-holm-v1", _expected_slots()) for task in TASKS
    )
    if (
        l1 != tuple(
            (baseline, f"l1-{baseline}-structural-support", "baseline_local")
            for baseline in BASELINES
        )
        or l2 != PAIRS
        or (registry.support.level_3.support_population_id, registry.support.level_3.baselines)
        != ("l3-all-primary-baselines-structural-support", BASELINES)
        or tuple(
            (row.route, row.level_1, row.level_2, row.level_3)
            for row in registry.planning.targets
        ) != (("3w", 10, 10, None), ("5w", 16, 16, 8))
        or registry.planning.calibration_seeds != tuple(range(10000, 10012))
        or families != expected_families
        or tuple(row.analysis_window_id for row in registry.non_primary_windows) != WINDOWS
        or registry.excluded_conditions != ("nomem", "filter_challenge")
        or tuple((row.operation, row.owner_id) for row in registry.offline_compute.rows)
        != OFFLINE_ROWS
    ):
        raise Phase13AnalysisError("ANALYSIS_SEMANTICS_INVALID")


def parse_analysis_registry(raw: bytes, root: Path) -> AnalysisRegistry:
    try:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise Phase13AnalysisError("MALFORMED_REGISTRY")
        registry = AnalysisRegistry.model_validate(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError) as error:
        raise Phase13AnalysisError("ANALYSIS_SEMANTICS_INVALID") from error
    if _canonical_hash(payload) != registry.registry_hash:
        raise Phase13AnalysisError("REGISTRY_HASH_MISMATCH")
    _validate_execution(registry, root)
    _validate_semantics(registry)
    return registry


def load_analysis_registry(path: Path, root: Path) -> AnalysisRegistry:
    try:
        raw = read_regular_nofollow(path)
    except AuthorityFileError as error:
        raise Phase13AnalysisError(str(error)) from error
    return parse_analysis_registry(raw, root)
