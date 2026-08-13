from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Final

from pydantic import ValidationError

from memcontam.readiness.phase13_authority_files import AuthorityFileError, read_regular_nofollow
from memcontam.readiness.phase13_execution_models import ExecutionRegistry
from memcontam.readiness.phase13_execution_semantics import (
    PARTITION_SHA256,
    ordered_stream_hash,
    validate_exact_inventory,
)


TASKS: Final = ("game24", "math_equation_balancer", "word_sorting")
SEEDS: Final = tuple(range(10000, 10012))
PARTITION_PATH: Final = "data/phase13/calibration_v2/seed_partition_registry_v1.json"


class Phase13ExecutionError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code

def _canonical_hash(payload: dict[str, object]) -> str:
    unsigned = dict(payload)
    unsigned.pop("registry_hash", None)
    return hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _decode(raw: bytes) -> dict[str, object]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Phase13ExecutionError("MALFORMED_REGISTRY") from error
    if not isinstance(value, dict):
        raise Phase13ExecutionError("MALFORMED_REGISTRY")
    return value


def _validation_code(error: ValidationError) -> str:
    issue = error.errors()[0]
    location = tuple(str(item) for item in issue["loc"])
    if "H" in location:
        return "BARE_H_PROHIBITED"
    if "operator_capacity" in location:
        return "CAPACITY_CONTRACT_INVALID"
    return "MALFORMED_REGISTRY"


def parse_execution_registry(raw: bytes, root: Path) -> ExecutionRegistry:
    payload = _decode(raw)
    try:
        registry = ExecutionRegistry.model_validate(payload)
    except ValidationError as error:
        raise Phase13ExecutionError(_validation_code(error)) from error
    if _canonical_hash(payload) != registry.registry_hash:
        raise Phase13ExecutionError("REGISTRY_HASH_MISMATCH")
    _validate_semantics(registry, root)
    return registry


def load_execution_registry(path: Path, root: Path) -> ExecutionRegistry:
    try:
        raw = read_regular_nofollow(path)
    except AuthorityFileError as error:
        raise Phase13ExecutionError(str(error)) from error
    return parse_execution_registry(raw, root)


def _load_partition(registry: ExecutionRegistry, root: Path) -> dict[str, object]:
    if (
        registry.source_partition.path != PARTITION_PATH
        or registry.source_partition.sha256 != PARTITION_SHA256
    ):
        raise Phase13ExecutionError("SOURCE_AUTHORITY_HASH_MISMATCH")
    try:
        raw = read_regular_nofollow(root / PARTITION_PATH)
    except AuthorityFileError as error:
        raise Phase13ExecutionError(str(error)) from error
    if hashlib.sha256(raw).hexdigest() != registry.source_partition.sha256:
        raise Phase13ExecutionError("SOURCE_AUTHORITY_HASH_MISMATCH")
    return _decode(raw)


def _validate_semantics(registry: ExecutionRegistry, root: Path) -> None:
    partition = _load_partition(registry, root)
    _validate_timing(registry)
    _validate_streams(registry, partition)
    _validate_arms_and_owners(registry)
    _validate_windows(registry)
    inventory_error = validate_exact_inventory(registry)
    if inventory_error is not None:
        raise Phase13ExecutionError(inventory_error)
    _validate_illustrations(registry)


def _validate_timing(registry: ExecutionRegistry) -> None:
    timing = registry.timing
    if (timing.L_min, timing.tau_star, timing.H_run, timing.minimum_stream_length) != (1, 2, 10, 11):
        raise Phase13ExecutionError("HORIZON_INVALID")
    if (
        (timing.absolute_trial_start, timing.absolute_trial_end) != (2, 11)
        or (timing.event_time_start, timing.event_time_end) != (0, 9)
    ):
        raise Phase13ExecutionError("EVENT_RANGE_INVALID")


def _validate_streams(registry: ExecutionRegistry, partition: dict[str, object]) -> None:
    tasks = partition.get("tasks")
    if not isinstance(tasks, dict) or tuple(stream.task for stream in registry.task_streams) != TASKS:
        raise Phase13ExecutionError("SOURCE_STREAM_IDENTITY_INVALID")
    for stream in registry.task_streams:
        source = tasks.get(stream.task)
        if not isinstance(source, dict):
            raise Phase13ExecutionError("SOURCE_STREAM_IDENTITY_INVALID")
        expected = (
            source.get("calibration_path"), source.get("calibration_sha256"),
            source.get("source_main_v1"), source.get("prospective_main_v2"),
        )
        source_main, remainder = expected[2], expected[3]
        if not isinstance(source_main, dict) or not isinstance(remainder, dict):
            raise Phase13ExecutionError("SOURCE_STREAM_IDENTITY_INVALID")
        claims = (
            stream.calibration_path, stream.calibration_sha256,
            stream.source_main_v1_path, stream.source_main_v1_sha256,
            stream.prospective_main_v2_ordered_signatures_sha256,
        )
        authoritative = (
            expected[0], expected[1], source_main.get("path"), source_main.get("sha256"),
            remainder.get("ordered_signatures_sha256"),
        )
        if claims != authoritative:
            raise Phase13ExecutionError("SOURCE_STREAM_IDENTITY_INVALID")
        trajectories = source.get("trajectories")
        if not isinstance(trajectories, list) or len(stream.suffixes) != len(SEEDS):
            raise Phase13ExecutionError("SUFFIX_ORDER_INVALID")
        for suffix, trajectory in zip(stream.suffixes, trajectories, strict=True):
            if not isinstance(trajectory, dict):
                raise Phase13ExecutionError("SUFFIX_ORDER_INVALID")
            ordered = trajectory.get("ordered_sample_ids")
            if not isinstance(ordered, list) or len(ordered) != 11:
                raise Phase13ExecutionError("SUFFIX_ORDER_INVALID")
            if (
                suffix.seed_id != trajectory.get("seed_id")
                or suffix.source_ordered_stream_sha256 != ordered_stream_hash(ordered)
                or trajectory.get("ordered_stream_sha256") != ordered_stream_hash(ordered)
            ):
                raise Phase13ExecutionError("SUFFIX_ORDER_INVALID")


def _validate_arms_and_owners(registry: ExecutionRegistry) -> None:
    if len(registry.memory_arms) != 4 or len({arm.arm_key for arm in registry.memory_arms}) != 4:
        raise Phase13ExecutionError("ARM_REGISTRY_INVALID")
    expected_owners = {
        "prefix": registry.prefix_owner_id,
        "execution": registry.execution_owner_id,
    }
    if any(expected_owners[component.owner_kind] != component.owner_id for component in registry.call_components):
        raise Phase13ExecutionError("OWNER_BINDING_INVALID")
    if {(item.owner_kind, item.phase) for item in registry.call_components} != {
        ("prefix", "burn_init"), ("execution", "trial")
    }:
        raise Phase13ExecutionError("OWNER_BINDING_INVALID")
    if len(registry.native_capacities) != 4 or len({item.baseline for item in registry.native_capacities}) != 4:
        raise Phase13ExecutionError("CAPACITY_CONTRACT_INVALID")


def _validate_windows(registry: ExecutionRegistry) -> None:
    rows = registry.analysis_windows
    if len({row.analysis_window_id for row in rows}) != len(rows):
        raise Phase13ExecutionError("WINDOW_REGISTRY_INVALID")
    primary = [row for row in rows if row.evidence_status == "confirmatory_primary"]
    if (
        len(primary) != 1
        or primary[0].analysis_window_id != registry.primary_analysis_window_id
        or primary[0].window_length != 5
        or primary[0].outcome_family != "verified_accuracy"
    ):
        raise Phase13ExecutionError("PRIMARY_WINDOW_INVALID")
    for row in rows:
        if row.source_execution_contract_id != registry.execution_contract_id or row.event_time_end != row.window_length - 1:
            raise Phase13ExecutionError("WINDOW_EVENT_RANGE_INVALID")
        if row.window_length < 10 and row.realization_disposition != "prefix_view":
            raise Phase13ExecutionError("WINDOW_REALIZATION_INVALID")
        if row.window_length < 10 and row.provider_execution_multiplicity != 0:
            raise Phase13ExecutionError("WINDOW_EXECUTION_MULTIPLICITY_INVALID")
    source_rows = [row for row in rows if row.provider_execution_multiplicity == 1]
    if len(source_rows) != 1 or source_rows[0].analysis_window_id != "accuracy-h10-sensitivity":
        raise Phase13ExecutionError("WINDOW_EXECUTION_MULTIPLICITY_INVALID")
    if primary[0].multiplicity_status != "primary_holm_family" or any(
        row.multiplicity_status == "primary_holm_family" for row in rows if row is not primary[0]
    ):
        raise Phase13ExecutionError("WINDOW_MULTIPLICITY_INVALID")


def _validate_illustrations(registry: ExecutionRegistry) -> None:
    planning = registry.planning_illustrations
    prefix, execution = registry.call_components
    templates = registry.execution_templates
    expected_templates = {
        (task, baseline, arm)
        for task in TASKS
        for baseline in ("fh_bounded", "rag_frozen", "bot_style", "reflexion_style")
        for arm in ("Clean", "Correct", "Irrelevant", "Contam")
    } | {(task, "nomem", "star_NoMem") for task in TASKS}
    if (
        len(templates) != 51
        or len({row.template_id for row in templates}) != 51
        or {(row.task, row.baseline, row.arm_key) for row in templates} != expected_templates
        or any(row.owner_id != registry.execution_owner_id for row in templates)
        or any(
            sum(row.nominal_semantic_calls_per_trial for row in templates if row.task == task)
            != execution.nominal_calls_per_activation
            or sum(row.raw_maximum_semantic_calls_per_trial for row in templates if row.task == task)
            != execution.raw_maximum_calls_per_activation
            for task in TASKS
        )
    ):
        raise Phase13ExecutionError("EXECUTION_TEMPLATE_INVALID")
    for illustration, multiplicity in (
        (planning.main, "main_seed_multiplicity"),
        (planning.calibration, "calibration_seed_multiplicity"),
    ):
        seeds = sum(getattr(row, multiplicity) for row in templates if row.baseline == "nomem")
        nominal = seeds * prefix.nominal_calls_per_activation + registry.timing.H_run * sum(
            getattr(row, multiplicity) * row.nominal_semantic_calls_per_trial for row in templates
        )
        raw = seeds * prefix.raw_maximum_calls_per_activation + registry.timing.H_run * sum(
            getattr(row, multiplicity) * row.raw_maximum_semantic_calls_per_trial for row in templates
        )
        reserved = (raw * 105 + 99) // 100
        if (
            illustration.task_seed_count != seeds
            or (illustration.nominal_semantic_calls, illustration.raw_maximum_semantic_calls, illustration.reserved_semantic_calls)
            != (nominal, raw, reserved)
            or illustration.raw_maximum_transport_attempts != raw * 4
            or illustration.reserved_transport_attempts != reserved * 4
            or illustration.maximum_input_tokens != reserved * 4 * 4096
            or illustration.maximum_output_tokens != reserved * 4 * 2048
        ):
            raise Phase13ExecutionError("CAPACITY_ILLUSTRATION_INVALID")
