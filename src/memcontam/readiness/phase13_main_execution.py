from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Final

from pydantic import ValidationError

from memcontam.experiment.phase12.filter_challenge.mft_state_models import JsonValue
from memcontam.readiness.phase13_authority_files import AuthorityFileError, read_regular_nofollow
from memcontam.readiness.phase13_execution_contract import CORE_MAIN_REGISTRY
from memcontam.readiness.phase13_main_execution_bindings import (
    MainExecutionBindingError,
    canonical_hash,
    validate_artifact_bindings,
    validate_semantic_joins,
)
from memcontam.readiness.phase13_main_checkpoint import validate_main_checkpoint_package
from memcontam.readiness.phase13_main_execution_models import (
    AuthorizedExecution,
    MainAuthorizationReport,
    MainExecutionFreeze,
    MainExecutionFreezeReport,
)
from memcontam.readiness.phase13_main_readiness import validate_main_readiness


EXPECTED_TASKS: Final = CORE_MAIN_REGISTRY.tasks
EXPECTED_SEEDS: Final = tuple(range(10))
EXPECTED_ARM_SEQUENCES: Final = tuple(
    CORE_MAIN_REGISTRY.arms[offset:] + CORE_MAIN_REGISTRY.arms[:offset] for offset in range(4)
)


class Phase13MainExecutionError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _validate_realizations(package: MainExecutionFreeze) -> None:
    dispatch: list[JsonValue] = [
        {"seed_id": seed_id, "task": task}
        for seed_id in package.dispatch.concrete_seed_ids
        for task in package.dispatch.task_order
    ]
    if (
        package.dispatch.task_order != EXPECTED_TASKS
        or package.dispatch.concrete_seed_ids != EXPECTED_SEEDS
        or package.dispatch.realization_id != "phase13-seed-major-global-dispatch-v1"
        or package.dispatch.expanded_dispatch_sha256 != canonical_hash(dispatch)
    ):
        raise Phase13MainExecutionError("MAIN_EXECUTION_DISPATCH_INVALID")
    sequence_indices = tuple(row.sequence_index for row in package.arm_order.sequences)
    sequences = tuple(row.arms for row in package.arm_order.sequences)
    expected_assignments = tuple(rank % 4 for rank in range(10))
    arm_payload: dict[str, JsonValue] = {
        "seed_sequence_indices": expected_assignments,
        "sequences": EXPECTED_ARM_SEQUENCES,
    }
    if (
        package.arm_order.realization_id != "phase13-four-sequence-cyclic-counterbalance-v1"
        or sequence_indices != (0, 1, 2, 3)
        or sequences != EXPECTED_ARM_SEQUENCES
        or package.arm_order.seed_sequence_indices != expected_assignments
        or package.arm_order.realization_sha256 != canonical_hash(arm_payload)
    ):
        raise Phase13MainExecutionError("MAIN_EXECUTION_ARM_ORDER_INVALID")
    expected_pairs = tuple(
        (task, baseline)
        for task in EXPECTED_TASKS
        for baseline in CORE_MAIN_REGISTRY.memory_baselines
        if (task, baseline) not in CORE_MAIN_REGISTRY.current_main_excluded_cells
    )
    cells = package.active_cells
    if (
        cells.tasks != EXPECTED_TASKS
        or cells.memory_baselines != CORE_MAIN_REGISTRY.memory_baselines
        or cells.arms != CORE_MAIN_REGISTRY.arms
        or cells.included_task_baseline_pairs != expected_pairs
        or cells.nomem_tasks != EXPECTED_TASKS
        or cells.memory_cell_count != len(expected_pairs) * len(CORE_MAIN_REGISTRY.arms)
        or cells.nomem_cell_count != len(EXPECTED_TASKS)
        or cells.attempted_trajectory_count
        != (cells.memory_cell_count + cells.nomem_cell_count) * len(EXPECTED_SEEDS)
    ):
        raise Phase13MainExecutionError("MAIN_EXECUTION_CELL_REGISTRY_INVALID")


def validate_main_execution_freeze(
    repository_root: Path,
    package_path: Path,
) -> MainExecutionFreezeReport:
    try:
        raw = read_regular_nofollow(package_path)
        package = MainExecutionFreeze.model_validate_json(raw)
        payload = package.model_dump(mode="json", exclude={"package_hash"})
        if package.package_hash != canonical_hash(payload):
            raise Phase13MainExecutionError("MAIN_EXECUTION_PACKAGE_HASH_INVALID")
        paths = validate_artifact_bindings(package, repository_root)
        _validate_realizations(package)
        validate_semantic_joins(package, paths)
        p4_raw = read_regular_nofollow(paths["mr_p4_manifest"])
        p4 = json.loads(p4_raw)
        if p4.get("status") != "MR_P4_CLOSED" or p4.get("closure_hash") != package.mr_p4_closure_hash:
            raise Phase13MainExecutionError("MAIN_EXECUTION_MR_P4_BINDING_INVALID")
        if canonical_hash(p4.get("level2_interactions")) != package.level2_registry_sha256:
            raise Phase13MainExecutionError("MAIN_EXECUTION_LEVEL2_BINDING_INVALID")
        if package.schema_version == "phase13_main_execution_freeze_v1":
            validate_main_readiness(
                paths["mr_p4_manifest"].parent,
                repository_root,
                hashlib.sha256(p4_raw).hexdigest(),
            )
        else:
            _validate_corrected_mr_p4(p4, repository_root)
        checkpoint = validate_main_checkpoint_package(
            paths["common_checkpoint_registry"].parent,
            repository_root,
        )
        if checkpoint.seed_ids != EXPECTED_SEEDS or checkpoint.tasks != EXPECTED_TASKS:
            raise Phase13MainExecutionError("MAIN_EXECUTION_CHECKPOINT_BINDING_INVALID")
        bindings = {binding.role: binding.sha256 for binding in package.artifacts}
        if (
            checkpoint.orders_sha256 != bindings["task_seed_orders"]
            or checkpoint.registry_sha256 != bindings["common_checkpoint_registry"]
        ):
            raise Phase13MainExecutionError("MAIN_EXECUTION_CHECKPOINT_BINDING_INVALID")
        return MainExecutionFreezeReport(
            package_id=package.package_id,
            package_sha256=hashlib.sha256(raw).hexdigest(),
            package_hash=package.package_hash,
            status="FROZEN",
            mr_p4_status="CLOSED",
            mr_p5_status="FROZEN",
            measured_main_a_trajectory_count=0,
        )
    except Phase13MainExecutionError:
        raise
    except MainExecutionBindingError as error:
        raise Phase13MainExecutionError(error.code) from error
    except (AuthorityFileError, OSError, ValidationError, ValueError, json.JSONDecodeError) as error:
        raise Phase13MainExecutionError("MAIN_EXECUTION_PACKAGE_INVALID") from error


def _validate_corrected_mr_p4(p4: dict[str, JsonValue], repository_root: Path) -> None:
    payload = dict(p4)
    closure_hash = payload.pop("closure_hash", None)
    if closure_hash != canonical_hash(payload):
        raise Phase13MainExecutionError("MAIN_EXECUTION_MR_P4_BINDING_INVALID")
    artifacts = p4.get("artifacts")
    if not isinstance(artifacts, dict):
        raise Phase13MainExecutionError("MAIN_EXECUTION_MR_P4_BINDING_INVALID")
    for identity in artifacts.values():
        if not isinstance(identity, dict):
            raise Phase13MainExecutionError("MAIN_EXECUTION_MR_P4_BINDING_INVALID")
        path, expected = identity.get("path"), identity.get("sha256")
        if not isinstance(path, str) or not isinstance(expected, str):
            raise Phase13MainExecutionError("MAIN_EXECUTION_MR_P4_BINDING_INVALID")
        if hashlib.sha256(read_regular_nofollow(repository_root / path)).hexdigest() != expected:
            raise Phase13MainExecutionError("MAIN_EXECUTION_MR_P4_BINDING_INVALID")


def validate_main_authorization(
    repository_root: Path,
    package_path: Path,
    authorization_path: Path,
    expected_authorization_sha256: str,
) -> MainAuthorizationReport:
    freeze = validate_main_execution_freeze(repository_root, package_path)
    try:
        raw = read_regular_nofollow(authorization_path)
        if hashlib.sha256(raw).hexdigest() != expected_authorization_sha256:
            raise Phase13MainExecutionError("MAIN_AUTHORIZATION_FILE_HASH_MISMATCH")
        authorization = AuthorizedExecution.model_validate_json(raw)
        payload = authorization.model_dump(mode="json", exclude={"authorization_hash"})
        if authorization.authorization_hash != canonical_hash(payload):
            raise Phase13MainExecutionError("MAIN_AUTHORIZATION_HASH_INVALID")
        expected_package_path = str(package_path.resolve().relative_to(repository_root.resolve()))
        if (
            authorization.execution_package_path != expected_package_path
            or authorization.execution_package_sha256 != freeze.package_sha256
            or authorization.execution_package_hash != freeze.package_hash
        ):
            raise Phase13MainExecutionError("MAIN_AUTHORIZATION_PACKAGE_BINDING_INVALID")
        return MainAuthorizationReport(
            authorization_id=authorization.authorization_id,
            authorization_sha256=hashlib.sha256(raw).hexdigest(),
            authorization_hash=authorization.authorization_hash,
            execution_package_sha256=authorization.execution_package_sha256,
            status="AUTHORIZED_EXECUTION",
            mr_p6_status="PASS",
            main_a_status="NOT_STARTED",
            measured_main_a_trajectory_count=0,
        )
    except Phase13MainExecutionError:
        raise
    except (AuthorityFileError, OSError, ValidationError, ValueError) as error:
        raise Phase13MainExecutionError("MAIN_AUTHORIZATION_INVALID") from error
