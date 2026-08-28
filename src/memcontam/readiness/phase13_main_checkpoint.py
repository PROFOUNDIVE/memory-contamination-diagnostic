from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from memcontam.experiment.phase12.filter_challenge.mft_state_models import JsonValue
from memcontam.readiness.phase13_authority_files import read_regular_nofollow
from memcontam.readiness.phase13_execution_contract import CORE_MAIN_REGISTRY
from memcontam.readiness.phase13_production_runtime_models import ProductionOrdinaryRunIdentity


Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
SEEDS: Final = tuple(range(10))
H_RUN: Final = 50
L_MIN: Final = 1
ORDER_LAW: Final = "sha256_task_seed_v1\\0{task}\\0{decimal(seed)}\\0{sample_id}"
LEGACY_TASKS: Final = ("game24", "math_equation_balancer", "word_sorting")
MMLU_TASKS: Final = ("mmlu_pro_engineering", "mmlu_pro_physics")


class Phase13MainCheckpointError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ArtifactIdentity(_FrozenModel):
    path: str = Field(min_length=1)
    sha256: Sha256


class SeedOrder(_FrozenModel):
    seed: int = Field(ge=0, le=9)
    ordered_sample_ids: tuple[str, ...]
    order_sha256: Sha256


class TaskOrders(_FrozenModel):
    source: ArtifactIdentity
    sample_ids: tuple[str, ...]
    seeds: tuple[SeedOrder, ...]


class TaskSeedOrders(_FrozenModel):
    schema_version: Literal["phase13_task_seed_orders_v1"]
    ordering_law: Literal["sha256_task_seed_v1\\0{task}\\0{decimal(seed)}\\0{sample_id}"]
    concrete_seed_ids: tuple[int, ...]
    tasks: dict[str, TaskOrders]
    orders_hash: Sha256


class CheckpointSeed(_FrozenModel):
    seed: int = Field(ge=0, le=9)
    concrete_seed_id: str = Field(pattern=r"^[0-9]$")
    tau_star: int = Field(ge=1)
    clean_prefix_sample_ids: tuple[str, ...]
    clean_prefix_sample_ids_sha256: Sha256
    suffix_sample_ids: tuple[str, ...]
    suffix_sample_ids_sha256: Sha256
    complete_order_sha256: Sha256


class TaskCheckpoint(_FrozenModel):
    route: Literal["3w"]
    L_min: Literal[1]
    H_run: Literal[50]
    static_route_constraints: tuple[str, ...]
    seeds: tuple[CheckpointSeed, ...]


class CommonCheckpointRegistry(_FrozenModel):
    schema_version: Literal["phase13_main_a_common_checkpoint_registry_v1"]
    task_seed_orders: ArtifactIdentity
    checkpoint_law: Literal["tau_star=min(T_fix)"]
    tasks: dict[str, TaskCheckpoint]
    registry_hash: Sha256


@dataclass(frozen=True, slots=True)
class MainCheckpointReport:
    tasks: tuple[str, ...]
    seed_ids: tuple[int, ...]
    tau_star_by_task: dict[str, int]
    suffix_lengths: dict[str, int]
    orders_sha256: str
    registry_sha256: str


def validate_main_checkpoint_package(root: Path, repository_root: Path) -> MainCheckpointReport:
    try:
        orders_raw = read_regular_nofollow(root / "task_seed_orders_v1.json")
        orders = TaskSeedOrders.model_validate_json(orders_raw)
        expected_orders = _expected_orders(repository_root)
        if orders.model_dump(mode="json", exclude={"orders_hash"}) != expected_orders:
            raise Phase13MainCheckpointError("MAIN_CHECKPOINT_ORDER_INVALID")
        if orders.orders_hash != _canonical_hash(expected_orders):
            raise Phase13MainCheckpointError("MAIN_CHECKPOINT_ORDERS_HASH_INVALID")
        registry_raw = read_regular_nofollow(root / "main_a_common_checkpoint_registry_v1.json")
        registry = CommonCheckpointRegistry.model_validate_json(registry_raw)
        expected_registry = _expected_registry(expected_orders, hashlib.sha256(orders_raw).hexdigest())
        if registry.registry_hash != _canonical_hash(expected_registry):
            raise Phase13MainCheckpointError("MAIN_CHECKPOINT_REGISTRY_HASH_INVALID")
        if registry.model_dump(mode="json", exclude={"registry_hash"}) != expected_registry:
            raise Phase13MainCheckpointError("MAIN_CHECKPOINT_REGISTRY_INVALID")
    except Phase13MainCheckpointError:
        raise
    except (OSError, ValueError) as error:
        raise Phase13MainCheckpointError("MAIN_CHECKPOINT_ARTIFACT_INVALID") from error
    return MainCheckpointReport(
        tasks=tuple(registry.tasks),
        seed_ids=SEEDS,
        tau_star_by_task={task: row.seeds[0].tau_star for task, row in registry.tasks.items()},
        suffix_lengths={task: H_RUN for task in registry.tasks},
        orders_sha256=hashlib.sha256(orders_raw).hexdigest(),
        registry_sha256=hashlib.sha256(registry_raw).hexdigest(),
    )


def production_identity_from_checkpoint(
    root: Path,
    repository_root: Path,
    *,
    task: str,
    trajectory_seed: int,
    execution_template_id: str,
    registration_packet_sha256: str,
) -> ProductionOrdinaryRunIdentity:
    report = validate_main_checkpoint_package(root, repository_root)
    try:
        registry = CommonCheckpointRegistry.model_validate_json(
            read_regular_nofollow(root / "main_a_common_checkpoint_registry_v1.json")
        )
        seed = registry.tasks[task].seeds[trajectory_seed]
    except (KeyError, IndexError, ValueError) as error:
        raise Phase13MainCheckpointError("MAIN_CHECKPOINT_IDENTITY_NOT_REGISTERED") from error
    return ProductionOrdinaryRunIdentity(
        execution_template_id=execution_template_id,
        trajectory_seed=trajectory_seed,
        concrete_seed_id=str(trajectory_seed),
        ordered_sample_ids_sha256=seed.suffix_sample_ids_sha256,
        registration_packet_sha256=registration_packet_sha256,
        scientific_result=False,
        checkpoint_registry_sha256=report.registry_sha256,
    )


def _expected_orders(repository_root: Path) -> dict[str, JsonValue]:
    tasks: dict[str, JsonValue] = {}
    for task in CORE_MAIN_REGISTRY.tasks:
        sample_ids, source = _sample_ids(repository_root, task)
        seeds: list[JsonValue] = []
        for seed in SEEDS:
            ordered = sorted(sample_ids, key=lambda sample_id: _order_digest(task, seed, sample_id))
            ordered_json: list[JsonValue] = []
            ordered_json.extend(ordered)
            seed_payload: dict[str, JsonValue] = {
                "seed": seed,
                "ordered_sample_ids": ordered_json,
                "order_sha256": _list_hash(ordered),
            }
            seeds.append(seed_payload)
        sample_ids_json: list[JsonValue] = []
        sample_ids_json.extend(sample_ids)
        task_payload: dict[str, JsonValue] = {
            "source": source,
            "sample_ids": sample_ids_json,
            "seeds": seeds,
        }
        tasks[task] = task_payload
    return {
        "schema_version": "phase13_task_seed_orders_v1",
        "ordering_law": ORDER_LAW,
        "concrete_seed_ids": list(SEEDS),
        "tasks": tasks,
    }


def _expected_registry(
    orders: dict[str, JsonValue], orders_sha256: str
) -> dict[str, JsonValue]:
    task_rows = orders["tasks"]
    if not isinstance(task_rows, dict):
        raise Phase13MainCheckpointError("MAIN_CHECKPOINT_ORDER_INVALID")
    tasks: dict[str, JsonValue] = {}
    for task, row in task_rows.items():
        if not isinstance(row, dict):
            raise Phase13MainCheckpointError("MAIN_CHECKPOINT_ORDER_INVALID")
        seed_rows = row.get("seeds")
        if not isinstance(seed_rows, list):
            raise Phase13MainCheckpointError("MAIN_CHECKPOINT_ORDER_INVALID")
        seeds: list[JsonValue] = []
        for seed_row in seed_rows:
            if not isinstance(seed_row, dict):
                raise Phase13MainCheckpointError("MAIN_CHECKPOINT_ORDER_INVALID")
            ordered_values = seed_row.get("ordered_sample_ids")
            seed = seed_row.get("seed")
            complete_order_sha256 = seed_row.get("order_sha256")
            if (
                not isinstance(ordered_values, list)
                or not all(isinstance(value, str) for value in ordered_values)
                or type(seed) is not int
                or not isinstance(complete_order_sha256, str)
            ):
                raise Phase13MainCheckpointError("MAIN_CHECKPOINT_ORDER_INVALID")
            ordered = [value for value in ordered_values if isinstance(value, str)]
            tau_star = _tau_star(
                task_stream_length=len(ordered),
                minimum_clean_prefix_length=L_MIN,
                execution_horizon=H_RUN,
                static_route_constraints_satisfied=True,
            )
            prefix, suffix = ordered[: tau_star - 1], ordered[tau_star - 1 : tau_star - 1 + H_RUN]
            prefix_json: list[JsonValue] = []
            prefix_json.extend(prefix)
            suffix_json: list[JsonValue] = []
            suffix_json.extend(suffix)
            checkpoint_payload: dict[str, JsonValue] = {
                "seed": seed,
                "concrete_seed_id": str(seed),
                "tau_star": tau_star,
                "clean_prefix_sample_ids": prefix_json,
                "clean_prefix_sample_ids_sha256": _list_hash(prefix),
                "suffix_sample_ids": suffix_json,
                "suffix_sample_ids_sha256": _list_hash(suffix),
                "complete_order_sha256": complete_order_sha256,
            }
            seeds.append(checkpoint_payload)
        tasks[task] = {
            "route": "3w",
            "L_min": L_MIN,
            "H_run": H_RUN,
            "static_route_constraints": [
                "complete_50_trial_suffix_available",
                "serialized_checkpoint_clone_supported",
                "deterministic_cost_feasibility_pass",
            ],
            "seeds": seeds,
        }
    return {
        "schema_version": "phase13_main_a_common_checkpoint_registry_v1",
        "task_seed_orders": {
            "path": "data/phase13/main/mr_p4/task_seed_orders_v1.json",
            "sha256": orders_sha256,
        },
        "checkpoint_law": "tau_star=min(T_fix)",
        "tasks": tasks,
    }


def _tau_star(
    *,
    task_stream_length: int,
    minimum_clean_prefix_length: int,
    execution_horizon: int,
    static_route_constraints_satisfied: bool,
) -> int:
    first = minimum_clean_prefix_length + 1
    last = task_stream_length - execution_horizon + 1
    feasible = range(first, last + 1) if static_route_constraints_satisfied else ()
    tau_star = min(feasible, default=None)
    if tau_star is None:
        raise Phase13MainCheckpointError("MAIN_CHECKPOINT_ROUTE_INFEASIBLE")
    return tau_star


def _sample_ids(
    repository_root: Path, task: str
) -> tuple[list[str], dict[str, JsonValue]]:
    if task in LEGACY_TASKS:
        path = repository_root / f"data/phase13/main/{task}_main_v1.jsonl"
        sample_ids = [json.loads(line)["sample_id"] for line in path.read_text().splitlines()]
    elif task in MMLU_TASKS:
        path = repository_root / "src/memcontam/readiness/data/mmlu_pro_dc_selection_v1.json"
        selection = json.loads(path.read_text())
        sample_ids = [f"{task}:{question_id}" for question_id in selection["tasks"][task]["question_ids"]]
    else:
        raise Phase13MainCheckpointError("MAIN_CHECKPOINT_TASK_INVALID")
    return sample_ids, {"path": str(path.relative_to(repository_root)), "sha256": _sha256(path)}


def _order_digest(task: str, seed: int, sample_id: str) -> bytes:
    return hashlib.sha256(f"sha256_task_seed_v1\0{task}\0{seed}\0{sample_id}".encode()).digest()


def _list_hash(values: list[str]) -> str:
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def _canonical_hash(payload: dict[str, JsonValue]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = [
    "MainCheckpointReport",
    "Phase13MainCheckpointError",
    "production_identity_from_checkpoint",
    "validate_main_checkpoint_package",
]
