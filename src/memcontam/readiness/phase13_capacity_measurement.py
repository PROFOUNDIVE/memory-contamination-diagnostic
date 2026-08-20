from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Literal, assert_never

from pydantic import BaseModel, ConfigDict

from memcontam.baselines.dynamic_cheatsheet_phase12 import core_synthesis_message
from memcontam.baselines.full_history import FullHistoryState
from memcontam.baselines.full_history_adapter import _messages as full_history_messages
from memcontam.baselines.prompt_budget import count_prompt_tokens, count_text_tokens
from memcontam.main_registry import Game24MainRow, MebMainRow, WordSortingMainRow
from memcontam.memory.stores import MemoryEntry
from memcontam.readiness.phase13_route_capacity import COMMON_VISIBLE_MEMORY_TOKENS
from memcontam.tasks.base import TaskInstance
from memcontam.tasks.dispatch import canonical_core_task_json, canonical_task_json

TaskName = Literal[
    "game24",
    "math_equation_balancer",
    "word_sorting",
    "mmlu_pro_engineering",
    "mmlu_pro_physics",
    "gpqa_diamond",
]

TASKS: tuple[TaskName, ...] = (
    "game24",
    "math_equation_balancer",
    "word_sorting",
    "mmlu_pro_engineering",
    "mmlu_pro_physics",
    "gpqa_diamond",
)
CORE_TASKS = frozenset({"mmlu_pro_engineering", "mmlu_pro_physics", "gpqa_diamond"})
TOKEN_ENCODING = "o200k_base"
ANSWER_OUTPUT_TOKENS = 4096
_MEASUREMENT_IMPLEMENTATION_PATHS = (
    "src/memcontam/readiness/phase13_capacity_measurement.py",
    "src/memcontam/baselines/prompt_budget.py",
    "src/memcontam/baselines/full_history_adapter.py",
    "src/memcontam/baselines/full_history_context.py",
    "src/memcontam/baselines/full_history_phase12.py",
    "src/memcontam/baselines/dynamic_cheatsheet_phase12.py",
    "src/memcontam/baselines/dynamic_cheatsheet_optional.py",
    "src/memcontam/experiment/phase12/runtime_registry.py",
    "src/memcontam/experiment/phase13_dc_rs_validation.py",
    "src/memcontam/experiment/phase13_ordinary_runtime.py",
    "src/memcontam/clients/provider_profile.py",
    "src/memcontam/readiness/phase13_route_capacity.py",
    "src/memcontam/main_registry.py",
    "src/memcontam/tasks/base.py",
    "src/memcontam/tasks/dispatch.py",
)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")


class _LegacyArtifact(_FrozenModel):
    path: str
    sha256: str
    main_count: int


class _LegacyManifest(_FrozenModel):
    registries: dict[str, _LegacyArtifact]


class _CoreArtifact(_FrozenModel):
    path: str
    sha256: str
    rows: int


class _CoreManifest(_FrozenModel):
    artifacts: dict[str, _CoreArtifact]


@dataclass(frozen=True, slots=True)
class CapacityReserves:
    per_task_R_FH: dict[str, int]
    per_task_I_DC_writer: dict[str, int]


class CapacityMeasurementError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def derive_capacity_reserves(repository_root: Path) -> CapacityReserves:
    tasks = _load_frozen_tasks(repository_root)
    return CapacityReserves(
        per_task_R_FH={task: _fh_reserve(rows) for task, rows in tasks.items()},
        per_task_I_DC_writer={
            task: _dc_writer_reserve(task, rows) for task, rows in tasks.items()
        },
    )


def measurement_implementation_sha256(repository_root: Path) -> str:
    digest = hashlib.sha256(b"phase13-capacity-measurement-v1\0")
    for relative in _MEASUREMENT_IMPLEMENTATION_PATHS:
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update((repository_root / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _load_frozen_tasks(repository_root: Path) -> dict[TaskName, tuple[TaskInstance, ...]]:
    main_root = repository_root / "data/phase13/main"
    legacy = _LegacyManifest.model_validate_json(
        (main_root / "main_registry_manifest_v1.json").read_bytes()
    )
    core_root = repository_root / "data/phase13/core/materialized"
    core = _CoreManifest.model_validate_json((core_root / "manifest.json").read_bytes())
    loaded: dict[TaskName, tuple[TaskInstance, ...]] = {}
    for task in TASKS:
        if task in CORE_TASKS:
            artifact = core.artifacts[task]
            path = _bound_path(core_root, artifact.path)
            _verify_artifact(path, artifact.sha256)
            rows = tuple(TaskInstance.model_validate_json(line) for line in _lines(path))
            expected_count = artifact.rows
        else:
            artifact = legacy.registries[task]
            path = _bound_path(main_root, artifact.path)
            _verify_artifact(path, artifact.sha256)
            rows = _legacy_tasks(task, _lines(path))
            expected_count = artifact.main_count
        if len(rows) != expected_count or any(row.task_name != task for row in rows):
            raise CapacityMeasurementError("CAPACITY_SOURCE_REGISTRY_MISMATCH")
        loaded[task] = rows
    return loaded


def _legacy_tasks(task: TaskName, lines: Sequence[str]) -> tuple[TaskInstance, ...]:
    match task:
        case "game24":
            rows = (Game24MainRow.model_validate_json(line) for line in lines)
            return tuple(
                TaskInstance(
                    sample_id=row.sample_id,
                    task_name=task,
                    input={"numbers": list(row.numbers)},
                    verifier_spec={"target": row.target},
                )
                for row in rows
            )
        case "math_equation_balancer":
            rows = (MebMainRow.model_validate_json(line) for line in lines)
            return tuple(
                TaskInstance(
                    sample_id=row.sample_id,
                    task_name=task,
                    input={"input": row.input},
                    verifier_spec=row.verifier_spec.model_dump(),
                )
                for row in rows
            )
        case "word_sorting":
            rows = (WordSortingMainRow.model_validate_json(line) for line in lines)
            return tuple(
                TaskInstance(
                    sample_id=row.sample_id,
                    task_name=task,
                    input={"words": list(row.words)},
                    verifier_spec={"sorted_words": list(row.sorted_words)},
                )
                for row in rows
            )
        case "mmlu_pro_engineering" | "mmlu_pro_physics" | "gpqa_diamond":
            raise CapacityMeasurementError("CAPACITY_SOURCE_REGISTRY_MISMATCH")
        case unreachable:
            assert_never(unreachable)


def _fh_reserve(tasks: Sequence[TaskInstance]) -> int:
    return max(
        count_prompt_tokens(
            full_history_messages(task, FullHistoryState(), {})[0], TOKEN_ENCODING
        )
        + ANSWER_OUTPUT_TOKENS
        for task in tasks
    )


def _dc_writer_reserve(task_name: TaskName, tasks: Sequence[TaskInstance]) -> int:
    raw_answer = " x" * COMMON_VISIBLE_MEMORY_TOKENS
    if count_text_tokens(raw_answer, TOKEN_ENCODING) != COMMON_VISIBLE_MEMORY_TOKENS:
        raise CapacityMeasurementError("CAPACITY_TOKEN_FILLER_MISMATCH")
    prior = MemoryEntry(
        entry_id="strategy",
        content=_complete_cheatsheet(COMMON_VISIBLE_MEMORY_TOKENS),
        memory_type="strategy",
    )
    canonical_tasks = tuple(
        canonical_core_task_json(task)
        if task_name in CORE_TASKS
        else canonical_task_json(task)
        for task in tasks
    )
    largest_first = sorted(
        range(len(canonical_tasks)),
        key=lambda index: count_text_tokens(canonical_tasks[index], TOKEN_ENCODING),
        reverse=True,
    )
    values: list[int] = []
    for current_index, canonical in enumerate(canonical_tasks):
        prior_indices = tuple(index for index in largest_first if index != current_index)[:3]
        if len(prior_indices) != 3:
            raise CapacityMeasurementError("CAPACITY_SOURCE_REGISTRY_MISMATCH")
        pairs = tuple(
            MemoryEntry(
                entry_id=f"archive-{index}",
                content=canonical_tasks[index],
                memory_type="dc_rs_io_pair",
                metadata={"generated_output": raw_answer},
            )
            for index in prior_indices
        )
        message, _spans, _aliases = core_synthesis_message(canonical, prior, pairs)
        values.append(count_prompt_tokens([message], TOKEN_ENCODING))
    return max(values)


def _complete_cheatsheet(token_budget: int) -> str:
    low = 0
    high = token_budget
    while low <= high:
        size = (low + high) // 2
        candidate = f"<cheatsheet>{' x' * size}</cheatsheet>"
        tokens = count_text_tokens(candidate, TOKEN_ENCODING)
        if tokens == token_budget:
            return candidate
        if tokens < token_budget:
            low = size + 1
        else:
            high = size - 1
    raise CapacityMeasurementError("CAPACITY_TOKEN_FILLER_MISMATCH")


def _bound_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise CapacityMeasurementError("CAPACITY_SOURCE_REGISTRY_MISMATCH")
    return root / path


def _verify_artifact(path: Path, expected_sha256: str) -> None:
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise CapacityMeasurementError("CAPACITY_SOURCE_REGISTRY_MISMATCH")


def _lines(path: Path) -> tuple[str, ...]:
    return tuple(line for line in path.read_text(encoding="utf-8").splitlines() if line)


__all__ = [
    "COMMON_VISIBLE_MEMORY_TOKENS",
    "CapacityReserves",
    "derive_capacity_reserves",
    "measurement_implementation_sha256",
]
