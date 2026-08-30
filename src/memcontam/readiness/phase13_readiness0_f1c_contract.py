from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypeAlias, assert_never

from memcontam.baselines.bot_read import COARSE_THOUGHT_STRUCTURES
from memcontam.readiness.phase13_legacy_rag_models import BRANCHES
from memcontam.readiness.phase13_main_checkpoint import CommonCheckpointRegistry
from memcontam.readiness.phase13_readiness0_f1c_models import Arm, RetrievalBaseline, Task
from memcontam.tasks.dispatch import canonical_task_json
from memcontam.tasks.game24 import build_instance as build_game24
from memcontam.tasks.math_equation_balancer import build_instance as build_meb
from memcontam.tasks.multiple_choice import build_instance as build_multiple_choice
from memcontam.tasks.word_sorting import build_instance as build_word_sorting


TASKS: Final[tuple[Task, ...]] = (
    "game24",
    "math_equation_balancer",
    "word_sorting",
    "mmlu_pro_engineering",
    "mmlu_pro_physics",
)
LEGACY_TASKS: Final[tuple[Task, ...]] = (
    "game24",
    "math_equation_balancer",
    "word_sorting",
)
ARMS: Final[tuple[Arm, ...]] = BRANCHES
CanonicalInput: TypeAlias = (
    None | bool | int | float | str | Sequence["CanonicalInput"] | Mapping[str, "CanonicalInput"]
)


class F1CReportError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class QuerySpec:
    text: str
    sample_id: str
    source: str


def canonical_hash(payload: CanonicalInput) -> str:
    return text_hash(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint(root: Path) -> CommonCheckpointRegistry:
    return CommonCheckpointRegistry.model_validate_json(
        (root / "data/phase13/main/mr_p4/main_a_common_checkpoint_registry_v1.json").read_bytes()
    )


def queries(root: Path) -> dict[Task, QuerySpec]:
    registry = checkpoint(root)
    result: dict[Task, QuerySpec] = {}
    for task in TASKS:
        sample_id = registry.tasks[task].seeds[0].suffix_sample_ids[0]
        source = (
            f"data/phase13/main/{task}_main_v1.jsonl"
            if task in LEGACY_TASKS
            else f"data/phase13/core/materialized/{task}.jsonl"
        )
        rows = (root / source).read_text(encoding="utf-8").splitlines()
        row = next(parsed for line in rows if (parsed := json.loads(line))["sample_id"] == sample_id)
        match task:
            case "game24":
                instance = build_game24(row)
            case "math_equation_balancer":
                instance = build_meb(row)
            case "word_sorting":
                instance = build_word_sorting(row)
            case "mmlu_pro_engineering" | "mmlu_pro_physics":
                instance = build_multiple_choice(row)
            case unreachable:
                assert_never(unreachable)
        text = canonical_task_json(instance)
        result[task] = QuerySpec(text=text, sample_id=sample_id, source=source)
    return result


def memory_candidates(
    root: Path,
    task: Task,
    arm: Arm,
    baseline: RetrievalBaseline,
) -> tuple[tuple[str, str], ...]:
    if baseline == "bot_style":
        return tuple((f"bot::{name}", description) for name, description in COARSE_THOUGHT_STRUCTURES)
    prefix = checkpoint(root).tasks[task].seeds[0].clean_prefix_sample_ids
    return tuple((f"dc-rs::{arm}::{sample_id}", sample_id) for sample_id in prefix)


__all__ = [
    "ARMS",
    "F1CReportError",
    "LEGACY_TASKS",
    "QuerySpec",
    "TASKS",
    "canonical_hash",
    "file_hash",
    "memory_candidates",
    "queries",
    "text_hash",
]
