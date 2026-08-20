from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from memcontam.baselines import dynamic_cheatsheet_phase12 as dc
from memcontam.baselines.prompt_budget import count_text_tokens
from memcontam.memory.cards_v3 import canonical_content_hash
from memcontam.memory.checkpoint_v3 import NativeEntry
from memcontam.readiness.phase13_route_capacity import COMMON_VISIBLE_MEMORY_TOKENS
from memcontam.tasks.base import TaskInstance


CORE_TASKS = frozenset({"mmlu_pro_engineering", "mmlu_pro_physics", "gpqa_diamond"})
ORDINARY_TASKS = frozenset(
    {
        "game24",
        "math_equation_balancer",
        "word_sorting",
        *CORE_TASKS,
    }
)


class DcRsRuntimeError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class OrdinaryHistoryIdentity:
    task_name: str
    run_id: str
    trial_id: str
    order_key: int | str


def configured_budget(context: Any) -> int:
    budget = context.baseline_configs.get("dc_rs", {}).get(
        "serialized_cheatsheet_budget_tokens"
    )
    if budget != COMMON_VISIBLE_MEMORY_TOKENS:
        raise DcRsRuntimeError("DC_RS_CHEATSHEET_BUDGET_REQUIRED")
    return budget


def validate_task(task: TaskInstance, code: str) -> None:
    if task.task_name not in ORDINARY_TASKS:
        raise DcRsRuntimeError(code)
    if task.task_name in CORE_TASKS:
        _validate_core_fields(task.task_name, task.input, code)


def validate_state(state: dc.DcRsStateV3, budget: int | None, code: str) -> None:
    if (
        state.allow_unparented_strategies is not True
        or state.filter_state is not None
        or state.admission_context is not None
    ):
        raise DcRsRuntimeError(code)
    archive_ids: list[str] = []
    for raw_entry in state.archive:
        try:
            archive_entry = dc._archive_entry(raw_entry)
            native = dc._archive_native(archive_entry)
        except dc.DcRsContractError as error:
            raise DcRsRuntimeError(code) from error
        if isinstance(raw_entry, NativeEntry) and raw_entry != native:
            raise DcRsRuntimeError(code)
        if "tool_trace" in archive_entry.metadata:
            raise DcRsRuntimeError(code)
        _validate_archive_input(archive_entry.content, code)
        archive_ids.append(archive_entry.entry_id)
    strategy_ids: list[str] = []
    for raw_entry in state.strategies or ():
        try:
            strategy_entry = dc._strategy_entry(raw_entry, allow_unparented=True)
        except dc.DcRsContractError as error:
            raise DcRsRuntimeError(code) from error
        if canonical_content_hash(strategy_entry.content) != strategy_entry.content_hash:
            raise DcRsRuntimeError(code)
        if not set(strategy_entry.direct_parent_ids).issubset(archive_ids):
            raise DcRsRuntimeError(code)
        if budget is not None and count_text_tokens(strategy_entry.content, "o200k_base") > budget:
            raise DcRsRuntimeError("DC_RS_CHEATSHEET_BUDGET_EXCEEDED")
        strategy_ids.append(strategy_entry.entry_id)
    all_ids = [*archive_ids, *strategy_ids]
    if len(set(all_ids)) != len(all_ids) or (
        state.injected_root_id is not None and state.injected_root_id not in archive_ids
    ):
        raise DcRsRuntimeError(code)


def validate_ordinary_history(
    state: dc.DcRsStateV3,
    identity: OrdinaryHistoryIdentity,
) -> None:
    current_index = _trajectory_index(identity.trial_id, identity.run_id)
    if type(identity.order_key) is not int or current_index != identity.order_key:
        raise DcRsRuntimeError("DC_RS_ORDINARY_HISTORY_UNPROVEN")
    if not state.archive:
        return
    for raw_entry in state.archive:
        archive_entry = dc._archive_entry(raw_entry)
        if (
            _validate_archive_input(
                archive_entry.content,
                "DC_RS_ORDINARY_HISTORY_UNPROVEN",
            )
            != identity.task_name
            or archive_entry.source_trial_id is None
        ):
            raise DcRsRuntimeError("DC_RS_ORDINARY_HISTORY_UNPROVEN")
        source_index = _trajectory_index(archive_entry.source_trial_id, identity.run_id)
        if source_index >= current_index:
            raise DcRsRuntimeError("DC_RS_ORDINARY_HISTORY_UNPROVEN")


def _trajectory_index(trial_id: str, run_id: str) -> int:
    prefix = f"{run_id}:trial:"
    if not trial_id.startswith(prefix):
        raise DcRsRuntimeError("DC_RS_ORDINARY_HISTORY_UNPROVEN")
    raw_index, separator, _suffix = trial_id.removeprefix(prefix).partition(":")
    if not separator:
        raise DcRsRuntimeError("DC_RS_ORDINARY_HISTORY_UNPROVEN")
    try:
        index = int(raw_index)
    except ValueError as error:
        raise DcRsRuntimeError("DC_RS_ORDINARY_HISTORY_UNPROVEN") from error
    if index < 1:
        raise DcRsRuntimeError("DC_RS_ORDINARY_HISTORY_UNPROVEN")
    return index


def _validate_archive_input(content: str, code: str) -> str:
    try:
        payload = json.loads(content)
    except (TypeError, json.JSONDecodeError) as error:
        raise DcRsRuntimeError(code) from error
    if not isinstance(payload, dict) or payload.get("task_name") not in ORDINARY_TASKS:
        raise DcRsRuntimeError(code)
    task_name = payload["task_name"]
    if task_name in CORE_TASKS and (
        set(payload) != {"input", "task_name"}
        or json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        != content
    ):
        raise DcRsRuntimeError(code)
    if task_name in CORE_TASKS:
        _validate_core_fields(task_name, payload["input"], code)
    elif (
        set(payload) != {"input", "metadata", "sample_id", "task_name"}
        or not isinstance(payload.get("sample_id"), str)
        or not payload["sample_id"]
        or not isinstance(payload.get("input"), dict)
        or not isinstance(payload.get("metadata"), dict)
        or json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        != content
    ):
        raise DcRsRuntimeError(code)
    return task_name


def _validate_core_fields(task_name: Any, task_input: Any, code: str) -> None:
    if (
        not isinstance(task_name, str)
        or task_name not in CORE_TASKS
        or not isinstance(task_input, dict)
        or set(task_input) != {"options", "question"}
    ):
        raise DcRsRuntimeError(code)
    question = task_input.get("question")
    options = task_input.get("options")
    if (
        not isinstance(question, str)
        or not question.strip()
        or not isinstance(options, list)
        or not 2 <= len(options) <= 26
        or any(not isinstance(option, str) or not option.strip() for option in options)
    ):
        raise DcRsRuntimeError(code)
