from __future__ import annotations

import json
from typing import Any

from memcontam.baselines import dynamic_cheatsheet_phase12 as dc
from memcontam.memory.cards_v3 import canonical_content_hash
from memcontam.memory.checkpoint_v3 import NativeEntry
from memcontam.tasks.base import TaskInstance


CORE_TASKS = frozenset({"mmlu_pro_engineering", "mmlu_pro_physics", "gpqa_diamond"})


class DcRsRuntimeError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def configured_budget(context: Any) -> int:
    budget = context.baseline_configs.get("dc_rs", {}).get(
        "serialized_cheatsheet_budget_bytes"
    )
    if type(budget) is not int or budget <= 0:
        raise DcRsRuntimeError("DC_RS_CHEATSHEET_BUDGET_REQUIRED")
    return budget


def validate_core_task(task: TaskInstance, code: str) -> None:
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
        _validate_core_input(archive_entry.content, code)
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
        if budget is not None and len(strategy_entry.content.encode("utf-8")) > budget:
            raise DcRsRuntimeError("DC_RS_CHEATSHEET_BUDGET_EXCEEDED")
        strategy_ids.append(strategy_entry.entry_id)
    all_ids = [*archive_ids, *strategy_ids]
    if len(set(all_ids)) != len(all_ids) or (
        state.injected_root_id is not None and state.injected_root_id not in archive_ids
    ):
        raise DcRsRuntimeError(code)


def _validate_core_input(content: str, code: str) -> None:
    try:
        payload = json.loads(content)
    except (TypeError, json.JSONDecodeError) as error:
        raise DcRsRuntimeError(code) from error
    if (
        not isinstance(payload, dict)
        or set(payload) != {"input", "task_name"}
        or payload.get("task_name") not in CORE_TASKS
        or json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        != content
    ):
        raise DcRsRuntimeError(code)
    _validate_core_fields(payload["task_name"], payload["input"], code)


def _validate_core_fields(task_name: object, task_input: object, code: str) -> None:
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
