from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Literal

from memcontam.readiness.phase13_execution_contract import CORE_MAIN_REGISTRY, ExecutionRegistry

COMMON_VISIBLE_MEMORY_TOKENS = CORE_MAIN_REGISTRY.writer_max_output_tokens
CAPACITY_AUDIT_TASKS = (
    "game24",
    "math_equation_balancer",
    "word_sorting",
    "mmlu_pro_engineering",
    "mmlu_pro_physics",
    "gpqa_diamond",
)
CapacityContractErrorCode = Literal[
    "FH_CAPACITY_CONTRACT_MISMATCH",
    "DC_RS_CAPACITY_CONTRACT_MISMATCH",
]
FH_CAPACITY_CONFIG = {
    "mode": "context_bounded_pair_atomic",
    "token_encoding": "o200k_base",
    "context_window_tokens": 1_050_000,
    "max_output_tokens": 4096,
    "fixed_prompt_overhead_tokens": 0,
    "safety_margin_tokens": 0,
    "history_capacity_tokens": COMMON_VISIBLE_MEMORY_TOKENS,
    "eviction_policy": "oldest_first_pair_atomic",
    "fh_mode": "bounded",
    "context_budget_id": "luna-common-visible-memory-capacity-v1",
}


class CapacityPlanningError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def capacity_contract_error(
    configs: Mapping[str, Mapping[str, Any]],
    visible_memory_tokens: int,
) -> CapacityContractErrorCode | None:
    fh_config = configs.get("fh_bounded", {})
    expected_fh = {
        **FH_CAPACITY_CONFIG,
        "history_capacity_tokens": visible_memory_tokens,
    }
    if any(fh_config.get(key, value) != value for key, value in expected_fh.items()):
        return "FH_CAPACITY_CONTRACT_MISMATCH"
    dc_budget = configs.get("dc_rs", {}).get(
        "serialized_cheatsheet_budget_tokens", visible_memory_tokens
    )
    if dc_budget != visible_memory_tokens:
        return "DC_RS_CAPACITY_CONTRACT_MISMATCH"
    return None


def bind_capacity_configs(
    configs: Mapping[str, Mapping[str, Any]],
    visible_memory_tokens: int,
) -> Mapping[str, Mapping[str, Any]]:
    bound = {name: dict(config) for name, config in configs.items()}
    bound["fh_bounded"] = {
        **bound.get("fh_bounded", {}),
        **FH_CAPACITY_CONFIG,
        "history_capacity_tokens": visible_memory_tokens,
    }
    bound["dc_rs"] = {
        **bound.get("dc_rs", {}),
        "serialized_cheatsheet_budget_tokens": visible_memory_tokens,
    }
    return bound


@dataclass(frozen=True, slots=True)
class CapacityPlan:
    nominal_semantic_calls: int
    raw_maximum_semantic_calls: int
    reserved_semantic_calls: int
    raw_maximum_transport_attempts: int
    reserved_transport_attempts: int
    maximum_input_tokens: int
    maximum_output_tokens: int


@dataclass(frozen=True, slots=True)
class CommonCapacityAudit:
    model_runtime_identity: str
    context_contract_id: str
    context_tokens: int
    provider_output_contract_id: str
    provider_max_output_tokens: int
    tokenizer_encoding_identity: str
    tokenizer_revision_version: str
    serialization_identity: str
    special_token_handling_identity: str
    message_framing_law_identity: str
    token_count_implementation_hash_version: str
    per_task_R_FH: Mapping[str, int]
    per_task_I_DC_writer: Mapping[str, int]
    per_task_F_DC_out: Mapping[str, int]
    fh_bounded_core_contract_id: str
    retention_truncation_rule_id: str
    context_resource_contract_id: str


@dataclass(frozen=True, slots=True)
class CommonCapacityRecord:
    audit: CommonCapacityAudit
    B_FH_feasible: int
    B_DC_feasible: int
    B_mem_tokens: int
    L_DC_tokens: int
    capacity_law_hash: str


def materialize_common_capacity(audit: CommonCapacityAudit) -> CommonCapacityRecord:
    task_set = set(CAPACITY_AUDIT_TASKS)
    reserve_rows = (
        audit.per_task_R_FH,
        audit.per_task_I_DC_writer,
        audit.per_task_F_DC_out,
    )
    identities = (
        audit.model_runtime_identity,
        audit.context_contract_id,
        audit.provider_output_contract_id,
        audit.tokenizer_encoding_identity,
        audit.tokenizer_revision_version,
        audit.serialization_identity,
        audit.special_token_handling_identity,
        audit.message_framing_law_identity,
        audit.token_count_implementation_hash_version,
        audit.fh_bounded_core_contract_id,
        audit.retention_truncation_rule_id,
        audit.context_resource_contract_id,
    )
    if (
        any(set(rows) != task_set for rows in reserve_rows)
        or any(type(value) is not int or value < 0 for rows in reserve_rows for value in rows.values())
        or audit.context_tokens <= 0
        or audit.provider_max_output_tokens < CORE_MAIN_REGISTRY.writer_max_output_tokens
        or any(not identity for identity in identities)
    ):
        raise CapacityPlanningError("COMMON_CAPACITY_AUDIT_INVALID")
    b_fh = min(audit.context_tokens - audit.per_task_R_FH[task] for task in task_set)
    b_dc = min(
        min(
            CORE_MAIN_REGISTRY.writer_max_output_tokens - audit.per_task_F_DC_out[task],
            audit.context_tokens
            - audit.per_task_I_DC_writer[task]
            - audit.per_task_F_DC_out[task],
        )
        for task in task_set
    )
    b_mem = min(b_fh, b_dc)
    if b_mem <= 0:
        raise CapacityPlanningError("COMMON_CAPACITY_NOT_READY")
    payload = {
        "audit": asdict(audit),
        "B_FH_feasible": b_fh,
        "B_DC_feasible": b_dc,
        "B_mem_tokens": b_mem,
        "L_DC_tokens": b_mem,
        "capacity_law_id": CORE_MAIN_REGISTRY.capacity_law_id,
        "capacity_unit": CORE_MAIN_REGISTRY.capacity_unit,
        "O_writer_reg": CORE_MAIN_REGISTRY.writer_max_output_tokens,
    }
    capacity_law_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return CommonCapacityRecord(audit, b_fh, b_dc, b_mem, b_mem, capacity_law_hash)


def recompute_capacity(
    execution: ExecutionRegistry,
    attempted_seed_counts: dict[str, int],
) -> CapacityPlan:
    if set(attempted_seed_counts) != set(execution.tasks) or any(
        type(count) is not int or count < 0 for count in attempted_seed_counts.values()
    ):
        raise CapacityPlanningError("ATTEMPTED_SEED_COUNTS_INVALID")
    nominal_by_task = {
        task: sum(
            row.nominal_semantic_calls_per_trial
            for row in execution.templates
            if row.task == task
        )
        for task in execution.tasks
    }
    maximum_by_task = {
        task: sum(
            row.maximum_semantic_calls_per_trial
            for row in execution.templates
            if row.task == task
        )
        for task in execution.tasks
    }
    nominal = sum(
        count
        * (
            execution.capacity.prefix_nominal_calls_per_seed
            + execution.H_run * nominal_by_task[task]
        )
        for task, count in attempted_seed_counts.items()
    )
    maximum = sum(
        count
        * (
            execution.capacity.prefix_maximum_calls_per_seed
            + execution.H_run * maximum_by_task[task]
        )
        for task, count in attempted_seed_counts.items()
    )
    reserved = math.ceil(maximum * (100 + execution.capacity.reserve_percent) / 100)
    attempts = execution.capacity.maximum_transport_attempts_per_semantic_call
    return CapacityPlan(
        nominal,
        maximum,
        reserved,
        maximum * attempts,
        reserved * attempts,
        reserved * attempts * execution.capacity.maximum_input_tokens_per_transport_attempt,
        reserved * attempts * execution.capacity.maximum_output_tokens_per_transport_attempt,
    )
