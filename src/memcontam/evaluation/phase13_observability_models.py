from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from memcontam.logging.schema import MemoryEvent, PromptSourceSpan
from memcontam.logging.schema_v3 import ContextEvent, MemoryBranchTrialLog, RetrievalEvent


Baseline = Literal["fh_bounded", "rag_frozen", "bot_style", "reflexion_style", "dc_rs"]
Task = Literal[
    "game24",
    "math_equation_balancer",
    "word_sorting",
    "mmlu_pro_engineering",
    "mmlu_pro_physics",
]
Arm = Literal["clean", "correct", "irrelevant", "contam"]
AggregateArm = Literal["clean", "correct", "irrelevant", "contam", "nomem"]
AggregateBaseline = Baseline | Literal["nomem"]
EvidenceScope = Literal["synthetic_contract_fixture", "production_runtime"]
MetricStatus = Literal[
    "supported", "not_applicable", "not_estimable", "unavailable", "not_registered"
]


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class Phase13ObservabilityError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class MetricValue(_FrozenModel):
    status: MetricStatus
    value: bool | int | float | str | None = None
    reason: str
    path: tuple[str, ...] = ()
    censoring_status: Literal["OBSERVED_END", "RIGHT_CENSORED"] | None = None
    censoring_endpoint_analysis_window_id: Literal["core_prefix_50"] | None = None


class Phase13TargetSetEvidence(_FrozenModel):
    target_set_id: str = Field(min_length=1)
    target_entry_ids: tuple[str, ...]
    answer_call_id: str | None = None
    answer_call_spans: tuple[PromptSourceSpan, ...] = ()
    source_package_manifest_sha256: str | None = None


class Phase13LineageNode(_FrozenModel):
    entry_id: str = Field(min_length=1)
    lineage_status: Literal["exact", "approximate", "unavailable"]
    injected_root_ids: tuple[str, ...] = ()
    direct_parent_ids: tuple[str, ...] = ()
    version_predecessor_id: str | None = None


class Phase13TrialEvidence(_FrozenModel):
    schema_version: Literal["phase13_trial_evidence_v1"] = "phase13_trial_evidence_v1"
    evidence_scope: EvidenceScope
    task: Task
    baseline: Baseline
    trajectory_seed: int = Field(ge=0)
    concrete_seed_id: str = Field(min_length=1)
    analysis_window_id: Literal["core_prefix_50"]
    trial_id: str = Field(min_length=1)
    order_key: int = Field(ge=0)
    trial: MemoryBranchTrialLog
    retrievals: tuple[RetrievalEvent, ...] = ()
    context: ContextEvent | None
    target_set: Phase13TargetSetEvidence
    verified_outcome: Literal[0, 1] | None
    memory_before_ids: tuple[str, ...]
    memory_after_ids: tuple[str, ...]
    new_entry_ids: tuple[str, ...] = ()
    updated_entry_ids: tuple[str, ...] = ()
    removed_entry_ids: tuple[str, ...] = ()
    memory_events: tuple[MemoryEvent, ...] = ()
    lineage: tuple[Phase13LineageNode, ...]

    @model_validator(mode="after")
    def _identity_integrity(self) -> Phase13TrialEvidence:
        if self.evidence_scope == "production_runtime" and (
            (self.trial.execution_status == "failed") != (self.verified_outcome is None)
        ):
            raise Phase13ObservabilityError("TECHNICAL_MISSINGNESS_OUTCOME_MISMATCH")
        identity_sets = (
            self.memory_before_ids,
            self.memory_after_ids,
            self.new_entry_ids,
            self.updated_entry_ids,
            self.removed_entry_ids,
        )
        if any(len(values) != len(set(values)) for values in identity_sets):
            raise Phase13ObservabilityError("DUPLICATE_ENTRY_ID")
        nodes = {node.entry_id: node for node in self.lineage}
        if len(nodes) != len(self.lineage):
            raise Phase13ObservabilityError("DUPLICATE_LINEAGE_NODE")
        for node in self.lineage:
            references = (*node.direct_parent_ids, node.version_predecessor_id)
            if node.lineage_status == "exact" and any(
                reference is not None and reference not in nodes for reference in references
            ):
                raise Phase13ObservabilityError("FABRICATED_LINEAGE")
        return self


class Phase13TrialAnalysis(_FrozenModel):
    schema_version: Literal["phase13_trial_analysis_v1"] = "phase13_trial_analysis_v1"
    evidence_scope: EvidenceScope
    task: Task
    baseline: Baseline
    arm: Arm
    trajectory_seed: int
    concrete_seed_id: str
    analysis_window_id: Literal["core_prefix_50"]
    trial_id: str
    order_key: int
    target_set_id: str
    target_present_in_store_before_answer: MetricValue
    target_retrieved: MetricValue
    target_final_context_included: MetricValue
    theory_exposure: MetricValue
    operational_use: MetricValue
    verified_outcome: Literal[0, 1]
    failure_class: MetricValue
    root_entry_ids: tuple[str, ...]
    descendant_entry_ids: tuple[str, ...]
    memory_before_ids: tuple[str, ...]
    memory_after_ids: tuple[str, ...]
    new_entry_ids: tuple[str, ...]
    updated_entry_ids: tuple[str, ...]
    removed_entry_ids: tuple[str, ...]
    generic_recurrence: MetricValue
    exact_lineage_recurrence: MetricValue
    exposure_conditioned_recurrence: MetricValue
    post_eviction_recurrence: MetricValue
    root_storage_persistence: MetricValue
    descendant_storage_persistence: MetricValue
    root_prompt_visibility: MetricValue
    descendant_prompt_visibility: MetricValue
    root_retention_duration: MetricValue
    prompt_retention_duration: MetricValue
    descendant_retention_duration: MetricValue
    propagation: MetricValue


class Phase13AggregateTrial(_FrozenModel):
    evidence_scope: Literal["synthetic_contract_fixture"]
    task: Task
    baseline: AggregateBaseline
    arm: AggregateArm
    trajectory_seed: int = Field(ge=0)
    concrete_seed_id: str = Field(min_length=1)
    analysis_window_id: Literal["core_prefix_50"]
    source_trial_count: Literal[50]
    structural_support: bool
    verified_outcome: Literal[0, 1]
    target_present_in_store_before_answer: MetricValue = MetricValue(
        status="unavailable", reason="PRODUCTION_RUNTIME_METADATA_JOIN_NOT_MATERIALIZED"
    )
    target_retrieved: MetricValue = MetricValue(
        status="unavailable", reason="PRODUCTION_RUNTIME_METADATA_JOIN_NOT_MATERIALIZED"
    )
    target_final_context_included: MetricValue = MetricValue(
        status="unavailable", reason="PRODUCTION_RUNTIME_METADATA_JOIN_NOT_MATERIALIZED"
    )
    theory_exposure: MetricValue = MetricValue(
        status="unavailable", reason="PRODUCTION_RUNTIME_METADATA_JOIN_NOT_MATERIALIZED"
    )
    generic_recurrence: MetricValue = MetricValue(
        status="unavailable", reason="PRODUCTION_RUNTIME_METADATA_JOIN_NOT_MATERIALIZED"
    )
    exact_lineage_recurrence: MetricValue = MetricValue(
        status="unavailable", reason="PRODUCTION_RUNTIME_METADATA_JOIN_NOT_MATERIALIZED"
    )
    exposure_conditioned_recurrence: MetricValue = MetricValue(
        status="unavailable", reason="PRODUCTION_RUNTIME_METADATA_JOIN_NOT_MATERIALIZED"
    )
    post_eviction_recurrence: MetricValue = MetricValue(
        status="unavailable", reason="PRODUCTION_RUNTIME_METADATA_JOIN_NOT_MATERIALIZED"
    )
    root_storage_persistence: MetricValue = MetricValue(
        status="unavailable", reason="PRODUCTION_RUNTIME_METADATA_JOIN_NOT_MATERIALIZED"
    )
    descendant_storage_persistence: MetricValue = MetricValue(
        status="unavailable", reason="PRODUCTION_RUNTIME_METADATA_JOIN_NOT_MATERIALIZED"
    )
    root_prompt_visibility: MetricValue = MetricValue(
        status="unavailable", reason="PRODUCTION_RUNTIME_METADATA_JOIN_NOT_MATERIALIZED"
    )
    descendant_prompt_visibility: MetricValue = MetricValue(
        status="unavailable", reason="PRODUCTION_RUNTIME_METADATA_JOIN_NOT_MATERIALIZED"
    )
    root_retention_duration: MetricValue = MetricValue(
        status="unavailable", reason="PRODUCTION_RUNTIME_METADATA_JOIN_NOT_MATERIALIZED"
    )
    prompt_retention_duration: MetricValue = MetricValue(
        status="unavailable", reason="PRODUCTION_RUNTIME_METADATA_JOIN_NOT_MATERIALIZED"
    )
    descendant_retention_duration: MetricValue = MetricValue(
        status="unavailable", reason="PRODUCTION_RUNTIME_METADATA_JOIN_NOT_MATERIALIZED"
    )
    propagation: MetricValue = MetricValue(
        status="unavailable", reason="PRODUCTION_RUNTIME_METADATA_JOIN_NOT_MATERIALIZED"
    )


class Phase13AggregateCell(_FrozenModel):
    task: Task
    baseline: AggregateBaseline
    attempted_seed_count: Literal[10]
    supported_seed_count_by_arm: dict[str, int]
    verified_accuracy_by_arm: dict[str, MetricValue]
    contrasts: dict[str, MetricValue]
    observability_rates: dict[str, dict[str, MetricValue]]
    exposure_conditional_diagnostic: MetricValue


class Phase13Aggregate(_FrozenModel):
    schema_version: Literal["phase13_aggregate_v1"] = "phase13_aggregate_v1"
    evidence_scope: Literal["synthetic_contract_fixture"]
    cells: tuple[Phase13AggregateCell, ...]


__all__ = [
    "AggregateArm",
    "AggregateBaseline",
    "Arm",
    "Baseline",
    "MetricValue",
    "Phase13Aggregate",
    "Phase13AggregateCell",
    "Phase13AggregateTrial",
    "Phase13LineageNode",
    "Phase13ObservabilityError",
    "Phase13TargetSetEvidence",
    "Phase13TrialAnalysis",
    "Phase13TrialEvidence",
    "Task",
]
