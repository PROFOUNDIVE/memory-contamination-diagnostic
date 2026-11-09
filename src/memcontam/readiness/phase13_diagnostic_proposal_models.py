from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from memcontam.tasks.base import TaskInstance

from .phase13_cost_policy_models import Sha256
from .phase13_diagnostic_cli_models import RegistrationSubject
from .phase13_live_diagnostic_models import DiagnosticModel
from .phase13_shared_budget_models import FilePin

Task = Literal["math_equation_balancer", "word_sorting"]
Baseline = Literal["bot_style", "dc_rs"]
Arm = Literal["clean", "correct", "irrelevant", "contam"]


class ProposalProjection(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)


class ProposalHeader(ProposalProjection):
    schema_id: str = Field(default="", alias="schema")


class ProposalRender(ProposalProjection):
    carrier: str
    renderer_id: str | None
    render_id: str | None
    family_binding: FilePin
    native_semantic_kind: str | None = None
    serialization_id: str | None = None
    triplet_id: str | None = None
    candidate_id: str | None = None
    query: str | None = None
    response: str | None = None
    query_sha256: Sha256 | None = None
    response_sha256: Sha256 | None = None
    serialized_content_sha256: Sha256 | None = None


class ProposalUnit(ProposalProjection):
    unit_id: str
    task: Task
    baseline: Baseline
    arm: Arm
    seed: int
    group_id: str
    sample_id: str
    partition: Literal["D_cal"]
    input_binding: FilePin
    inventory_index: int
    source_state_sha256: Sha256 | None
    prefix_id: str | None
    checkpoint_id: str | None
    root_count: Literal[0, 1]
    render: ProposalRender | None
    observation_steps: Literal[1]


class ProposalGroup(ProposalProjection):
    group_id: str
    task: Task
    baseline: Baseline
    seed: int
    arm_order: tuple[Arm, ...]
    suffix_sample_id: str
    prefix_id: str | None
    prefix_sample_ids: tuple[str, ...] | None
    checkpoint_id: str | None
    checkpoint_index: int | None
    state_sha256: Sha256 | None
    state_binding: FilePin | None


class ProposalInput(ProposalProjection):
    task: Task
    inventory_index: int
    task_instance: TaskInstance
    canonical_signature: Sha256
    inventory_binding: FilePin
    partition: Literal["D_cal"]


class DcContract(ProposalProjection):
    family: FilePin
    renderer_id: str
    native_order: tuple[str, ...]


class ProposalResources(ProposalProjection):
    embedding: str
    dimension: int
    normalize: bool
    native_model: str
    actual_cache_binding: FilePin | None
    intended_runtime_binding: FilePin | None


class Task11Proposal(ProposalProjection):
    schema_id: Literal["phase13-task11-proposed16-v1"] = Field(alias="schema")
    ordinary_units: tuple[ProposalUnit, ...]
    checkpoint_groups: tuple[ProposalGroup, ...]
    inputs: tuple[ProposalInput, ...]
    seed: Literal[0]
    dc_contract: DcContract
    resources: ProposalResources


class GroupFacts(DiagnosticModel):
    group_id: str
    prefix_id: str = Field(min_length=1)
    source_identity: str = Field(min_length=1)
    checkpoint_id: str = Field(min_length=1)
    checkpoint_index: int = Field(ge=0)
    prefix_sample_ids: tuple[str, ...] = Field(min_length=1)
    state_binding: FilePin
    evidence: FilePin


class RenderFacts(DiagnosticModel):
    unit_id: str
    renderer_id: str = Field(min_length=1)
    render_id: str = Field(min_length=1)
    evidence: FilePin


class PreparationMetadata(DiagnosticModel):
    schema_version: Literal["phase13_diagnostic_proposal_preparation_v1"]
    proposal: FilePin
    subject: RegistrationSubject
    groups: tuple[GroupFacts, ...] = Field(min_length=4, max_length=4)
    renders: tuple[RenderFacts, ...]


class Prerequisite(DiagnosticModel):
    path: str
    code: str


class PreparedProposal(DiagnosticModel):
    schema_version: Literal["phase13_diagnostic_proposal_result_v1"] = "phase13_diagnostic_proposal_result_v1"
    status: Literal["PENDING_PREREQUISITES", "PREPARED_NOT_ADMITTED"]
    proposal: FilePin
    schedule: tuple[ProposalUnit, ...]
    prerequisites: tuple[Prerequisite, ...]
    subject: RegistrationSubject | None
    provider_calls: Literal[0] = 0
