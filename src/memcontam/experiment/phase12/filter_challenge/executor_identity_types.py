from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from memcontam.contamination.phase12.models import canonical_json_hash
from memcontam.experiment.phase12.filter_challenge.executor_identity_values import (
    canonical_identity_value,
)
from memcontam.experiment.phase12.filter_challenge.executor_types import (
    BaselineFamily,
    PairingIdentity,
)
from memcontam.tasks.base import TaskInstance
from memcontam.tasks.dispatch import canonical_task_json


@dataclass(frozen=True, slots=True)
class RuntimeIdentityProjection:
    baseline_family: BaselineFamily
    source_checkpoint_id: str
    source_checkpoint_hash: str
    rag_mode: str
    task_payload_hash: str
    prompt_payload_hash: str
    model_snapshot: str
    decoding_contract_hash: str
    tool_mode: str | None
    tool_permissions_hash: str | None
    verifier_version: str
    context_budget_capacity_hash: str
    retriever_index_capacity_hash: str | None
    resource_retry_limit_hash: str | None

    def matches(self, identity: PairingIdentity) -> bool:
        return (
            self.baseline_family == identity.baseline_family
            and self.source_checkpoint_id == identity.source_checkpoint_id
            and self.source_checkpoint_hash == identity.source_checkpoint_hash
            and self.rag_mode == identity.rag_mode
            and self.prompt_payload_hash == identity.prompt_payload_hash
            and self.model_snapshot == identity.model_snapshot
            and self.decoding_contract_hash == identity.decoding_contract_hash
            and (self.tool_mode is None or self.tool_mode == identity.tool_mode)
            and (
                self.tool_permissions_hash is None
                or self.tool_permissions_hash == identity.tool_permissions_hash
            )
            and self.verifier_version == identity.verifier_version
            and self.context_budget_capacity_hash == identity.context_budget_capacity_hash
            and (
                self.retriever_index_capacity_hash is None
                or self.retriever_index_capacity_hash == identity.retriever_index_capacity_hash
            )
            and (
                self.resource_retry_limit_hash is None
                or self.resource_retry_limit_hash == identity.resource_retry_limit_hash
            )
        )


@dataclass(frozen=True, slots=True)
class ProjectionInputs:
    baseline_family: BaselineFamily
    source_checkpoint_id: str
    source_checkpoint_hash: str
    rag_mode: str
    task: TaskInstance
    model: str
    decoding: Any
    prompt_contract: Any
    tool_mode: str | None
    tool_permissions: Any
    verifier: Any
    context_budget_capacity: Any
    retriever_index_capacity: Any
    resource_retry_limit: Any


def build_projection(inputs: ProjectionInputs) -> RuntimeIdentityProjection:
    task_payload_hash = sha256(canonical_task_json(inputs.task).encode()).hexdigest()
    return RuntimeIdentityProjection(
        baseline_family=inputs.baseline_family,
        source_checkpoint_id=inputs.source_checkpoint_id,
        source_checkpoint_hash=inputs.source_checkpoint_hash,
        rag_mode=inputs.rag_mode,
        task_payload_hash=task_payload_hash,
        prompt_payload_hash=canonical_json_hash(
            {"task_payload_hash": task_payload_hash, "contract": inputs.prompt_contract}
        ),
        model_snapshot=inputs.model,
        decoding_contract_hash=canonical_json_hash(inputs.decoding),
        tool_mode=inputs.tool_mode,
        tool_permissions_hash=(
            None
            if inputs.tool_permissions is None
            else canonical_json_hash(inputs.tool_permissions)
        ),
        verifier_version=canonical_json_hash(canonical_identity_value(inputs.verifier)),
        context_budget_capacity_hash=canonical_json_hash(inputs.context_budget_capacity),
        retriever_index_capacity_hash=(
            None
            if inputs.retriever_index_capacity is None
            else canonical_json_hash(inputs.retriever_index_capacity)
        ),
        resource_retry_limit_hash=(
            None
            if inputs.resource_retry_limit is None
            else canonical_json_hash(inputs.resource_retry_limit)
        ),
    )
