from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from memcontam.clients.base import LLMClient
from memcontam.contamination.phase12.models import CandidateRegistry
from memcontam.contamination.phase12.registry import (
    _parse_triplet,
    _reject_selection_markers,
    _validate_registry,
)
from memcontam.evaluation.phase13_observability_registration import ObservabilityRegistrationPacket
from memcontam.tasks.base import TaskInstance
from memcontam.tasks.game24 import build_instance as game24
from memcontam.tasks.math_equation_balancer import build_instance as equation
from memcontam.tasks.word_sorting import build_instance as words

from .phase13_legacy_rag_models import CorpusBundle, IndexBundle
from .phase13_main_checkpoint import CommonCheckpointRegistry
from .phase13_main_request_client import MainRequestClientV3
from .phase13_new_mcq_rag_models import AuthoritySelection, InterventionRegistry
from .phase13_v3_entrypoint import EntrypointError, SelectedExecutionV3
from .phase13_validated_ordinary_resources import ValidatedOrdinaryResources


@dataclass(frozen=True, slots=True)
class PreloadedMainResources:
    selected: SelectedExecutionV3

    def __post_init__(self) -> None:
        registry = self.checkpoint_registry
        _ = self.packet, self.candidate_registry, self.new_mcq_registry
        for task in registry.tasks:
            rows = self.tasks(task)
            by_id = {row.sample_id: row for row in rows}
            for seed in registry.tasks[task].seeds:
                order = sorted(by_id, key=lambda sample: hashlib.sha256(
                    f"sha256_task_seed_v1\0{task}\0{seed.seed}\0{sample}".encode()).digest())
                if (tuple(order[:seed.tau_star - 1]) != seed.clean_prefix_sample_ids
                    or tuple(order[seed.tau_star - 1:seed.tau_star - 1 + 50]) != seed.suffix_sample_ids):
                    raise EntrypointError("MAIN_AUTHORIZATION_BINDING_MISMATCH")

    @property
    def checkpoint_registry(self) -> CommonCheckpointRegistry:
        return CommonCheckpointRegistry.model_validate_json(self.selected.resource("common_checkpoint_registry"))

    @property
    def packet(self) -> ObservabilityRegistrationPacket:
        return ObservabilityRegistrationPacket.model_validate_json(self.selected.resource("observability_packet"))

    @property
    def candidate_registry(self) -> CandidateRegistry:
        payload = json.loads(self.selected.resource("candidate_registry"))
        _reject_selection_markers(payload)
        registry = CandidateRegistry(
            registry_id=payload["registry_id"], schema_version=payload["schema_version"],
            frozen_at=payload["frozen_at"], audit_registry_id=payload["audit_registry_id"],
            audit_registry_hash=payload["audit_registry_hash"],
            triplets=tuple(_parse_triplet(item) for item in payload["triplets"]),
        )
        _validate_registry(registry)
        return registry

    @property
    def new_mcq_registry(self) -> InterventionRegistry:
        raw = self.selected.resource("main_new_mcq_authority_selection")
        authority = AuthoritySelection.model_validate_json(raw)
        registry = InterventionRegistry.model_validate_json(self.selected.resource("main_new_mcq_intervention_registry"))
        if (self.selected.resource_binding("main_new_mcq_authority_selection").sha256 != registry.authority_selection_sha256
            or authority.task_selections != {task: row.selected_candidate_id for task, row in registry.tasks.items()}):
            raise EntrypointError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
        return registry

    def tasks(self, task: str) -> tuple[TaskInstance, ...]:
        lines = self.selected.resource("task_" + task).splitlines()
        builders = {"game24": game24, "math_equation_balancer": equation, "word_sorting": words}
        if task in builders:
            return tuple(builders[task](json.loads(line)) for line in lines)
        return tuple(TaskInstance.model_validate_json(line) for line in lines)

    def ordinary(self, client: LLMClient) -> ValidatedOrdinaryResources:
        raw = self.selected.resource("common_checkpoint_registry")
        return ValidatedOrdinaryResources(client, 8192, raw, self.selected.resource_binding("common_checkpoint_registry").sha256, self.selected.costs,
            client if isinstance(client, MainRequestClientV3) else None)

    def legacy_bundles(self, task: str) -> tuple[CorpusBundle, IndexBundle]:
        return (
            CorpusBundle.model_validate_json(self.selected.resource("rag_corpus_" + task)),
            IndexBundle.model_validate_json(self.selected.resource("rag_index_" + task)),
        )
