from __future__ import annotations

import json
from pathlib import Path

from memcontam.evaluation.phase13_observability_registration import registered_verifier_result
from memcontam.experiment.phase12.game24_runner import RuntimeIdentities
from memcontam.experiment.phase12.live_branch import LiveArmBranch
from memcontam.experiment.phase12.runtime_registry import PHASE13_CORE_BASELINE_REGISTRY
from memcontam.experiment.phase13_ordinary_runtime import (
    ProspectiveOrdinaryContext,
    ProspectiveOrdinaryRun,
)
from memcontam.memory.checkpoint_v3 import NativeEntry, NativeState, serialize_checkpoint
from memcontam.memory.embeddings import BgeM3EmbeddingProvider
from memcontam.tasks.base import TaskInstance

from .phase13_diagnostic_admission import AdmittedDiagnosticExecution
from .phase13_diagnostic_cli_models import NativeInput


def load_embedding(cache: str | None) -> BgeM3EmbeddingProvider:
    return BgeM3EmbeddingProvider(cache_folder=None if cache is None else Path(cache), local_files_only=True)


def verify_answer(answer: str, task: TaskInstance) -> bool:
    return registered_verifier_result(task, answer).is_correct


def build_native(admitted: AdmittedDiagnosticExecution, embedding: BgeM3EmbeddingProvider) -> ProspectiveOrdinaryRun:
    scope = admitted.resources.scope
    native = NativeInput.model_validate_json(scope.input_utf8)
    branch = native.branch
    state = NativeState.from_mapping(json.loads(scope.state_utf8))
    checkpoint = serialize_checkpoint(state, checkpoint_index=branch.checkpoint_index)
    trial_prefix = scope.run_id if native.arm == "clean" else f"{scope.run_id}:{native.arm}"
    expected = next((entry for entry in state.entries if isinstance(entry, NativeEntry)
                     and entry.entry_id == branch.injected_root_id), None)
    context = ProspectiveOrdinaryContext(task=native.tasks[0], client=admitted.execution_client,
        model=native.model, verifier=verify_answer, decoding=native.decoding,
        identities=RuntimeIdentities(scope.run_id, f"{trial_prefix}:trial:1:{native.tasks[0].sample_id}", 1, native.baseline),
        branch=native.arm, embedding_provider=embedding, baseline_configs=native.baseline_configs,
        diagnostic_prefix_history=admitted.prefix_history, expected_intervention=expected)
    restored = PHASE13_CORE_BASELINE_REGISTRY[native.baseline].restore_state(state, context)
    return ProspectiveOrdinaryRun(task_name=native.task_name, baseline=native.baseline,
        run_id=scope.run_id, model=native.model, client=admitted.execution_client,
        verifier=verify_answer, decoding=native.decoding, arm=native.arm,
        branch=LiveArmBranch(arm=branch.arm, prefix_identity=branch.prefix_identity,
            source_identity=branch.source_identity, checkpoint=checkpoint, state=restored,
            root_count=branch.root_count, injected_root_id=branch.injected_root_id,
            candidate_triplet_id=branch.candidate_triplet_id, native_render_id=branch.native_render_id),
        tasks=native.tasks, trajectory_seed=native.trajectory_seed, embedding_provider=embedding,
        baseline_configs=native.baseline_configs, diagnostic_context=admitted)
