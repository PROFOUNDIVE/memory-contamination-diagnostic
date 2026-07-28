from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, assert_never

from memcontam.experiment.phase12.filter_challenge.executor_identity_types import (
    ProjectionInputs,
    RuntimeIdentityProjection,
    build_projection,
)
from memcontam.experiment.phase12.filter_challenge.executor_types import (
    BoTExecutionRequest,
    FullHistoryExecutionRequest,
    NativeExecutionRequest,
    RagFrozenExecutionRequest,
    ReflexionExecutionRequest,
    SourceSnapshot,
)
from memcontam.experiment.phase12.filter_challenge.executor_identity_values import (
    canonical_identity_value,
    held_fixed_config,
    selected_config,
    service_contract_identity,
)
from memcontam.experiment.phase12.filter_challenge.executor_source import source_snapshot
from memcontam.memory.checkpoint_v3 import Phase12Checkpoint, deserialize_checkpoint

REGISTRY_ONLY_IDENTITY_FIELDS: Final = (
    "probe_id",
    "fidelity_label",
    "raw_parser_version",
    "canonicalizer_version",
    "base_prompt_hash",
    "formatter_version",
)
FAMILY_REGISTRY_ONLY_IDENTITY_FIELDS: Final = {
    "full_history": (
        "tool_mode",
        "tool_permissions_hash",
        "retriever_index_capacity_hash",
        "resource_retry_limit_hash",
    ),
    "rag_frozen": ("tool_mode", "tool_permissions_hash", "resource_retry_limit_hash"),
    "bot_style": (),
    "reflexion_style": ("retriever_index_capacity_hash",),
}
_INSTRUMENTATION_KEYS: Final = frozenset(
    {
        "_logging_answer_call_provenance_observer",
        "_logging_event_callback",
        "_logging_recorded_response_sink",
        "_logging_trial_context",
        "_phase12_reflection_hook",
        "arm",
        "run_id",
        "update_enabled",
    }
)


@dataclass(frozen=True, slots=True)
class _RagPairIdentity:
    source: SourceSnapshot
    retriever: Any
    context_contract: dict[str, Any]


def runtime_identity_projections(
    execution: NativeExecutionRequest,
) -> tuple[RuntimeIdentityProjection, RuntimeIdentityProjection]:
    match execution:
        case FullHistoryExecutionRequest(native_request=request):
            config = held_fixed_config(request.context_config, _INSTRUMENTATION_KEYS)
            projection = build_projection(
                ProjectionInputs(
                    "full_history",
                    request.checkpoint.identity.checkpoint_id,
                    request.checkpoint.canonical_sha256,
                    "not_applicable",
                    request.task,
                    request.model,
                    config,
                    config,
                    None,
                    None,
                    request.verifier,
                    config,
                    None,
                    None,
                )
            )
            return projection, projection
        case RagFrozenExecutionRequest(native_request=request):
            shared = _RagPairIdentity(
                source_snapshot(execution),
                _rag_retriever_identity(request.source_state.index),
                {
                    "control_included_document_ids": request.control_trial.included_document_ids,
                    "challenge_included_document_ids": request.challenge_trial.included_document_ids,
                    "control_claimed_exposure_document_ids": (
                        request.control_trial.claimed_exposure_document_ids
                    ),
                    "challenge_claimed_exposure_document_ids": (
                        request.challenge_trial.claimed_exposure_document_ids
                    ),
                },
            )
            return (
                _rag_projection(request.control_trial, shared),
                _rag_projection(request.challenge_trial, shared),
            )
        case BoTExecutionRequest(control=control, challenge=challenge):
            return _bot_projection(control), _bot_projection(challenge)
        case ReflexionExecutionRequest(
            source_checkpoint=checkpoint,
            control_trial=control,
            challenge_trial=challenge,
        ):
            context = _checkpoint_context(checkpoint, reflection_window=3)
            return _reflexion_projection(control, context), _reflexion_projection(
                challenge, context
            )
        case unreachable:
            assert_never(unreachable)


def _rag_projection(trial: Any, shared: _RagPairIdentity) -> RuntimeIdentityProjection:
    context_capacity = 3 if trial.included_document_ids is None else len(
        trial.included_document_ids
    )
    contract = {
        "branch": trial.branch,
        "condition_id": trial.condition_id,
        "context_contract": shared.context_contract,
        "rag_mode": trial.rag_mode,
    }
    return build_projection(
        ProjectionInputs(
            "rag_frozen",
            shared.source.checkpoint_id,
            shared.source.canonical_sha256,
            trial.rag_mode,
            trial.task,
            trial.model,
            contract,
            contract,
            None,
            None,
            trial.verifier,
            {"context_capacity": context_capacity, "contract": shared.context_contract},
            shared.retriever,
            None,
        )
    )


def _bot_projection(execution: Any) -> RuntimeIdentityProjection:
    config = held_fixed_config(execution.config, _INSTRUMENTATION_KEYS)
    tool_mode = str(execution.config.get("tool_mode", "text_only"))
    tool_permissions = selected_config(
        execution.config,
        ("max_tool_rounds", "tool_executor", "tool_runtime_contract", "tools"),
    )
    context = _checkpoint_context(execution.checkpoint)
    retriever = {
        "embedding_provider": service_contract_identity(
            execution.config.get("embedding_provider")
        ),
        "retrieval_k": 1,
    }
    identity = {
        "backbone": execution.identity.backbone,
        "baseline": execution.identity.baseline,
        "task_name": execution.identity.task_name,
    }
    return build_projection(
        ProjectionInputs(
            "bot_style",
            execution.checkpoint.identity.checkpoint_id,
            execution.checkpoint.canonical_sha256,
            "not_applicable",
            execution.task,
            execution.model,
            {"config": config, "identity": identity},
            {"config": config, "identity": identity},
            tool_mode,
            tool_permissions,
            execution.verifier,
            context,
            retriever,
            {"max_tool_rounds": execution.config.get("max_tool_rounds", 3)},
        )
    )


def _reflexion_projection(trial: Any, context: Any) -> RuntimeIdentityProjection:
    config = held_fixed_config(trial.config, _INSTRUMENTATION_KEYS)
    contract = {
        "branch": trial.branch,
        "condition_id": trial.condition_id,
        "config": config,
        "order_key": trial.order_key,
    }
    return build_projection(
        ProjectionInputs(
            "reflexion_style",
            context["source_checkpoint_id"],
            context["source_checkpoint_hash"],
            "not_applicable",
            trial.task,
            trial.model,
            contract,
            contract,
            trial.tool_mode,
            selected_config(trial.config, ("tools",)),
            trial.verifier,
            context,
            None,
            {"max_attempts": trial.config.get("max_attempts", 2)},
        )
    )


def _checkpoint_context(
    checkpoint: Phase12Checkpoint, *, reflection_window: int | None = None
) -> dict[str, Any]:
    state = deserialize_checkpoint(checkpoint)
    return {
        "active_capacity": state.native_state.get("active_capacity"),
        "reflection_window": reflection_window,
        "source_checkpoint_id": checkpoint.identity.checkpoint_id,
        "source_checkpoint_hash": checkpoint.canonical_sha256,
    }


def _rag_retriever_identity(index: Any) -> Any:
    if index is None:
        return None
    return {
        "document_capacity": len(index.documents),
        "embedding_contract": canonical_identity_value(index.embedding_contract),
        "index_version": index.index_version,
        "retrieval_k": 3,
        "serialization_id": index.serialization_id,
    }
