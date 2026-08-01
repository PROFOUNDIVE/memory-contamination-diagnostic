from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import pytest

from memcontam.experiment.phase12.filter_challenge.executor import (
    BoTExecutionRequest,
    ControlResultCache,
    FullHistoryExecutionRequest,
    PairExecutorError,
    PairingIdentity,
    RagFrozenExecutionRequest,
    ReflexionExecutionRequest,
    execute_isolated_pair,
)
from .phase12_filter_v5_executor_cases import (
    bot_case,
    full_history_case,
    rag_case,
    reflexion_case,
)
from .phase12_filter_v5_executor_support import pair_request, task


def test_executor_rejects_rag_challenge_task_mismatch() -> None:
    # Given: RAG challenge execution receives a different native task payload.
    case = rag_case()
    execution = case.request.execution
    assert isinstance(execution, RagFrozenExecutionRequest)
    request = replace(
        case.request,
        execution=replace(
            execution,
            native_request=replace(
                execution.native_request,
                challenge_trial=replace(
                    execution.native_request.challenge_trial,
                    task=task().model_copy(update={"sample_id": "different-sample"}),
                ),
            ),
        ),
    )

    # When / Then: runtime task identity rejects the pair before native execution.
    with pytest.raises(PairExecutorError, match="PAIRED_EXECUTION_IDENTITY_MISMATCH"):
        execute_isolated_pair(request)


def test_executor_rejects_rag_same_capacity_inclusion_mismatch() -> None:
    # Given: RAG challenge changes exact included IDs without changing capacity.
    case = rag_case()
    execution = case.request.execution
    assert isinstance(execution, RagFrozenExecutionRequest)
    request = replace(
        case.request,
        execution=replace(
            execution,
            native_request=replace(
                execution.native_request,
                challenge_trial=replace(
                    execution.native_request.challenge_trial,
                    included_document_ids=("candidate", "source-a", "source-c"),
                ),
            ),
        ),
    )

    # When / Then: exact RAG context identity rejects before native retrieval.
    with pytest.raises(PairExecutorError, match="PAIRED_EXECUTION_IDENTITY_MISMATCH"):
        execute_isolated_pair(request)


def test_executor_rejects_bot_challenge_task_mismatch() -> None:
    # Given: BoT challenge execution receives a different native task payload.
    case = bot_case()
    execution = case.request.execution
    assert isinstance(execution, BoTExecutionRequest)
    request = replace(
        case.request,
        execution=replace(
            execution,
            challenge=replace(
                execution.challenge,
                task=task().model_copy(update={"sample_id": "different-sample"}),
            ),
        ),
    )

    # When / Then: runtime task identity rejects the pair before native execution.
    with pytest.raises(PairExecutorError, match="PAIRED_EXECUTION_IDENTITY_MISMATCH"):
        execute_isolated_pair(request)


def test_executor_rejects_reflexion_challenge_retry_mismatch() -> None:
    # Given: Reflexion challenge execution changes its native retry limit.
    case = reflexion_case()
    execution = case.request.execution
    assert isinstance(execution, ReflexionExecutionRequest)
    request = replace(
        case.request,
        execution=replace(
            execution,
            challenge_trial=replace(execution.challenge_trial, config={"max_attempts": 1}),
        ),
    )

    # When / Then: runtime retry identity rejects the pair before native execution.
    with pytest.raises(PairExecutorError, match="PAIRED_EXECUTION_IDENTITY_MISMATCH"):
        execute_isolated_pair(request)


def test_full_history_context_budget_change_cannot_reuse_cached_control() -> None:
    # Given: a cached control and a second request with a changed native context budget.
    cache = ControlResultCache()
    first = full_history_case(cache=cache)
    second = full_history_case(cache=cache)
    execute_isolated_pair(first.request)
    execution = second.request.execution
    assert isinstance(execution, FullHistoryExecutionRequest)
    changed_execution = replace(
        execution,
        native_request=replace(
            execution.native_request,
            context_config={
                **execution.native_request.context_config,
                "context_window_tokens": 99_999,
            },
        ),
    )
    changed_request, _sink = pair_request(changed_execution, second.request.candidate, cache=cache)

    # When: the changed-budget pair executes.
    evidence = execute_isolated_pair(changed_request)

    # Then: its derived cache identity prevents control reuse.
    assert evidence.control_from_cache is False


@pytest.mark.parametrize(
    ("control_transcript", "challenge_transcript", "accepted"),
    [
        ((), (), True),
        (("control",), (), False),
        ((), ("challenge",), False),
        (("control",), ("challenge",), False),
    ],
)
def test_executor_requires_both_transcripts_to_be_empty(
    control_transcript: tuple[str, ...],
    challenge_transcript: tuple[str, ...],
    accepted: bool,
) -> None:
    # Given: one of the four empty/nonempty transcript combinations.
    case = full_history_case()
    request = replace(
        case.request,
        isolation=replace(
            case.request.isolation,
            control_transcript=control_transcript,
            challenge_transcript=challenge_transcript,
        ),
    )

    # When / Then: only the empty/empty pair is accepted.
    if accepted:
        assert execute_isolated_pair(request).assessment_id == request.assessment_id
    else:
        with pytest.raises(PairExecutorError, match="NONEMPTY_TRANSCRIPT"):
            execute_isolated_pair(request)


@pytest.mark.parametrize("case_factory", [full_history_case, rag_case, bot_case, reflexion_case])
def test_nonempty_transcript_rejects_every_native_family(case_factory) -> None:
    # Given: a native pair carries one preexisting conversation turn.
    case = case_factory()
    request = replace(
        case.request,
        isolation=replace(case.request.isolation, control_transcript=("turn",)),
    )

    # When / Then: the common isolation gate rejects before family dispatch.
    with pytest.raises(PairExecutorError, match="NONEMPTY_TRANSCRIPT"):
        execute_isolated_pair(request)


@pytest.mark.parametrize(
    "mutate_identity",
    [
        lambda identity: replace(identity, prompt_payload_hash="different"),
        lambda identity: replace(identity, decoding_contract_hash="different"),
        lambda identity: replace(identity, tool_mode="different"),
        lambda identity: replace(identity, tool_permissions_hash="different"),
        lambda identity: replace(identity, verifier_version="different"),
        lambda identity: replace(identity, context_budget_capacity_hash="different"),
        lambda identity: replace(identity, retriever_index_capacity_hash="different"),
        lambda identity: replace(identity, resource_retry_limit_hash="different"),
    ],
)
def test_executor_rejects_each_derivable_requested_identity_mismatch(
    mutate_identity: Callable[[PairingIdentity], PairingIdentity],
) -> None:
    # Given: a BoT request whose caller assertion disagrees with one derived field.
    case = bot_case()
    request = replace(
        case.request,
        identity=mutate_identity(case.request.identity),
    )

    # When / Then: every derivable field is checked before native dispatch.
    with pytest.raises(PairExecutorError, match="PAIRED_EXECUTION_IDENTITY_MISMATCH"):
        execute_isolated_pair(request)
