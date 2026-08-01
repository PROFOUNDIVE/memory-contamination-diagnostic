from __future__ import annotations

from dataclasses import replace

import pytest
from pydantic import TypeAdapter

from memcontam.experiment.phase12.filter_challenge.contracts import ChallengeRoutingDecision
from memcontam.experiment.phase12.filter_challenge.executor import (
    BoTExecutionRequest,
    ControlResultCache,
    PairExecutorError,
    RagFrozenExecutionRequest,
    ReflexionExecutionRequest,
    consume_routing,
    execute_isolated_pair,
)
from .phase12_filter_v5_executor_cases import (
    bot_case,
    full_history_case,
    rag_case,
    reflexion_case,
)


@pytest.mark.parametrize("case_factory", [full_history_case, rag_case, bot_case, reflexion_case])
def test_executor_rejects_caller_identity_that_disagrees_with_native_model(
    case_factory,
) -> None:
    # Given: the caller claims a model snapshot different from both native arms.
    case = case_factory()
    request = replace(
        case.request,
        identity=replace(case.request.identity, model_snapshot="different-model"),
    )

    # When / Then: native runtime identity prevents a fabricated matched status.
    with pytest.raises(PairExecutorError, match="PAIRED_EXECUTION_IDENTITY_MISMATCH"):
        execute_isolated_pair(request)


def test_executor_rejects_rag_challenge_model_mismatch() -> None:
    # Given: RAG control and challenge arms resolve different native models.
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
                    execution.native_request.challenge_trial, model="different-model"
                ),
            ),
        ),
    )

    # When / Then: the pair cannot emit matched identity evidence.
    with pytest.raises(PairExecutorError, match="PAIRED_EXECUTION_IDENTITY_MISMATCH"):
        execute_isolated_pair(request)


def test_executor_rejects_bot_challenge_model_mismatch() -> None:
    # Given: BoT control and challenge arms resolve different native models.
    case = bot_case()
    execution = case.request.execution
    assert isinstance(execution, BoTExecutionRequest)
    request = replace(
        case.request,
        execution=replace(execution, challenge=replace(execution.challenge, model="different-model")),
    )

    # When / Then: the pair cannot emit matched identity evidence.
    with pytest.raises(PairExecutorError, match="PAIRED_EXECUTION_IDENTITY_MISMATCH"):
        execute_isolated_pair(request)


def test_executor_rejects_reflexion_challenge_model_mismatch() -> None:
    # Given: Reflexion control and challenge arms resolve different native models.
    case = reflexion_case()
    execution = case.request.execution
    assert isinstance(execution, ReflexionExecutionRequest)
    request = replace(
        case.request,
        execution=replace(
            execution,
            challenge_trial=replace(execution.challenge_trial, model="different-model"),
        ),
    )

    # When / Then: the pair cannot emit matched identity evidence.
    with pytest.raises(PairExecutorError, match="PAIRED_EXECUTION_IDENTITY_MISMATCH"):
        execute_isolated_pair(request)


def test_seed_coupled_control_cache_isolated_by_paired_seed_replay_id() -> None:
    # Given: two seed-coupled replicates with different paired seed/replay identities.
    cache = ControlResultCache()
    first = full_history_case("seed_coupled", cache=cache)
    second = full_history_case("seed_coupled", cache=cache)
    first_request = replace(
        first.request,
        identity=replace(
            first.request.identity,
            replicate_id=0,
            paired_seed_replay_id="seed-replay-0",
        ),
    )
    second_request = replace(
        second.request,
        identity=replace(
            second.request.identity,
            replicate_id=1,
            paired_seed_replay_id="seed-replay-1",
        ),
    )

    # When: both native pairs execute against one cache.
    execute_isolated_pair(first_request)
    evidence = execute_isolated_pair(second_request)

    # Then: the second replicate cannot reuse the first seed's control.
    assert evidence.control_from_cache is False
    assert second.control_calls is not None and second.control_calls.calls == 1


def test_counterbalanced_order_is_fixed_by_replicate_schedule() -> None:
    # Given: replicate one repeats replicate zero's control-first order.
    first = bot_case("counterbalanced", order="control_first")
    second = bot_case("counterbalanced", order="control_first")
    first_request = replace(
        first.request,
        identity=replace(
            first.request.identity,
            replicate_id=0,
            paired_seed_replay_id="seed-replay-0",
        ),
    )
    second_request = replace(
        second.request,
        identity=replace(
            second.request.identity,
            replicate_id=1,
            paired_seed_replay_id="seed-replay-1",
        ),
    )

    # When: the first scheduled order executes and the second order is repeated.
    execute_isolated_pair(first_request)

    # Then: the frozen alternating schedule rejects the repeated order.
    with pytest.raises(PairExecutorError, match="COUNTERBALANCED_ORDER_REQUIRED"):
        execute_isolated_pair(second_request)


def test_transcript_free_sessions_are_isolated_by_session_and_client() -> None:
    # Given: independent sessions and clients that do not expose transcripts.
    case = full_history_case()
    request = replace(
        case.request,
        isolation=replace(
            case.request.isolation,
            control_transcript=(),
            challenge_transcript=(),
        ),
    )

    # When: the pair executes without transcript evidence.
    evidence = execute_isolated_pair(request)

    # Then: independent session and client identities are sufficient isolation evidence.
    assert evidence.assessment_id == request.assessment_id


def test_routing_jointly_binds_assessment_and_shared_key() -> None:
    # Given: one executed assessment and one validated routing decision.
    case = full_history_case()
    evidence = execute_isolated_pair(case.request)
    routing: ChallengeRoutingDecision = TypeAdapter(ChallengeRoutingDecision).validate_python(
        {
            "assessment_state": "contradicted",
            "route_target": "quarantine",
            "audit_flag": False,
            "routing_reason_code": "CONTRADICTED",
        }
    )

    # When: routing is consumed once for both experiment arms.
    consumption = consume_routing(
        evidence.assessment_id,
        routing,
        evidence.shared_assessment_key,
        evidence,
    )

    # Then: Contam shadows and Filter applies the exact same assessment key.
    assert consumption.contam.effect == "shadow"
    assert consumption.filter.effect == "apply"
    assert consumption.contam.shared_assessment_key is evidence.shared_assessment_key
    assert consumption.filter.shared_assessment_key is evidence.shared_assessment_key

    with pytest.raises(PairExecutorError, match="ASSESSMENT_IDENTITY_MISMATCH"):
        consume_routing(
            "different-assessment",
            routing,
            evidence.shared_assessment_key,
            evidence,
        )
    with pytest.raises(PairExecutorError, match="ASSESSMENT_IDENTITY_MISMATCH"):
        consume_routing(
            evidence.assessment_id,
            routing,
            replace(evidence.shared_assessment_key, candidate_version="different-version"),
            evidence,
        )
