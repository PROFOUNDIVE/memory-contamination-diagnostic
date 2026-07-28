from __future__ import annotations

from dataclasses import astuple, replace

import pytest
from pydantic import TypeAdapter

from memcontam.experiment.phase12.filter_challenge.adapters.full_history import (
    FullHistoryPairRequest,
    FullHistoryProvisionalAdapter,
)
from memcontam.experiment.phase12.filter_challenge.contracts import (
    ChallengeRoutingDecision,
)
from memcontam.experiment.phase12.filter_challenge.executor import (
    ActivationContext,
    ControlResultCache,
    FullHistoryExecutionRequest,
    PairExecutorError,
    activation_decision,
    build_control_cache_key,
    build_shared_assessment_key,
    consume_routing,
    evaluability_rate,
    execute_isolated_pair,
)
from memcontam.experiment.phase12.filter_challenge.executor_source import source_snapshot
from memcontam.experiment.phase12.filter_challenge.executor_types import (
    ExecutionOrder,
    ReplicateSeedContract,
)
from memcontam.memory.checkpoint_v3 import NativeState, serialize_checkpoint
from phase12_filter_v5_executor_cases import (
    bot_case,
    full_history_case,
    rag_case,
    reflexion_case,
)


class _NativeExecutionError(RuntimeError):
    pass


@pytest.mark.parametrize(
    ("case_factory", "adapter_name"),
    [
        (full_history_case, "FullHistoryProvisionalAdapter"),
        (rag_case, "RagFrozenProvisionalAdapter"),
        (bot_case, "BoTStyleChallengeAdapter"),
        (reflexion_case, "ReflexionProvisionalAdapter"),
    ],
)
def test_executor_invokes_each_approved_native_path(case_factory, adapter_name: str) -> None:
    # Given: one typed request for an approved native baseline family.
    case = case_factory()

    # When: the executor runs the isolated native pair.
    evidence = execute_isolated_pair(case.request)

    # Then: actual call IDs and assessment evidence come only from that native path.
    assert evidence.adapter_name == adapter_name
    assert evidence.control_answer_call_id != evidence.challenge_answer_call_id
    assert case.sink.assessments == [evidence]
    assert case.sink.trials == []
    assert case.sink.calls == []


def test_executor_propagates_native_full_history_adapter_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: the approved native Full History adapter fails during execution.
    case = full_history_case()

    def fail_native_execute(
        _adapter: FullHistoryProvisionalAdapter, _request: FullHistoryPairRequest
    ) -> None:
        raise _NativeExecutionError("native execution reached")

    monkeypatch.setattr(FullHistoryProvisionalAdapter, "execute", fail_native_execute)

    # When / Then: the failure propagates instead of caller-fabricated evidence succeeding.
    with pytest.raises(_NativeExecutionError, match="native execution reached"):
        execute_isolated_pair(case.request)


def test_executor_derives_forbidden_writeback_from_native_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a native adapter outcome carrying an actual challenge write event.
    case = full_history_case()
    native_execute = FullHistoryProvisionalAdapter.execute

    def execute_with_writeback(
        adapter: FullHistoryProvisionalAdapter, request: FullHistoryPairRequest
    ):
        result = native_execute(adapter, request)
        outcome = replace(
            result.challenge_outcome,
            memory_write_event={"entry_id": "forbidden-write"},
        )
        return replace(result, challenge_outcome=outcome)

    monkeypatch.setattr(FullHistoryProvisionalAdapter, "execute", execute_with_writeback)

    # When / Then: the executor rejects writeback derived from the native result.
    with pytest.raises(PairExecutorError, match="UPDATER_NOT_DISABLED"):
        execute_isolated_pair(case.request)


@pytest.mark.parametrize("contract", ["deterministic", "seed_coupled"])
def test_exact_control_key_reuses_only_the_native_control(
    contract: ReplicateSeedContract,
) -> None:
    # Given: two independent native pairs sharing every control-affecting dimension.
    cache = ControlResultCache()
    first = full_history_case(contract, assessment_id="first", cache=cache)
    second = full_history_case(contract, assessment_id="second", cache=cache)

    # When: both assessments execute.
    first_evidence = execute_isolated_pair(first.request)
    second_evidence = execute_isolated_pair(second.request)

    # Then: control is reused, challenge remains fresh, and the literal key is complete.
    assert first_evidence.control_from_cache is False
    assert second_evidence.control_from_cache is True
    assert second.control_calls is not None and second.challenge_calls is not None
    assert second.control_calls.calls == 0
    assert second.challenge_calls.calls == 1
    identity = first.request.identity
    assert astuple(first_evidence.control_cache_key) == (
        identity.source_checkpoint_hash,
        identity.baseline_family,
        identity.rag_mode,
        identity.probe_id,
        identity.prompt_payload_hash,
        identity.replicate_seed_contract,
        identity.replicate_id,
        identity.paired_seed_replay_id,
        identity.model_snapshot,
        identity.decoding_contract_hash,
        identity.fidelity_label,
        identity.tool_mode,
        identity.tool_permissions_hash,
        identity.raw_parser_version,
        identity.canonicalizer_version,
        identity.verifier_version,
        identity.base_prompt_hash,
        identity.formatter_version,
        identity.context_budget_capacity_hash,
        identity.retriever_index_capacity_hash,
        identity.noncandidate_memory_hash,
        identity.resource_retry_limit_hash,
    )


@pytest.mark.parametrize("order", ["control_first", "challenge_first"])
def test_counterbalanced_pairs_execute_fresh_in_the_explicit_order(
    order: ExecutionOrder,
) -> None:
    # Given: a counterbalanced BoT pair with an explicit arm order and a populated cache.
    cache = ControlResultCache()
    events: list[str] = []
    first = bot_case("counterbalanced", order=order, cache=cache, event_order=events)
    second = bot_case("counterbalanced", order=order, cache=cache, event_order=events)

    # When: both pairs execute.
    execute_isolated_pair(first.request)
    execute_isolated_pair(second.request)

    # Then: no control is reused and the first native call follows the declared order twice.
    expected = "control" if order == "control_first" else "challenge"
    assert events[0] == expected
    assert events[4] == expected
    assert first.control_calls is not None and second.control_calls is not None
    assert first.control_calls.calls == second.control_calls.calls == 2


def test_executor_rejects_duplicate_sessions_equal_transcripts_and_shared_clients() -> None:
    # Given: an otherwise valid direct native request.
    case = full_history_case()

    # When / Then: each concrete isolation violation fails before native execution.
    with pytest.raises(PairExecutorError, match="DUPLICATE_SESSION_ID"):
        execute_isolated_pair(
            replace(
                case.request,
                isolation=replace(
                    case.request.isolation,
                    control_session_id=case.request.isolation.challenge_session_id,
                ),
            )
        )
    with pytest.raises(PairExecutorError, match="SHARED_TRANSCRIPT"):
        execute_isolated_pair(
            replace(
                case.request,
                isolation=replace(
                    case.request.isolation,
                    control_transcript=case.request.isolation.challenge_transcript,
                ),
            )
        )
    execution = case.request.execution
    assert isinstance(execution, FullHistoryExecutionRequest)
    shared = replace(
        execution,
        native_request=replace(
            execution.native_request,
            control_client=execution.native_request.challenge_client,
        ),
    )
    with pytest.raises(PairExecutorError, match="SHARED_CLIENT"):
        execute_isolated_pair(replace(case.request, execution=shared))


def test_noncandidate_hash_covers_canonical_memory_bytes_and_metadata() -> None:
    # Given: two source checkpoints differing only in native metadata.
    case = full_history_case()
    execution = case.request.execution
    assert isinstance(execution, FullHistoryExecutionRequest)
    state = execution.native_request.checkpoint.state
    changed_checkpoint = serialize_checkpoint(
        NativeState(state.baseline, state.entries, {**state.native_state, "metadata": "changed"})
    )
    changed_execution = replace(
        execution,
        native_request=replace(execution.native_request, checkpoint=changed_checkpoint),
    )

    # When: canonical source snapshots are hashed.
    original = source_snapshot(execution)
    changed = source_snapshot(changed_execution)

    # Then: noncandidate memory identity changes with metadata, not merely entry IDs.
    assert original.noncandidate_memory_bytes != changed.noncandidate_memory_bytes
    assert original.noncandidate_memory_hash != changed.noncandidate_memory_hash


def test_activation_uses_tau_for_root_and_distinct_evolved_checkpoint_for_later_write() -> None:
    # Given: activation checkpoint tau and a later branch-local checkpoint.
    case = full_history_case()
    candidate = case.request.candidate
    context = ActivationContext("tau", "branch-later", ("old",), "candidate", "filter")
    root = candidate.model_copy(update={"source_checkpoint_id": "tau"})
    later = candidate.model_copy(
        update={"candidate_entry_id": "later", "source_checkpoint_id": "branch-later"}
    )

    # When / Then: root and later candidates bind to their distinct checkpoints.
    assert activation_decision(context, root, later_native_write=False).status == "assess"
    assert activation_decision(context, later, later_native_write=True).status == "assess"
    with pytest.raises(PairExecutorError, match="EVOLVED_BRANCH_CHECKPOINT_REQUIRED"):
        activation_decision(
            replace(context, evolved_branch_checkpoint_id="tau"),
            later,
            later_native_write=True,
        )


def test_contam_shadow_and_filter_apply_reuse_the_exact_shared_assessment_key() -> None:
    # Given: one assessment key and a quarantine decision.
    case = full_history_case()
    key = build_shared_assessment_key(
        case.request.identity,
        case.request.candidate,
        case.request.candidate_version,
        case.request.probe_configuration_hash,
    )
    routing: ChallengeRoutingDecision = TypeAdapter(ChallengeRoutingDecision).validate_python(
        {
            "assessment_state": "contradicted",
            "route_target": "quarantine",
            "audit_flag": False,
            "routing_reason_code": "CONTRADICTED",
        }
    )

    # When: both arms consume the same assessment.
    evidence = execute_isolated_pair(case.request)
    consumption = consume_routing(evidence.assessment_id, routing, key, evidence)

    # Then: Contam shadows, Filter applies, and neither substitutes a key.
    assert consumption.contam.effect == "shadow" and consumption.contam.route_target is None
    assert consumption.filter.effect == "apply" and consumption.filter.route_target == "quarantine"
    assert consumption.contam.shared_assessment_key is key
    assert consumption.filter.shared_assessment_key is key
    assert evaluability_rate(0, 0) is None
    assert evaluability_rate(3, 4) == 0.75
    assert build_control_cache_key(case.request.identity) == key.control_cache_key
