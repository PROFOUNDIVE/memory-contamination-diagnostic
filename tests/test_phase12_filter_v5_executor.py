from __future__ import annotations

from dataclasses import fields, replace
from hashlib import sha256

import pytest
from pydantic import TypeAdapter

from memcontam.experiment.phase12.filter_challenge.contracts import (
    AnswerCallRelation,
    CandidateExposureRecord,
    ChallengeCandidate,
    ChallengeRoutingDecision,
)
from memcontam.experiment.phase12.filter_challenge.executor import (
    ActivationContext,
    ControlResultCache,
    IsolatedPairArm,
    IsolatedPairRequest,
    PairArmResult,
    PairExecutorError,
    PairingIdentity,
    ScriptedSession,
    SourceSnapshot,
    activation_decision,
    build_control_cache_key,
    build_shared_assessment_key,
    consume_routing,
    evaluability_rate,
    execute_isolated_pair,
)


class _Client:
    pass


def _candidate(
    baseline: str = "full_history", native_kind: str = "full_history_transcript", *, unsupported: bool = False
) -> ChallengeCandidate:
    routability = (
        {"routability": "unsupported", "reason_code": "PROBE_MAPPING_UNSUPPORTED"}
        if unsupported
        else {"routability": "challenge_routable_v1", "challenge_suite_key": "suite-1"}
    )
    return ChallengeCandidate.model_validate(
        {
            "candidate_entry_id": "candidate",
            "candidate_native_content": "candidate content",
            "candidate_native_kind": native_kind,
            "baseline_family": baseline,
            "rag_mode": "frozen" if baseline == "rag_frozen" else "not_applicable",
            "source_checkpoint_id": "checkpoint-1",
            "source_active_state_hash": sha256(b"source").hexdigest(),
            "routability": routability,
        }
    )


def _source() -> SourceSnapshot:
    canonical_bytes = b"source"
    noncandidate_entry_ids = ("source-1",)
    return SourceSnapshot(
        checkpoint_id="checkpoint-1",
        canonical_bytes=canonical_bytes,
        canonical_sha256=sha256(canonical_bytes).hexdigest(),
        noncandidate_entry_ids=noncandidate_entry_ids,
        noncandidate_memory_hash=sha256("\0".join(noncandidate_entry_ids).encode()).hexdigest(),
    )


def _identity() -> PairingIdentity:
    return PairingIdentity(
        source_checkpoint_id="checkpoint-1",
        source_checkpoint_hash=sha256(b"source").hexdigest(),
        baseline_family="full_history",
        rag_mode="not_applicable",
        candidate_native_kind="full_history_transcript",
        probe_id="probe-1",
        prompt_payload_hash="prompt",
        replicate_seed_contract="deterministic",
        model_snapshot="model-1",
        decoding_contract_hash="decoding",
        fidelity_label="fidelity-1",
        tool_mode="text_only",
        tool_permissions_hash="permissions",
        raw_parser_version="parser-1",
        canonicalizer_version="canonicalizer-1",
        verifier_version="verifier-1",
        base_prompt_hash="base-prompt",
        formatter_version="formatter-1",
        context_budget_capacity_hash="capacity",
        retriever_index_capacity_hash="retriever",
        noncandidate_memory_hash=sha256(b"source-1").hexdigest(),
        resource_retry_limit_hash="limits",
    )


def _relation(call_id: str) -> AnswerCallRelation:
    return TypeAdapter(AnswerCallRelation).validate_python(
        {
            "answer_call_provenance_status": "explicit_matched",
            "answer_call_id": call_id,
            "parsed_response_source_call_id": call_id,
            "parser_result_id": f"{call_id}:parser",
            "verifier_result_id": f"{call_id}:verifier",
        }
    )


def _result(candidate: ChallengeCandidate, call_id: str, *, exposed: bool) -> PairArmResult:
    return PairArmResult(
        answer_relation=_relation(call_id),
        candidate_exposure=CandidateExposureRecord(
            candidate_entry_id=candidate.candidate_entry_id,
            candidate_final_context_inclusion=exposed,
            candidate_final_context_source_ids=(candidate.candidate_entry_id,) if exposed else (),
        ),
        updater_enabled=False,
        memory_write_event_id=None,
        displaced_noncandidate_entry_ids=(),
    )


def _arm(
    label: str,
    candidate: ChallengeCandidate,
    identity: PairingIdentity,
    calls: list[str],
    *,
    exposed: bool,
) -> IsolatedPairArm:
    def run(adapter: object) -> PairArmResult:
        assert adapter.__class__.__name__ == "FullHistoryProvisionalAdapter"
        calls.append(label)
        return _result(candidate, f"{label}-answer", exposed=exposed)

    return IsolatedPairArm(
        identity=identity,
        client=_Client(),
        session=ScriptedSession(f"{label}-session"),
        transcript=[],
        run=run,
    )


def _request(
    candidate: ChallengeCandidate,
    control: IsolatedPairArm,
    challenge: IsolatedPairArm,
    source_supplier,
    cache: ControlResultCache | None = None,
) -> IsolatedPairRequest:
    return IsolatedPairRequest(
        assessment_id="assessment-1",
        candidate=candidate,
        candidate_version="native-v1",
        source=_source(),
        source_snapshot=source_supplier,
        control=control,
        challenge=challenge,
        probe_configuration_hash="probe-config",
        ordinary_trial_ids=("ordinary-trial",),
        control_cache=cache,
    )


def test_executor_dispatches_isolated_pair_and_reuses_only_a_deterministic_control() -> None:
    # Given: a matched Full History pair with isolated scripted arms and a deterministic cache contract.
    candidate = _candidate()
    identity = _identity()
    calls: list[str] = []
    cache = ControlResultCache()
    request = _request(
        candidate,
        _arm("control", candidate, identity, calls, exposed=False),
        _arm("challenge", candidate, identity, calls, exposed=True),
        _source,
        cache,
    )

    # When: the executor invokes the native adapter through both isolated arms twice.
    first = execute_isolated_pair(request)
    second = execute_isolated_pair(
        replace(
            request,
            assessment_id="assessment-2",
            control=_arm("control-2", candidate, identity, calls, exposed=False),
            challenge=_arm("challenge-2", candidate, identity, calls, exposed=True),
        )
    )

    # Then: only the exact deterministic control is cached and evidence stays outside ordinary trials.
    assert calls == ["control", "challenge", "challenge-2"]
    assert first.control_from_cache is False
    assert second.control_from_cache is True
    assert first.paired_execution_identity.paired_execution_identity_status == "matched"
    assert first.assessment_id not in request.ordinary_trial_ids
    assert first.control_answer_call_id == "control-answer"
    assert first.challenge_answer_call_id == "challenge-answer"
    assert '"assessment_id":"assessment-1"' in first.to_json()


def test_executor_rejects_identity_isolation_source_and_updater_failures() -> None:
    # Given: a matched pair whose individual contracts can be corrupted one dimension at a time.
    candidate = _candidate()
    identity = _identity()
    calls: list[str] = []
    control = _arm("control", candidate, identity, calls, exposed=False)
    challenge = _arm("challenge", candidate, identity, calls, exposed=True)

    # When / Then: every paired-identity field, shared session, source drift, and updater write fails closed.
    for field in fields(PairingIdentity):
        changed = replace(challenge, identity=replace(identity, **{field.name: "different"}))
        with pytest.raises(PairExecutorError, match="PAIRED_EXECUTION_IDENTITY_MISMATCH"):
            execute_isolated_pair(_request(candidate, control, changed, _source))

    with pytest.raises(PairExecutorError, match="SHARED_CLIENT"):
        execute_isolated_pair(_request(candidate, replace(control, client=challenge.client), challenge, _source))
    with pytest.raises(PairExecutorError, match="SHARED_SESSION"):
        execute_isolated_pair(_request(candidate, replace(control, session=challenge.session), challenge, _source))
    with pytest.raises(PairExecutorError, match="SHARED_TRANSCRIPT"):
        execute_isolated_pair(_request(candidate, replace(control, transcript=challenge.transcript), challenge, _source))

    drifted = SourceSnapshot(
        checkpoint_id="checkpoint-1",
        canonical_bytes=b"drifted",
        canonical_sha256=sha256(b"drifted").hexdigest(),
        noncandidate_entry_ids=("source-1",),
        noncandidate_memory_hash=sha256(b"source-1").hexdigest(),
    )
    snapshots = iter((_source(), drifted))
    with pytest.raises(PairExecutorError, match="SOURCE_DRIFT"):
        execute_isolated_pair(_request(candidate, control, challenge, lambda: next(snapshots)))

    updater = replace(_result(candidate, "control-write", exposed=False), updater_enabled=True)
    with pytest.raises(PairExecutorError, match="UPDATER_NOT_DISABLED"):
        execute_isolated_pair(
            _request(
                candidate,
                replace(control, run=lambda _adapter: updater),
                challenge,
                _source,
            )
        )


def test_control_and_shared_keys_change_for_every_identity_and_candidate_input() -> None:
    # Given: a complete paired identity and a native candidate.
    candidate = _candidate()
    identity = _identity()
    control_key = build_control_cache_key(identity)
    shared_key = build_shared_assessment_key(identity, candidate, "native-v1", "probe-config")

    # When / Then: each exact cache dimension and shared-assessment input changes the key.
    for field in fields(control_key):
        changed_identity = replace(identity, **{field.name: "different"})
        assert build_control_cache_key(changed_identity) != control_key
    for field in fields(PairingIdentity):
        changed_identity = replace(identity, **{field.name: "different"})
        assert build_shared_assessment_key(changed_identity, candidate, "native-v1", "probe-config") != shared_key
    assert build_shared_assessment_key(identity, candidate, "native-v2", "probe-config") != shared_key
    assert (
        build_shared_assessment_key(
            identity, candidate.model_copy(update={"candidate_native_content": "other"}), "native-v1", "probe-config"
        )
        != shared_key
    )
    assert build_shared_assessment_key(identity, candidate, "native-v1", "other-config") != shared_key


def test_activation_domain_routes_only_filter_and_handles_empty_denominators() -> None:
    # Given: a Filter-Challenge activation point with one grandfathered source entry and a root candidate.
    candidate = _candidate()
    context = ActivationContext("checkpoint-1", ("source-1",), "candidate", "filter")
    old_candidate = candidate.model_copy(update={"candidate_entry_id": "source-1"})
    unsupported = _candidate(unsupported=True)

    # When: root, later, unsupported, and cross-arm entries are classified for assessment.
    root = activation_decision(context, candidate, later_native_write=False)
    later = activation_decision(context, candidate.model_copy(update={"candidate_entry_id": "later"}), later_native_write=True)
    shadow = activation_decision(
        replace(context, arm="contam"), candidate.model_copy(update={"candidate_entry_id": "later"}), later_native_write=True
    )

    # Then: old entries are grandfathered, unsupported mappings are not evaluable, and only Filter consumes routing.
    assert activation_decision(context, old_candidate, later_native_write=False).status == "grandfathered"
    assert root.status == later.status == "assess"
    assert shadow.status == "not_assessed"
    assert activation_decision(context, unsupported, later_native_write=False).status == "not_evaluable"
    with pytest.raises(PairExecutorError, match="RAG_FROZEN_LATER_WRITE"):
        activation_decision(
            context,
            _candidate("rag_frozen", "rag_document").model_copy(update={"candidate_entry_id": "later"}),
            later_native_write=True,
        )

    routing = TypeAdapter(ChallengeRoutingDecision).validate_python(
        {
            "assessment_state": "contradicted",
            "route_target": "quarantine",
            "audit_flag": False,
            "routing_reason_code": "CONTRADICTED",
        }
    )
    assert consume_routing("contam", routing).route_target is None
    assert consume_routing("filter", routing).route_target == "quarantine"
    assert evaluability_rate(0, 0) is None
    assert evaluability_rate(3, 4) == 0.75
