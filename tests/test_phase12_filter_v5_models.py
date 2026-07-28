from inspect import Signature, signature

import pytest
from pydantic import TypeAdapter, ValidationError

import memcontam.experiment.phase12.filter_challenge as filter_challenge
from memcontam.experiment.phase12.filter_challenge import AnswerCallRelation, ChallengeCandidate, ChallengeRoutingDecision
from memcontam.experiment.phase12.filter_challenge.adapters.base import AnswerCallObserver, AnswerParser, AnswerVerifier, ChallengeAdapter, ChallengeCallClient, FrozenCheckpoint
from memcontam.experiment.phase12.filter_challenge.audit import ChallengeAuditLabels, PostRouteAuditJoin
from tests.phase12_filter_v5_model_cases import AUDIT_LABELS, DOMAIN_SCHEMA_VERSION, EXPECTED_PUBLIC_DOMAIN_MODEL_NAMES, FORBIDDEN_PROPERTIES, INVALID_CONTRACT_PAYLOADS, LEGAL_VARIANT_PAYLOADS, NOT_EVALUABLE_ROUTING, PUBLIC_CONTRACT_PAYLOADS, PUBLIC_SCHEMA_SUBJECTS, PUBLIC_SCHEMAS, ROUTABLE_CANDIDATE


def _object_schemas(value):
    if isinstance(value, dict):
        if value.get("type") == "object":
            yield value
        for child in value.values():
            yield from _object_schemas(child)
    elif isinstance(value, list):
        for child in value:
            yield from _object_schemas(child)


def test_public_domain_registry_is_exact() -> None:
    # Given: the package root and ordered public names.
    # When: its exports are resolved.
    # Then: all thirteen stable contracts are available.
    assert filter_challenge.PUBLIC_DOMAIN_MODEL_NAMES == EXPECTED_PUBLIC_DOMAIN_MODEL_NAMES
    assert tuple(getattr(filter_challenge, name) for name in EXPECTED_PUBLIC_DOMAIN_MODEL_NAMES) == PUBLIC_SCHEMA_SUBJECTS


@pytest.mark.parametrize("schema", PUBLIC_SCHEMAS)
def test_every_reachable_public_object_schema_is_strict_and_audit_free(schema) -> None:
    # Given: one schema reachable from each public contract or discriminated union.
    # When: every object schema is traversed.
    # Then: each object pins the domain version, forbids extras, and omits audit metadata.
    for object_schema in _object_schemas(schema):
        properties = object_schema["properties"]
        assert properties["schema_version"]["const"] == DOMAIN_SCHEMA_VERSION
        assert object_schema["additionalProperties"] is False
        assert not set(properties) & set(FORBIDDEN_PROPERTIES)


@pytest.mark.parametrize("forbidden_field", FORBIDDEN_PROPERTIES)
def test_candidate_rejects_every_forbidden_policy_visible_field(forbidden_field: str) -> None:
    # Given: a legal candidate and one audit-only property.
    payload = dict(ROUTABLE_CANDIDATE, **{forbidden_field: "audit-only"})

    # When: it crosses the policy boundary.
    # Then: strict validation rejects the extra property.
    with pytest.raises(ValidationError):
        ChallengeCandidate.model_validate(payload)


@pytest.mark.parametrize(("subject", "payload"), PUBLIC_CONTRACT_PAYLOADS + LEGAL_VARIANT_PAYLOADS)
def test_public_contracts_and_variants_accept_only_legal_payloads(subject, payload) -> None:
    # Given: every public contract and every additional discriminated variant.
    # When: each legal payload is parsed.
    # Then: the returned object has the exact domain version.
    assert TypeAdapter(subject).validate_python(payload).schema_version == DOMAIN_SCHEMA_VERSION


@pytest.mark.parametrize(("subject", "payload"), INVALID_CONTRACT_PAYLOADS)
def test_contracts_reject_illegal_variant_combinations(subject, payload) -> None:
    # Given: a discriminated or cross-field combination outside the contract.
    # When: it is parsed.
    # Then: validation rejects it.
    with pytest.raises(ValidationError):
        TypeAdapter(subject).validate_python(payload)


def test_explicit_matched_relation_rejects_unequal_call_ids() -> None:
    # Given: explicit provenance with different answer and parser source call IDs.
    # When: it is parsed as explicit_matched.
    # Then: callers must use the mismatched relation instead.
    with pytest.raises(ValidationError):
        TypeAdapter(AnswerCallRelation).validate_python(
            {
                "answer_call_provenance_status": "explicit_matched",
                "answer_call_id": "answer-call-1",
                "parsed_response_source_call_id": "parser-source-call-2",
                "parser_result_id": "parser-1",
                "verifier_result_id": "verifier-1",
            }
        )


def test_audit_join_requires_routing_and_protocols_are_audit_free() -> None:
    # Given: audit labels and a validated fail-open routing decision.
    decision = TypeAdapter(ChallengeRoutingDecision).validate_python(NOT_EVALUABLE_ROUTING)
    labels = ChallengeAuditLabels(**AUDIT_LABELS)

    # When: audit data is joined and protocol signatures are inspected.
    joined = PostRouteAuditJoin(candidate_entry_id="candidate-opaque-1", routing_decision=decision, audit_labels=labels)

    # Then: routing precedes audit access and no protocol accepts audit metadata.
    assert joined.routing_decision.route_target == "active"
    with pytest.raises(ValidationError):
        PostRouteAuditJoin.model_validate({"candidate_entry_id": "candidate-opaque-1", "audit_labels": AUDIT_LABELS})
    methods = (ChallengeAdapter.execute, ChallengeCallClient.answer, AnswerCallObserver.observe, AnswerParser.parse, AnswerVerifier.verify, FrozenCheckpoint.snapshot_id)
    for method in methods:
        parameters = set(signature(method).parameters)
        assert not any(fragment in parameter for fragment in FORBIDDEN_PROPERTIES for parameter in parameters)
        assert signature(method).return_annotation is not Signature.empty
