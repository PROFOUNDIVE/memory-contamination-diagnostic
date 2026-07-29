from __future__ import annotations

from pydantic import TypeAdapter

from memcontam.experiment.phase12.filter_challenge import PUBLIC_DOMAIN_MODEL_NAMES

from memcontam.experiment.phase12.filter_challenge.contracts import (
    AnswerCallRelation,
    CandidateAssessmentAggregate,
    CandidateExposureRecord,
    ChallengeAssessmentState,
    ChallengeCandidate,
    ChallengeRoutability,
    ChallengeRoutingDecision,
    FilterPolicyIdentity,
    OperationalProbeSuite,
    PairedExecutionIdentity,
    ProbeDisposition,
    ProbeEligibilityState,
    ProbeInventoryManifest,
)
from memcontam.experiment.phase12.filter_challenge.evidence_contract import (
    canonical_json_bytes,
    sha256_bytes,
)
from memcontam.experiment.phase12.filter_challenge.mft_state_models import JsonValue


from typing import Final
_FORBIDDEN_PROPERTIES: Final = {
    "candidate_role", "correctness_label", "irrelevance_label", "is_injected",
    "origin_class", "injection_class", "treatment_arm", "future_main_outcome",
    "suffix_outcome",
}


def public_domain_schema_schema_names() -> tuple[str, ...]:
    return PUBLIC_DOMAIN_MODEL_NAMES


def public_domain_schema_hashes() -> dict[str, JsonValue]:
    return {
        name: sha256_bytes(canonical_json_bytes(schema))
        for name, schema in _public_schemas()
    }


def policy_visible_schema_boundary_valid() -> bool:
    return all(
        not _contains_forbidden_property(schema)
        for _, schema in _public_schemas()
    )


def _public_schemas() -> tuple[tuple[str, dict[str, JsonValue]], ...]:
    schemas = {
        "FilterPolicyIdentity": FilterPolicyIdentity.model_json_schema(),
        "ChallengeCandidate": ChallengeCandidate.model_json_schema(),
        "ChallengeRoutability": TypeAdapter(ChallengeRoutability).json_schema(),
        "ProbeInventoryManifest": ProbeInventoryManifest.model_json_schema(),
        "OperationalProbeSuite": OperationalProbeSuite.model_json_schema(),
        "ProbeEligibilityState": TypeAdapter(ProbeEligibilityState).json_schema(),
        "PairedExecutionIdentity": TypeAdapter(PairedExecutionIdentity).json_schema(),
        "AnswerCallRelation": TypeAdapter(AnswerCallRelation).json_schema(),
        "CandidateExposureRecord": CandidateExposureRecord.model_json_schema(),
        "ProbeDisposition": TypeAdapter(ProbeDisposition).json_schema(),
        "ChallengeAssessmentState": TypeAdapter(ChallengeAssessmentState).json_schema(),
        "ChallengeRoutingDecision": TypeAdapter(ChallengeRoutingDecision).json_schema(),
        "CandidateAssessmentAggregate": CandidateAssessmentAggregate.model_json_schema(),
    }
    return tuple((name, schemas[name]) for name in PUBLIC_DOMAIN_MODEL_NAMES)


def _contains_forbidden_property(value: object) -> bool:
    if isinstance(value, dict):
        properties = value.get("properties")
        if isinstance(properties, dict) and any(
            name in _FORBIDDEN_PROPERTIES or (name.startswith("B") and name[1:].isdigit())
            for name in properties
        ):
            return True
        return any(_contains_forbidden_property(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_forbidden_property(item) for item in value)
    return False
