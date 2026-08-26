from __future__ import annotations

from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from memcontam.evaluation.phase13_observability_models import (
    MetricValue,
    Phase13ObservabilityError,
)


REGISTERED_FAILURE_CLASSES: Final = {
    "game24": "G24_CANONICAL_FALSE_RULE_APPLICABLE_INSTANCE_SUBSTANTIVE_FAILURE_V1",
    "math_equation_balancer": "MEB_CANONICAL_FALSE_RULE_APPLICABLE_INSTANCE_SUBSTANTIVE_FAILURE_V1",
    "word_sorting": "WS_CANONICAL_FALSE_RULE_APPLICABLE_INSTANCE_SUBSTANTIVE_FAILURE_V1",
    "mmlu_pro_engineering": "MMLUENG_SURFACE_CUE_HEURISTIC_APPLICABLE_INSTANCE_SUBSTANTIVE_FAILURE_V1",
    "mmlu_pro_physics": "MMLUPHY_SURFACE_CUE_HEURISTIC_APPLICABLE_INSTANCE_SUBSTANTIVE_FAILURE_V1",
}
AUTHORITY_HASHES: Final = {
    "experiment_design_revised_v9": "373e97317ad22b925a878a1c0972bc1220e44d21c3c83d251efcc6fa03ff46be",
    "protocol_revised_v8": "022879f559b145e30e645b6ccbd139e9927899d370f1956d27a0562580acf85f",
}
VERIFIER_PATHS: Final = {
    "game24": "src/memcontam/verifiers/game24.py",
    "math_equation_balancer": "src/memcontam/verifiers/math_equation_balancer.py",
    "word_sorting": "src/memcontam/verifiers/word_sorting.py",
    "mmlu_pro_engineering": "src/memcontam/tasks/multiple_choice.py",
    "mmlu_pro_physics": "src/memcontam/tasks/multiple_choice.py",
}
APPLICABILITY_PATHS: Final = {
    "game24": "data/phase12/registries/candidate_registry_v1.json",
    "math_equation_balancer": "data/phase12/registries/candidate_registry_v1.json",
    "word_sorting": "data/phase12/registries/candidate_registry_v1.json",
    "mmlu_pro_engineering": "src/memcontam/readiness/phase13_new_mcq_candidate.py",
    "mmlu_pro_physics": "src/memcontam/readiness/phase13_new_mcq_candidate.py",
}


class BoundIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ObservabilityRegistrationPacket(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["phase13_observability_registration_packet_v1"]
    packet_id: Literal["OBSERVABILITY_REGISTRATION_PACKET_V1"]
    authority_hashes: dict[str, str]
    failure_classes: dict[str, str]
    recurrence_lookback_h: Literal[10]
    exposure_conditioning: Literal["CURRENT_Z_T_EQUALS_1_PRIOR_MATCH_NEED_NOT_BE_EXPOSED"]
    exact_lineage_recurrence: Literal["SAME_EXACT_ROOT_EXPOSED_AT_BOTH_OCCURRENCES"]
    post_eviction: Literal[
        "EXACT_ROOT_PRESENT_EXPLICITLY_REMOVED_ABSENT_AFTER_NEXT_ORDINARY_ROW_FIRST_RISK"
    ]
    retention: Literal["FIRST_CONTINUOUS_FINITE_WINDOW_EPISODE_NO_GAP_BRIDGING"]
    censoring: Literal["RIGHT_CENSORED_AT_REGISTERED_OR_FIXTURE_ENDPOINT"]
    u_t_status: Literal["NOT_REGISTERED_FOR_CURRENT_MAIN"]
    implementation_identities: dict[str, BoundIdentity]
    verifier_identities: dict[str, BoundIdentity]
    applicability_identities: dict[str, BoundIdentity]

    @model_validator(mode="after")
    def _exact_registry(self) -> ObservabilityRegistrationPacket:
        expected_tasks = set(REGISTERED_FAILURE_CLASSES)
        if (
            self.failure_classes != REGISTERED_FAILURE_CLASSES
            or set(self.verifier_identities) != expected_tasks
            or set(self.applicability_identities) != expected_tasks
            or set(self.implementation_identities)
            != {"registration", "sequence", "authority_state"}
            or self.authority_hashes != AUTHORITY_HASHES
            or {
                task: identity.path for task, identity in self.verifier_identities.items()
            }
            != VERIFIER_PATHS
            or {
                task: identity.path for task, identity in self.applicability_identities.items()
            }
            != APPLICABILITY_PATHS
        ):
            raise Phase13ObservabilityError("OBSERVABILITY_REGISTRATION_PACKET_STALE")
        return self


def load_registration_packet(path: Path) -> ObservabilityRegistrationPacket:
    return ObservabilityRegistrationPacket.model_validate_json(path.read_bytes())


def classify_registered_failure(
    task: str,
    verified_outcome: int,
    precomputed_failure_class: str | None,
) -> MetricValue:
    expected = REGISTERED_FAILURE_CLASSES.get(task)
    if expected is None or precomputed_failure_class not in {None, expected}:
        raise Phase13ObservabilityError("UNKNOWN_TASK_FAILURE_CLASS")
    if verified_outcome == 1:
        if precomputed_failure_class is not None:
            raise Phase13ObservabilityError("CORRECT_RESPONSE_HAS_FAILURE_CLASS")
        return MetricValue(status="supported", reason="NO_REGISTERED_SUBSTANTIVE_FAILURE")
    return MetricValue(
        status="supported",
        value=precomputed_failure_class,
        reason=(
            "PACKET_BOUND_PRECOMPUTED_KAPPA"
            if precomputed_failure_class is not None
            else "NO_REGISTERED_SUBSTANTIVE_FAILURE"
        ),
    )


__all__ = [
    "BoundIdentity",
    "APPLICABILITY_PATHS",
    "AUTHORITY_HASHES",
    "ObservabilityRegistrationPacket",
    "REGISTERED_FAILURE_CLASSES",
    "VERIFIER_PATHS",
    "classify_registered_failure",
    "load_registration_packet",
]
