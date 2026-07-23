from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Literal, cast


PRIMARY_TASKS = ("game24", "math_equation_balancer", "word_sorting")


class CandidateCertificationError(ValueError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def canonical_json_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        default=_json_default,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json_default(value: object) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(cast(Any, value))
    raise TypeError(f"cannot serialize {type(value)!r}")


def canonical_content_hash(content: str) -> str:
    return canonical_json_hash({"content": content})


@dataclass(frozen=True)
class CertificationEvidence:
    certification_id: str
    suite_id: str
    evidence_layer: Literal["build", "calibration"]
    counterexample: str | tuple[str, str]
    content_hash: str

    @property
    def canonical_payload(self) -> dict[str, Any]:
        return {
            "certification_id": self.certification_id,
            "suite_id": self.suite_id,
            "evidence_layer": self.evidence_layer,
            "counterexample": self.counterexample,
        }


@dataclass(frozen=True)
class CandidateVariant:
    candidate_id: str
    rule_id: str
    role: Literal["false", "correct", "irrelevant"]
    applicability: tuple[str, ...]
    render_id: str
    control_id: str | None
    content: str
    content_hash: str
    code_variant: str | None = None
    main_example_ids: tuple[str, ...] = ()
    outcome_selected: bool = False
    in_b_star: bool = False


@dataclass(frozen=True)
class CandidateTriplet:
    triplet_id: str
    task: Literal["game24", "math_equation_balancer", "word_sorting"]
    false_candidate: CandidateVariant
    correct_twin: CandidateVariant
    irrelevant_control: CandidateVariant
    counterexample: str | tuple[str, str]
    certification_evidence: CertificationEvidence
    frozen_at: str


@dataclass(frozen=True)
class CandidateRegistry:
    registry_id: str
    schema_version: str
    frozen_at: str
    audit_registry_id: str
    audit_registry_hash: str
    triplets: tuple[CandidateTriplet, ...]

    @property
    def artifact_hash(self) -> str:
        return canonical_json_hash(
            {
                "registry_id": self.registry_id,
                "schema_version": self.schema_version,
                "frozen_at": self.frozen_at,
                "audit_registry_id": self.audit_registry_id,
                "audit_registry_hash": self.audit_registry_hash,
                "triplets": self.triplets,
            }
        )


@dataclass(frozen=True)
class HiddenAuditOrigin:
    candidate_id: str
    origin_class: Literal["protocol_injected"]
    source_reference: str
    independent_of_outcomes: Literal[True]
    content_hash: str


@dataclass(frozen=True)
class HiddenAuditRegistry:
    registry_id: str
    schema_version: str
    frozen_at: str
    origins: tuple[HiddenAuditOrigin, ...]

    @property
    def artifact_hash(self) -> str:
        return canonical_json_hash(
            {
                "registry_id": self.registry_id,
                "schema_version": self.schema_version,
                "frozen_at": self.frozen_at,
                "origins": self.origins,
            }
        )


@dataclass(frozen=True)
class CertificationResult:
    triplet_id: str
    suite_id: str
    counterexample: str | tuple[str, str]
    false_rule_result: bool | int
    correct_rule_result: bool | int
    passed: bool
    evidence_hash: str


@dataclass(frozen=True)
class FrozenCandidateRegistry:
    registry: CandidateRegistry
    certification_results: tuple[CertificationResult, ...]
    artifact_hash: str
