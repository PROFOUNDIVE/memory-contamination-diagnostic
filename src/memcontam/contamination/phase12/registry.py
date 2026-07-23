from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from memcontam.contamination.phase12.certification import CertificationSuite, certify_triplet
from memcontam.contamination.phase12.models import (
    CandidateCertificationError,
    CandidateRegistry,
    CandidateTriplet,
    CandidateVariant,
    CertificationEvidence,
    FrozenCandidateRegistry,
    HiddenAuditOrigin,
    HiddenAuditRegistry,
    PRIMARY_TASKS,
    canonical_content_hash,
    canonical_json_hash,
)


def load_candidate_registry(path: Path) -> CandidateRegistry:
    payload = _read_json(path)
    _reject_selection_markers(payload)
    try:
        triplets = tuple(_parse_triplet(item) for item in payload["triplets"])
        registry = CandidateRegistry(
            registry_id=payload["registry_id"],
            schema_version=payload["schema_version"],
            frozen_at=payload["frozen_at"],
            audit_registry_id=payload["audit_registry_id"],
            audit_registry_hash=payload["audit_registry_hash"],
            triplets=triplets,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CandidateCertificationError("INVALID_CANDIDATE_REGISTRY") from exc
    _validate_registry(registry)
    return registry


def load_hidden_audit_registry(path: Path) -> HiddenAuditRegistry:
    payload = _read_json(path)
    try:
        origins = tuple(
            HiddenAuditOrigin(
                candidate_id=item["candidate_id"],
                origin_class=item["origin_class"],
                source_reference=item["source_reference"],
                independent_of_outcomes=item["independent_of_outcomes"],
                content_hash=item["content_hash"],
            )
            for item in payload["origins"]
        )
        registry = HiddenAuditRegistry(
            registry_id=payload["registry_id"],
            schema_version=payload["schema_version"],
            frozen_at=payload["frozen_at"],
            origins=origins,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CandidateCertificationError("INVALID_AUDIT_REGISTRY") from exc
    if any(
        origin.content_hash
        != canonical_json_hash(
            {
                "candidate_id": origin.candidate_id,
                "origin_class": origin.origin_class,
                "source_reference": origin.source_reference,
                "independent_of_outcomes": origin.independent_of_outcomes,
            }
        )
        for origin in registry.origins
    ):
        raise CandidateCertificationError("AUDIT_CONTENT_HASH_MISMATCH")
    return registry


def freeze_registry(
    registry: CandidateRegistry, suite: CertificationSuite
) -> FrozenCandidateRegistry:
    results = tuple(certify_triplet(triplet, suite) for triplet in registry.triplets)
    return FrozenCandidateRegistry(
        registry=registry,
        certification_results=results,
        artifact_hash=canonical_json_hash(
            {
                "registry_hash": registry.artifact_hash,
                "certification_results": results,
            }
        ),
    )


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CandidateCertificationError("CANDIDATE_REGISTRY_READ_ERROR") from exc
    if not isinstance(payload, dict):
        raise CandidateCertificationError("INVALID_CANDIDATE_REGISTRY")
    return payload


def _parse_triplet(payload: Mapping[str, Any]) -> CandidateTriplet:
    counterexample = _counterexample(payload["counterexample"])
    evidence_payload = payload["certification_evidence"]
    evidence = CertificationEvidence(
        certification_id=evidence_payload["certification_id"],
        suite_id=evidence_payload["suite_id"],
        evidence_layer=evidence_payload["evidence_layer"],
        counterexample=_counterexample(evidence_payload["counterexample"]),
        content_hash=evidence_payload["content_hash"],
    )
    return CandidateTriplet(
        triplet_id=payload["triplet_id"],
        task=payload["task"],
        false_candidate=_parse_variant(payload["false_candidate"]),
        correct_twin=_parse_variant(payload["correct_twin"]),
        irrelevant_control=_parse_variant(payload["irrelevant_control"]),
        counterexample=counterexample,
        certification_evidence=evidence,
        frozen_at=payload["frozen_at"],
    )


def _parse_variant(payload: Mapping[str, Any]) -> CandidateVariant:
    return CandidateVariant(
        candidate_id=payload["candidate_id"],
        rule_id=payload["rule_id"],
        role=payload["role"],
        applicability=tuple(payload["applicability"]),
        render_id=payload["render_id"],
        control_id=payload.get("control_id"),
        content=payload["content"],
        content_hash=payload["content_hash"],
        code_variant=payload.get("code_variant"),
        main_example_ids=tuple(payload.get("main_example_ids", ())),
        outcome_selected=payload.get("outcome_selected", False),
        in_b_star=payload.get("in_b_star", False),
    )


def _counterexample(value: Any) -> str | tuple[str, str]:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and len(value) == 2 and all(isinstance(item, str) for item in value):
        return tuple(value)
    raise CandidateCertificationError("INVALID_COUNTEREXAMPLE")


def _validate_registry(registry: CandidateRegistry) -> None:
    if registry.schema_version != "phase12-candidate-registry-v1":
        raise CandidateCertificationError("INVALID_CANDIDATE_REGISTRY")
    if tuple(triplet.task for triplet in registry.triplets) != PRIMARY_TASKS:
        raise CandidateCertificationError("PRIMARY_TRIPLET_SET_MISMATCH")
    if len({triplet.triplet_id for triplet in registry.triplets}) != len(registry.triplets):
        raise CandidateCertificationError("DUPLICATE_TRIPLET_ID")
    for triplet in registry.triplets:
        if triplet.frozen_at != registry.frozen_at:
            raise CandidateCertificationError("FREEZE_TIMESTAMP_MISMATCH")
        if triplet.certification_evidence.evidence_layer not in {"build", "calibration"}:
            raise CandidateCertificationError("MAIN_EVIDENCE_FORBIDDEN")
        if triplet.certification_evidence.content_hash != canonical_json_hash(
            triplet.certification_evidence.canonical_payload
        ):
            raise CandidateCertificationError("CERTIFICATION_EVIDENCE_HASH_MISMATCH")
        for candidate in (
            triplet.false_candidate,
            triplet.correct_twin,
            triplet.irrelevant_control,
        ):
            if candidate.content_hash != canonical_content_hash(candidate.content):
                raise CandidateCertificationError("CANDIDATE_CONTENT_HASH_MISMATCH")
        if triplet.false_candidate.code_variant is None:
            raise CandidateCertificationError("MISSING_CODE_VARIANT")


def _reject_selection_markers(payload: Mapping[str, Any]) -> None:
    forbidden = {"outcome_selected", "selected_by_outcomes"}

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                if key in forbidden and nested:
                    raise CandidateCertificationError("OUTCOME_SELECTED_CANDIDATE")
                if key == "selection_basis" and isinstance(nested, str) and "outcome" in nested:
                    raise CandidateCertificationError("OUTCOME_SELECTED_CANDIDATE")
                if key == "main_example_leakage" and nested:
                    raise CandidateCertificationError("MAIN_EXAMPLE_LEAKAGE")
                if key == "main_example_ids" and nested:
                    raise CandidateCertificationError("MAIN_EXAMPLE_LEAKAGE")
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(payload)
