from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from memcontam.experiment.phase12.contracts import (
    FidelityCertificate,
    Phase12IntegrationCertificate,
    ValidatedExploratoryActivation,
    ValidatedRouteSelection,
    canonical_json_hash,
)
from memcontam.manifests.archive_validation import ArchiveValidationReport


_PRE_ROUTE_FAMILIES = {"readiness", "pilot_a", "pilot_b", "behavioral"}
_ROUTE_BOUND_FAMILIES = {"main_a", "main_b", "main_c", "sequential", "extension"}


class AdmissionDenied(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class ScientificRunRequest:
    run_family: str
    candidate: str
    mode: str
    scientific_result: bool
    trajectory_seed: int | None
    abstract_seed_slot: str | None
    route_selection_manifest_id: str | None = None
    seed_allocation_manifest_id: str | None = None
    exploratory_activation_manifest_id: str | None = None

    @property
    def is_exploratory(self) -> bool:
        return self.run_family == "exploratory_code" or self.mode == "python_sandbox"


@dataclass(frozen=True)
class CertificateBundle:
    bfv2_certificate: FidelityCertificate | None
    p12i_certificate: Phase12IntegrationCertificate | None
    p12i_artifacts: Mapping[str, Any]

    @classmethod
    def empty(cls) -> CertificateBundle:
        return cls(None, None, {})

    @property
    def has_evidence(self) -> bool:
        return (
            self.bfv2_certificate is not None
            or self.p12i_certificate is not None
            or bool(self.p12i_artifacts)
        )


@dataclass(frozen=True)
class AdmissionDecision:
    scientific_admission_ref: dict[str, str] | None

    @property
    def admitted(self) -> bool:
        return True


def evaluate_scientific_admission(
    request: ScientificRunRequest,
    certificates: CertificateBundle,
    archive: ArchiveValidationReport,
    route_selection: ValidatedRouteSelection | None,
    exploratory_activation: ValidatedExploratoryActivation | None,
) -> AdmissionDecision:
    if request.is_exploratory:
        return _evaluate_exploratory(
            request, certificates, archive, route_selection, exploratory_activation
        )
    if request.run_family not in _PRE_ROUTE_FAMILIES | _ROUTE_BOUND_FAMILIES:
        raise AdmissionDenied("RUN_FAMILY_UNSUPPORTED")
    if not request.scientific_result:
        if certificates.has_evidence:
            raise AdmissionDenied("ADMISSION_EVIDENCE_FORBIDDEN")
        return AdmissionDecision(None)
    reference = _validate_readiness(certificates, archive)
    if request.run_family in _PRE_ROUTE_FAMILIES:
        if route_selection is not None or request.route_selection_manifest_id is not None:
            raise AdmissionDenied("ROUTE_SELECTION_FORBIDDEN_PRE_ROUTE")
        if request.seed_allocation_manifest_id is not None:
            raise AdmissionDenied("SEED_ALLOCATION_FORBIDDEN_PRE_ROUTE")
        return AdmissionDecision(reference)
    _validate_route(request, route_selection)
    return AdmissionDecision(reference)


def _evaluate_exploratory(
    request: ScientificRunRequest,
    certificates: CertificateBundle,
    archive: ArchiveValidationReport,
    route_selection: ValidatedRouteSelection | None,
    exploratory_activation: ValidatedExploratoryActivation | None,
) -> AdmissionDecision:
    if request.run_family != "exploratory_code" or request.mode != "python_sandbox":
        raise AdmissionDenied("EXPLORATORY_MODE_MISMATCH")
    if not request.scientific_result:
        if certificates.has_evidence:
            raise AdmissionDenied("ADMISSION_EVIDENCE_FORBIDDEN")
        if request.route_selection_manifest_id is not None:
            raise AdmissionDenied("SOURCE_ROUTE_SELECTION_FORBIDDEN")
        if request.seed_allocation_manifest_id is not None:
            raise AdmissionDenied("SOURCE_SEED_ALLOCATION_FORBIDDEN")
        if request.exploratory_activation_manifest_id is not None:
            raise AdmissionDenied("EXPLORATORY_ACTIVATION_FORBIDDEN")
        return AdmissionDecision(None)
    reference = _validate_readiness(certificates, archive)
    route = _validate_route(request, route_selection, validate_slot=False)
    if request.exploratory_activation_manifest_id is None:
        raise AdmissionDenied("EXPLORATORY_ACTIVATION_REQUIRED")
    if exploratory_activation is None:
        raise AdmissionDenied("EXPLORATORY_ACTIVATION_REQUIRED")
    if (
        exploratory_activation.estimated_exploratory_calls
        > exploratory_activation.exploratory_call_budget
    ):
        raise AdmissionDenied("EXPLORATORY_BUDGET_INSUFFICIENT")
    if (
        exploratory_activation.exploratory_call_budget
        + exploratory_activation.reproducibility_reserve
        > exploratory_activation.remaining_call_capacity
    ):
        raise AdmissionDenied("REPRODUCIBILITY_RESERVE_INSUFFICIENT")
    if (
        exploratory_activation.route_selection_manifest_id != route.route_selection_manifest_id
        or exploratory_activation.seed_allocation_manifest_id != route.seed_allocation_manifest_id
        or exploratory_activation.exploratory_activation_manifest_id
        != request.exploratory_activation_manifest_id
    ):
        raise AdmissionDenied("EXPLORATORY_ACTIVATION_INVALID")
    _validate_slot(
        request.abstract_seed_slot,
        request.trajectory_seed,
        exploratory_activation.exploratory_slot_to_seed,
    )
    return AdmissionDecision(reference)


def _validate_readiness(
    certificates: CertificateBundle, archive: ArchiveValidationReport
) -> dict[str, str]:
    from memcontam.readiness.phase12_certificate import CertificateError, validate_p12i

    if certificates.bfv2_certificate is None:
        raise AdmissionDenied("BFV2_CERTIFICATE_REQUIRED")
    if certificates.p12i_certificate is None:
        raise AdmissionDenied("P12I_CERTIFICATE_REQUIRED")
    artifacts = dict(certificates.p12i_artifacts)
    artifacts["bfv2"] = certificates.bfv2_certificate
    try:
        report = validate_p12i(certificates.p12i_certificate, artifacts)
    except CertificateError as error:
        raise AdmissionDenied(error.code) from error
    if not report.scientific_admission:
        raise AdmissionDenied("F1C_NOT_PASS")
    if not archive.archive_valid:
        raise AdmissionDenied(archive.reason_code or "READINESS_ARCHIVE_NOT_PASS")
    return {
        "bfv2_certificate_id": certificates.bfv2_certificate.certificate_id,
        "p12i_certificate_id": certificates.p12i_certificate.certificate_id,
        "readiness_bundle_hash": canonical_json_hash(
            {
                "archive": archive.to_dict(),
                "bfv2": certificates.bfv2_certificate.model_dump(mode="json"),
                "p12i": certificates.p12i_certificate.model_dump(mode="json"),
            }
        ),
    }


def _validate_route(
    request: ScientificRunRequest,
    route_selection: ValidatedRouteSelection | None,
    *,
    validate_slot: bool = True,
) -> ValidatedRouteSelection:
    if request.route_selection_manifest_id is None:
        raise AdmissionDenied("ROUTE_SELECTION_REQUIRED")
    if request.seed_allocation_manifest_id is None:
        raise AdmissionDenied("SEED_ALLOCATION_REQUIRED")
    if route_selection is None:
        raise AdmissionDenied("ROUTE_SELECTION_INVALID")
    if request.candidate != route_selection.selected_route:
        raise AdmissionDenied("ROUTE_SELECTION_MISMATCH")
    if (
        request.route_selection_manifest_id != route_selection.route_selection_manifest_id
        or request.seed_allocation_manifest_id != route_selection.seed_allocation_manifest_id
    ):
        raise AdmissionDenied("ROUTE_SELECTION_MISMATCH")
    if validate_slot:
        _validate_slot(
            request.abstract_seed_slot, request.trajectory_seed, route_selection.slot_to_seed
        )
    return route_selection


def _validate_slot(slot: str | None, seed: int | None, slots: Mapping[str, int]) -> None:
    if slot is None or seed is None or slots.get(slot) != seed:
        raise AdmissionDenied("SEED_ASSIGNMENT_MISMATCH")


__all__ = [
    "AdmissionDecision",
    "AdmissionDenied",
    "CertificateBundle",
    "ScientificRunRequest",
    "evaluate_scientific_admission",
]
