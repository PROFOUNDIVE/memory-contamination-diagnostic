from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from memcontam.experiment.phase12.filter_challenge.bct import (
    BCTContractError,
    BCTEvidence,
    validate_bct_evidence,
)
from memcontam.experiment.phase12.filter_challenge.bct_live_authorization import (
    CalibrationAuthorizationError,
    load_authorization,
)
from memcontam.experiment.phase12.filter_challenge.mft_state_models import JsonValue
from memcontam.experiment.phase12.filter_challenge.pilot_b_readiness import (
    readiness_from_fixture,
)
from memcontam.experiment.phase12.filter_challenge.registry import parse_selected_policy
from memcontam.experiment.phase12.filter_challenge.registry_calibration import (
    CalibrationStageResult,
    ScreeningAuthorizationV1,
)
from memcontam.experiment.phase12.filter_challenge.registry_common import (
    RegistryValidationError,
)
from memcontam.experiment.phase12.filter_challenge.registry_search import SelectedPolicy
from memcontam.experiment.phase12.filter_challenge.rootless_local_firewall import (
    has_forbidden_rootless_profile,
)
from memcontam.manifests.aggregate_manifest import AggregateManifest
from memcontam.manifests.archive_validation import ArchiveValidationReport
from memcontam.manifests.claim_scope import ClaimScopeError, build_claim_scope
from memcontam.readiness.scientific_admission import (
    AdmissionDenied,
    CertificateBundle,
    evaluate_scientific_admission,
)


ROOTLESS_PROFILE = "local_rootless_non_authoritative"
ROOTLESS_FORBIDDEN = "ROOTLESS_PROFILE_FORBIDDEN"
ROOTLESS_RECEIPT = {
    "schema_version": "rootless_local_receipt_v1",
    "profile": ROOTLESS_PROFILE,
    "kind": "rootless_local_receipt",
    "terminal": "LOCAL_ROOTLESS_BCT_REVIEW_REQUIRED",
}
MALFORMED_ROOTLESS_RECEIPT = {
    **ROOTLESS_RECEIPT,
    "schema_version": "filter_challenge_bct_readiness_v1",
    "kind": "selected_policy",
    "terminal": "READY_FOR_SEPARATE_FILTER_V5_PILOT_B_AUTHORIZATION",
    "unexpected": True,
}


@pytest.fixture(params=(ROOTLESS_RECEIPT, MALFORMED_ROOTLESS_RECEIPT))
def rootless_payload(request: pytest.FixtureRequest) -> dict[str, JsonValue]:
    return dict(request.param)


def test_rootless_models_are_strict_closed_and_historically_incompatible() -> None:
    from memcontam.experiment.phase12.filter_challenge.rootless_local_models import (
        RootlessLocalReceipt,
    )

    receipt = RootlessLocalReceipt.model_validate(ROOTLESS_RECEIPT)

    assert tuple(RootlessLocalReceipt.model_fields) == (
        "schema_version",
        "profile",
        "kind",
        "terminal",
    )
    assert receipt.terminal == "LOCAL_ROOTLESS_BCT_REVIEW_REQUIRED"
    assert receipt.terminal != "READY_FOR_SEPARATE_FILTER_V5_PILOT_B_AUTHORIZATION"
    for field, replacement in (
        ("profile", "authoritative"),
        ("schema_version", "filter_challenge_bct_evidence_v1"),
        ("kind", "selected_policy"),
        ("terminal", "READY_FOR_SEPARATE_FILTER_V5_PILOT_B_AUTHORIZATION"),
    ):
        with pytest.raises(ValidationError):
            RootlessLocalReceipt.model_validate({**ROOTLESS_RECEIPT, field: replacement})
    for payload in (
        {key: value for key, value in ROOTLESS_RECEIPT.items() if key != "profile"},
        {**ROOTLESS_RECEIPT, "extra": "forbidden"},
    ):
        with pytest.raises(ValidationError):
            RootlessLocalReceipt.model_validate(payload)
    for historical_model in (BCTEvidence, CalibrationStageResult, SelectedPolicy):
        with pytest.raises(ValidationError):
            historical_model.model_validate(ROOTLESS_RECEIPT)


def test_rootless_discriminator_handles_duplicate_profile_and_untrusted_text() -> None:
    duplicate_profile = (
        b'{"profile":"local_rootless_non_authoritative","profile":"authoritative"}'
    )

    assert has_forbidden_rootless_profile(ROOTLESS_RECEIPT)
    assert has_forbidden_rootless_profile(duplicate_profile)
    assert not has_forbidden_rootless_profile(b"{not-json}\xff")


def test_claim_aggregation_rejects_rootless_before_claim_parsing(
    rootless_payload: dict[str, JsonValue],
) -> None:
    with pytest.raises(ClaimScopeError, match=ROOTLESS_FORBIDDEN) as error:
        build_claim_scope((rootless_payload,), AggregateManifest(()))

    assert error.value.code == ROOTLESS_FORBIDDEN


def test_scientific_admission_rejects_rootless_before_request_evaluation(
    rootless_payload: dict[str, JsonValue],
) -> None:
    with pytest.raises(AdmissionDenied, match=ROOTLESS_FORBIDDEN) as error:
        evaluate_scientific_admission(
            rootless_payload,
            CertificateBundle.empty(),
            ArchiveValidationReport(True, 0),
            None,
            None,
        )

    assert error.value.code == ROOTLESS_FORBIDDEN


def test_historical_bct_rejects_rootless_before_evidence_deserialization(
    rootless_payload: dict[str, JsonValue],
) -> None:
    with pytest.raises(BCTContractError, match=ROOTLESS_FORBIDDEN) as error:
        validate_bct_evidence(rootless_payload)

    assert error.value.code == ROOTLESS_FORBIDDEN


def test_pilot_b_readiness_rejects_rootless_before_fixture_deserialization(
    tmp_path: Path,
    rootless_payload: dict[str, JsonValue],
) -> None:
    fixture = tmp_path / "rootless-readiness.json"
    fixture.write_text(json.dumps(rootless_payload), encoding="utf-8")

    with pytest.raises(ValueError, match=ROOTLESS_FORBIDDEN):
        readiness_from_fixture(fixture)


def test_selected_policy_rejects_rootless_before_policy_deserialization(
    rootless_payload: dict[str, JsonValue],
) -> None:
    with pytest.raises(RegistryValidationError, match=ROOTLESS_FORBIDDEN) as error:
        parse_selected_policy(rootless_payload)

    assert error.value.code == ROOTLESS_FORBIDDEN


def test_authorization_rejects_rootless_before_digest_or_model_deserialization(
    tmp_path: Path,
    rootless_payload: dict[str, JsonValue],
) -> None:
    path = tmp_path / "rootless-authorization.json"
    path.write_text(json.dumps(rootless_payload), encoding="utf-8")

    for expected_digest in (hashlib.sha256(path.read_bytes()).hexdigest(), "0" * 64):
        with pytest.raises(CalibrationAuthorizationError, match=ROOTLESS_FORBIDDEN) as error:
            load_authorization(path, expected_digest, ScreeningAuthorizationV1)

        assert error.value.code == ROOTLESS_FORBIDDEN
