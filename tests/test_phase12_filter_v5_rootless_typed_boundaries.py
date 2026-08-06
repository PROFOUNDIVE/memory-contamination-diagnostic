from __future__ import annotations

import pytest

from memcontam.experiment.phase12.filter_challenge.bct import (
    BCTAuthorizationError,
    BCTContractError,
    authorize_client_construction,
    validate_bct_evidence,
)
from memcontam.experiment.phase12.filter_challenge.registry import parse_selected_policy
from memcontam.experiment.phase12.filter_challenge.registry_common import RegistryValidationError
from memcontam.experiment.phase12.filter_challenge.rootless_local_firewall import (
    has_forbidden_rootless_profile,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_models import (
    RootlessLocalReceipt,
)
from memcontam.manifests.aggregate_manifest import AggregateManifest
from memcontam.manifests.archive_validation import ArchiveValidationReport
from memcontam.manifests.claim_scope import ClaimScopeError, build_claim_scope
from memcontam.readiness.scientific_admission import (
    AdmissionDenied,
    CertificateBundle,
    evaluate_scientific_admission,
)


ROOTLESS_FORBIDDEN = "ROOTLESS_PROFILE_FORBIDDEN"
ROOTLESS_RECEIPT = {
    "schema_version": "rootless_local_receipt_v1",
    "profile": "local_rootless_non_authoritative",
    "kind": "rootless_local_receipt",
    "terminal": "LOCAL_ROOTLESS_BCT_REVIEW_REQUIRED",
}


def test_typed_rootless_receipt_is_rejected_at_direct_api_seams() -> None:
    receipt = RootlessLocalReceipt.model_validate(ROOTLESS_RECEIPT)

    assert has_forbidden_rootless_profile(receipt)
    with pytest.raises(ClaimScopeError, match=ROOTLESS_FORBIDDEN):
        build_claim_scope((receipt,), AggregateManifest(()))
    with pytest.raises(AdmissionDenied, match=ROOTLESS_FORBIDDEN):
        evaluate_scientific_admission(
            receipt, CertificateBundle.empty(), ArchiveValidationReport(True, 0), None, None
        )
    with pytest.raises(BCTContractError, match=ROOTLESS_FORBIDDEN):
        validate_bct_evidence(receipt)
    with pytest.raises(RegistryValidationError, match=ROOTLESS_FORBIDDEN):
        parse_selected_policy(receipt)
    with pytest.raises(BCTAuthorizationError, match=ROOTLESS_FORBIDDEN):
        authorize_client_construction(receipt, receipt, pytest.fail)
