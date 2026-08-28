from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from memcontam.experiment.phase12.filter_challenge.mft_state_models import JsonValue
from memcontam.readiness.phase13_authority_files import read_regular_nofollow
from memcontam.readiness.phase13_cost_policy import validate_cost_policy_package
from memcontam.readiness.phase13_execution_contract import CORE_MAIN_REGISTRY


Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
ACTIVATION_PATH = Path("data/phase13/main/cost_envelope_v2/activated_policy_v1.json")


class Phase13CostActivationError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class AuthorityIdentity(_FrozenModel):
    router_sha256: Sha256
    post_cutoff_addendum_sha256: Sha256
    experiment_design_v10_sha256: Sha256


class CandidateIdentity(_FrozenModel):
    path: Literal["data/phase13/main/cost_envelope_v2/candidate_manifest_v1.json"]
    sha256: Sha256
    provenance_status: Literal["PRESERVED_HISTORICAL_CONTROLLED_HANDOFF"]


class ActivatedCostPolicy(_FrozenModel):
    schema_version: Literal["phase13_activated_cost_policy_v1"]
    status: Literal["PASS"]
    authority: AuthorityIdentity
    candidate: CandidateIdentity
    cmax_main_krw: Literal[442130]
    core_authorization_gate_krw: Literal[450000]
    margin_to_core_gate_krw: Literal[7870]
    main_execution_authorized: Literal[False]
    main_a_measured_scientific_execution_count: Literal[0]
    activation_hash: Sha256


def validate_activated_cost_policy(root: Path) -> ActivatedCostPolicy:
    try:
        artifact = ActivatedCostPolicy.model_validate_json(
            read_regular_nofollow(root / ACTIVATION_PATH)
        )
    except (OSError, ValidationError) as error:
        raise Phase13CostActivationError("COST_ACTIVATION_ARTIFACT_INVALID") from error
    authority = dict(CORE_MAIN_REGISTRY.authority_stack)
    expected_authority = AuthorityIdentity(
        router_sha256=CORE_MAIN_REGISTRY.authority_router_sha256,
        post_cutoff_addendum_sha256=authority["post_cutoff_addendum"],
        experiment_design_v10_sha256=authority["experiment_design"],
    )
    if artifact.authority != expected_authority:
        raise Phase13CostActivationError("COST_ACTIVATION_AUTHORITY_MISMATCH")
    candidate_path = root / artifact.candidate.path
    if hashlib.sha256(read_regular_nofollow(candidate_path)).hexdigest() != artifact.candidate.sha256:
        raise Phase13CostActivationError("COST_ACTIVATION_CANDIDATE_MISMATCH")
    report = validate_cost_policy_package(root)
    if (
        report.cmax_main_krw != artifact.cmax_main_krw
        or report.core_authorization_gate_krw != artifact.core_authorization_gate_krw
        or report.margin_krw != artifact.margin_to_core_gate_krw
    ):
        raise Phase13CostActivationError("COST_ACTIVATION_PROOF_MISMATCH")
    payload = artifact.model_dump(mode="json", exclude={"activation_hash"})
    if _canonical_hash(payload) != artifact.activation_hash:
        raise Phase13CostActivationError("COST_ACTIVATION_HASH_MISMATCH")
    return artifact


def _canonical_hash(payload: dict[str, JsonValue]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


__all__ = ["Phase13CostActivationError", "validate_activated_cost_policy"]
