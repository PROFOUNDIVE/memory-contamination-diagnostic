from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING
from functools import lru_cache
from pathlib import Path
from typing import Final, TypeVar

from pydantic import BaseModel, ValidationError

from memcontam.baselines.prompt_budget import count_prompt_tokens
from memcontam.clients.base import LLMClient, LLMResponse
from memcontam.readiness.phase13_authority_files import AuthorityFileError, read_regular_nofollow
from memcontam.readiness.phase13_cost_policy_models import (
    CandidateManifest,
    CostPolicyValidationReport,
    CostProof,
    RetryFailureContract,
    StageEnvelope,
    StageEnvelopeRegistry,
)
from memcontam.readiness.phase13_cost_policy_handoff import ControlledExternalWrite


PACKAGE = Path("data/phase13/main/cost_envelope_v2")
MANIFEST = PACKAGE / "candidate_manifest_corrected_v2.json"
INPUT_KRW_PER_TOKEN = Decimal("0.0004")
OUTPUT_KRW_PER_TOKEN = Decimal("0.00192")
ModelT = TypeVar("ModelT", bound=BaseModel)
CANONICAL_ARTIFACT_PATHS: Final = {
    "corrected_cost_envelope": PACKAGE / "corrected_cost_envelope_v2.txt",
    "stage_envelope_registry": PACKAGE / "stage_envelope_registry_corrected_v2.json",
    "retry_failure_contract": PACKAGE / "retry_failure_contract_v1.json",
    "cost_proof": PACKAGE / "cost_proof_corrected_v2.json",
    "controlled_external_write": PACKAGE / "controlled_external_write_v1.json",
    "residual_authority_patch": PACKAGE / "post_cutoff_addendum_residual_v1.patch",
}
CANONICAL_SOURCE_FILENAMES: Final = {
    "post_cutoff_addendum": "2026-08-24_Phase13_MainA_PostCutoff_Acceleration_Addendum.md",
    "experiment_v10": "Phase 13-Compatible Pilot Main and Exploratory Experiment Design revised-v10.md",
}
CANONICAL_SOURCE_HASHES: Final = {
    "post_cutoff_addendum": "786e1ef1db7656e38beb5ab9ec316adc7df9bb1cc2f16d389f3612c76fbd2015",
    "experiment_v10": "bf6cf602d3ead47e95d9e158c1e3fe89ffab1ba4093a40f7d7ccb781faa0e0ec",
}
CANONICAL_STAGES: Final = (
    ("full_history_generate", "FH_generation", 10000, 50, 10050, 9330, 512),
    ("rag_generate", "RAG_generation", 6000, 30, 6030, 344, 512),
    ("bot_problem_distill", "BoT_problem_distillation", 10000, 50, 10050, 1177, 384),
    ("bot_instantiate_solve", "BoT_solve", 10000, 50, 10050, 1949, 512),
    ("bot_thought_distill", "BoT_thought_distillation", 10000, 50, 10050, 2545, 384),
    ("reflexion_generate", "Reflexion_actor_generation", 20000, 50, 20050, 2282, 512),
    ("reflexion_reflect", "Reflexion_reflection", 20000, 50, 20050, 3349, 384),
    ("dc_rs_generate", "DC_RS_generation", 10000, 50, 10050, 9212, 512),
    ("dc_rs_synthesize", "DC_RS_writer_synthesis", 10000, 50, 10050, 13521, 8192),
    ("no_memory_generate", "NoMem_generation", 2500, 0, 2500, 1160, 512),
)
CANONICAL_RESIDUAL_PATCH: Final = b"""--- a/2026-08-24_Phase13_MainA_PostCutoff_Acceleration_Addendum.md
+++ b/2026-08-24_Phase13_MainA_PostCutoff_Acceleration_Addendum.md
@@ -3623,7 +3623,7 @@
 >
 this addendum on its enumerated narrow clauses
 >
-Experiment-v12
+Experiment-v10
 ```
 
 The router must treat the following as binding current-Main states:
"""


class Phase13CostPolicyError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        self.provider_attempts_count = (
            0 if code in {"INPUT_ENVELOPE_EXCEEDED", "SEMANTIC_STAGE_REQUIRED"} else 1
        )
        super().__init__(code)


class Phase13ProviderCallError(RuntimeError):
    def __init__(self, provider_error: Exception) -> None:
        self.provider_error = provider_error
        self.provider_attempts_count = getattr(provider_error, "provider_attempts_count", 1)
        self.provider_latency_ms = getattr(provider_error, "provider_latency_ms", 0)
        self.provider_status = getattr(provider_error, "provider_status", None)
        self.provider_incomplete_reason = getattr(
            provider_error, "provider_incomplete_reason", None
        )
        self.provider_usage = getattr(provider_error, "provider_usage", None)
        self.provider_token_usage = getattr(provider_error, "provider_token_usage", None)
        self.provider_cost_usd = getattr(provider_error, "provider_cost_usd", None)
        self.provider_response_id = getattr(provider_error, "provider_response_id", None)
        super().__init__(str(provider_error))


@dataclass(frozen=True, slots=True)
class CostPolicyBundle:
    manifest: CandidateManifest
    registry: StageEnvelopeRegistry
    retry: RetryFailureContract
    proof: CostProof


def _canonical_hash(
    model: CandidateManifest | StageEnvelopeRegistry | RetryFailureContract | CostProof,
    hash_field: str,
) -> str:
    payload = model.model_dump(mode="json", exclude={hash_field})
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _read(root: Path, relative: Path) -> bytes:
    try:
        return read_regular_nofollow(root / relative)
    except AuthorityFileError as error:
        raise Phase13CostPolicyError(error.code) from error


def _parse(model_type: type[ModelT], raw: bytes) -> ModelT:
    try:
        return model_type.model_validate_json(raw)
    except ValidationError as error:
        raise Phase13CostPolicyError("MALFORMED_COST_POLICY_ARTIFACT") from error


@lru_cache(maxsize=None)
def load_cost_policy_bundle(root: Path) -> CostPolicyBundle:
    manifest_raw = _read(root, MANIFEST)
    manifest = _parse(CandidateManifest, manifest_raw)
    if _canonical_hash(manifest, "manifest_hash") != manifest.manifest_hash:
        raise Phase13CostPolicyError("MANIFEST_HASH_MISMATCH")
    required = {
        "stage_envelope_registry",
        "retry_failure_contract",
        "cost_proof",
        "corrected_cost_envelope",
        "controlled_external_write",
        "residual_authority_patch",
    }
    if set(manifest.artifacts) != required:
        raise Phase13CostPolicyError("COST_POLICY_ARTIFACT_SET_MISMATCH")
    if any(
        manifest.artifacts[role].path != str(path)
        for role, path in CANONICAL_ARTIFACT_PATHS.items()
    ) or {
        role: source.filename
        for role, source in manifest.controlled_external_write_sources.items()
    } != CANONICAL_SOURCE_FILENAMES:
        raise Phase13CostPolicyError("CANONICAL_PATH_MISMATCH")
    if {
        role: source.source_sha256
        for role, source in manifest.controlled_external_write_sources.items()
    } != CANONICAL_SOURCE_HASHES:
        raise Phase13CostPolicyError("CONTROLLED_HANDOFF_MISMATCH")

    loaded: dict[str, bytes] = {}
    for role, identity in manifest.artifacts.items():
        raw = _read(root, Path(identity.path))
        if hashlib.sha256(raw).hexdigest() != identity.sha256:
            raise Phase13CostPolicyError("ARTIFACT_HASH_MISMATCH")
        loaded[role] = raw
    registry = _parse(StageEnvelopeRegistry, loaded["stage_envelope_registry"])
    retry = _parse(RetryFailureContract, loaded["retry_failure_contract"])
    proof = _parse(CostProof, loaded["cost_proof"])
    try:
        ControlledExternalWrite.model_validate_json(loaded["controlled_external_write"])
    except ValidationError as error:
        raise Phase13CostPolicyError("CONTROLLED_HANDOFF_MISMATCH") from error
    if loaded["residual_authority_patch"] != CANONICAL_RESIDUAL_PATCH:
        raise Phase13CostPolicyError("CONTROLLED_HANDOFF_MISMATCH")
    bundle = CostPolicyBundle(manifest, registry, retry, proof)
    _validate_bundle(bundle, root)
    return bundle


def _validate_bundle(bundle: CostPolicyBundle, root: Path) -> None:
    registry, retry, proof = bundle.registry, bundle.retry, bundle.proof
    if _canonical_hash(registry, "registry_hash") != registry.registry_hash:
        raise Phase13CostPolicyError("STAGE_ENVELOPE_HASH_MISMATCH")
    if _canonical_hash(retry, "contract_hash") != retry.contract_hash:
        raise Phase13CostPolicyError("RETRY_FAILURE_HASH_MISMATCH")
    if _canonical_hash(proof, "proof_hash") != proof.proof_hash:
        raise Phase13CostPolicyError("COST_PROOF_HASH_MISMATCH")
    if (
        proof.package_selection_path
        != "data/phase13/main/post_cutoff_package_selection_v2.json"
        or proof.common_capacity_path != "data/phase13/common_capacity_v1.json"
    ):
        raise Phase13CostPolicyError("CANONICAL_PATH_MISMATCH")
    if (
        proof.stage_envelope_registry_hash != registry.registry_hash
        or proof.stage_envelope_registry_id != registry.registry_id
        or proof.stage_envelope_registry_authority_sha256 != registry.authority_sha256
        or proof.retry_failure_contract_hash != retry.contract_hash
        or proof.retry_failure_contract_id != retry.contract_id
        or proof.terminal_failure_contract_id != retry.terminal_failure_contract_id
        or proof.terminal_failure_contract_sha256 != retry.terminal_failure_contract_sha256
        or hashlib.sha256(_read(root, Path(proof.package_selection_path))).hexdigest()
        != proof.package_selection_sha256
        or hashlib.sha256(_read(root, Path(proof.common_capacity_path))).hexdigest()
        != proof.common_capacity_sha256
    ):
        raise Phase13CostPolicyError("COST_POLICY_BINDING_MISMATCH")
    stages = {stage.semantic_stage_id: stage for stage in registry.stages}
    costs = {stage.semantic_stage_id: stage for stage in proof.stage_costs}
    observed_stages = tuple(
        (
            stage.semantic_stage_id,
            stage.authority_stage_id,
            stage.suffix_calls,
            stage.prefix_calls,
            stage.calls,
            stage.maximum_input_tokens,
            stage.maximum_output_tokens,
        )
        for stage in registry.stages
    )
    if observed_stages != CANONICAL_STAGES:
        raise Phase13CostPolicyError("CANONICAL_STAGE_MISMATCH")
    if len(stages) != 10 or set(stages) != set(costs):
        raise Phase13CostPolicyError("STAGE_REGISTRY_MISMATCH")
    if (
        sum(stage.prefix_calls for stage in registry.stages)
        != proof.reconciliation.prefix_semantic_calls
        or sum(stage.suffix_calls for stage in registry.stages)
        != proof.reconciliation.suffix_semantic_calls
        or any(stage.calls != stage.suffix_calls + stage.prefix_calls for stage in registry.stages)
        or hashlib.sha256(_read(root, Path(proof.cost_envelope_path)).removesuffix(b"\n")).hexdigest()
        != proof.cost_envelope_sha256
    ):
        raise Phase13CostPolicyError("CLEAN_PREFIX_RECONCILIATION_MISMATCH")
    total = 0
    for stage_id, stage in stages.items():
        input_krw = _ceil(stage.calls * stage.maximum_input_tokens * INPUT_KRW_PER_TOKEN)
        output_krw = _ceil(stage.calls * stage.maximum_output_tokens * OUTPUT_KRW_PER_TOKEN)
        cost = costs[stage_id]
        if (cost.input_krw_ceiling, cost.output_krw_ceiling, cost.stage_krw) != (
            input_krw,
            output_krw,
            input_krw + output_krw,
        ):
            raise Phase13CostPolicyError("STAGE_COST_MISMATCH")
        total += cost.stage_krw
    if (
        total != proof.cmax_main_krw
        or sum(stage.calls for stage in registry.stages) != proof.semantic_calls
        or proof.budget.core_authorization_gate_krw - total
        != proof.margin_to_core_gate_krw
    ):
        raise Phase13CostPolicyError("COST_PROOF_TOTAL_MISMATCH")


def _ceil(value: Decimal) -> int:
    return int(value.to_integral_value(rounding=ROUND_CEILING))


def validate_cost_policy_package(root: Path) -> CostPolicyValidationReport:
    bundle = load_cost_policy_bundle(root)
    proof, capacity = bundle.proof, bundle.registry.capacity
    return CostPolicyValidationReport(
        policy_id=bundle.manifest.policy_id,
        activation_status=bundle.manifest.activation_status,
        total_budget_ceiling_krw=proof.budget.total_budget_ceiling_krw,
        reserve_fraction=proof.budget.reserve_fraction,
        core_authorization_gate_krw=proof.budget.core_authorization_gate_krw,
        cmax_main_krw=proof.cmax_main_krw,
        margin_krw=proof.margin_to_core_gate_krw,
        writer_cap=capacity.writer_max_output_tokens,
        common_capacity_tokens=capacity.B_mem_tokens,
        maximum_transport_attempts=bundle.retry.maximum_transport_attempts_per_semantic_call,
        execution_envelope_registry_id=bundle.registry.registry_id,
        execution_envelope_registry_sha256=bundle.registry.authority_sha256,
        terminal_failure_contract_sha256=bundle.retry.terminal_failure_contract_sha256,
        cost_envelope_sha256=proof.cost_envelope_sha256,
        manifest_sha256=hashlib.sha256(_read(root, MANIFEST)).hexdigest(),
    )


class CostPolicyClient:
    def __init__(self, client: LLMClient, bundle: CostPolicyBundle) -> None:
        self._client = client
        self._retry = bundle.retry
        self._stages = {stage.semantic_stage_id: stage for stage in bundle.registry.stages}
        self._registry_id = bundle.registry.registry_id
        self._registry_sha256 = bundle.registry.registry_hash
        self._failure_contract_id = bundle.retry.contract_id
        self._failure_contract_sha256 = bundle.retry.contract_hash
        self._terminal_failure_contract_id = bundle.retry.terminal_failure_contract_id
        self._terminal_failure_contract_sha256 = bundle.retry.terminal_failure_contract_sha256
        self._rate_card_sha256 = hashlib.sha256(
            json.dumps(
                bundle.proof.rate_card.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

    def chat(self, messages: list[dict[str, str]], model: str, config: dict) -> LLMResponse:
        stage = self._stage(config.get("method_stage"))
        if count_prompt_tokens(messages, "o200k_base") > stage.maximum_input_tokens:
            raise Phase13CostPolicyError("INPUT_ENVELOPE_EXCEEDED")
        bound = {
            **config,
            "max_output_tokens": stage.maximum_output_tokens,
            "_phase13_maximum_input_tokens": stage.maximum_input_tokens,
            "_phase13_execution_envelope_id": self._registry_id,
            "_phase13_execution_envelope_sha256": self._registry_sha256,
            "_phase13_maximum_transport_attempts": self._retry.maximum_transport_attempts_per_semantic_call,
            "_phase13_failure_contract_id": self._failure_contract_id,
            "_phase13_failure_contract_sha256": self._failure_contract_sha256,
            "_phase13_terminal_failure_contract_id": self._terminal_failure_contract_id,
            "_phase13_terminal_failure_contract_sha256": self._terminal_failure_contract_sha256,
            "_phase13_rate_card_sha256": self._rate_card_sha256,
        }
        try:
            response = self._client.chat(messages, model, bound)
        except Exception as error:
            raise Phase13ProviderCallError(error) from error
        attempts = response.raw.get("attempts")
        if attempts != self._retry.maximum_transport_attempts_per_semantic_call:
            raise Phase13CostPolicyError("TRANSPORT_ATTEMPT_EXCEEDED")
        if response.token_usage.get("prompt_tokens", 0) > stage.maximum_input_tokens:
            raise Phase13CostPolicyError("OBSERVED_INPUT_ENVELOPE_EXCEEDED")
        if response.token_usage.get("completion_tokens", 0) > stage.maximum_output_tokens:
            raise Phase13CostPolicyError("OBSERVED_OUTPUT_ENVELOPE_EXCEEDED")
        return response

    def _stage(self, stage_id: object) -> StageEnvelope:
        if not isinstance(stage_id, str) or stage_id not in self._stages:
            raise Phase13CostPolicyError("SEMANTIC_STAGE_REQUIRED")
        return self._stages[stage_id]


def bind_cost_policy_client(client: LLMClient, root: Path) -> CostPolicyClient:
    return CostPolicyClient(client, load_cost_policy_bundle(root))


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    report = validate_cost_policy_package(root)
    print(json.dumps(report.model_dump(mode="json"), sort_keys=True))


__all__ = [
    "CostPolicyClient",
    "Phase13CostPolicyError",
    "Phase13ProviderCallError",
    "bind_cost_policy_client",
    "load_cost_policy_bundle",
    "main",
    "validate_cost_policy_package",
]


if __name__ == "__main__":
    main()
