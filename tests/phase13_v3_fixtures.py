from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Final

from memcontam.logging.schema import MethodCall
from memcontam.readiness.phase13_cost_policy_models import RateCard, StageEnvelope
from memcontam.readiness.phase13_main_live_dispatch import MainUnitDispatchOutput
from memcontam.readiness.phase13_main_production import ProductionObject
from memcontam.readiness.phase13_main_runner_models import MainRunBinding, MainRunError

AUTHORITY: Final = hashlib.sha256(b"synthetic-v3-authority").hexdigest()


@dataclass(frozen=True, slots=True)
class SyntheticAuthority:
    identity: str = AUTHORITY


@dataclass(frozen=True, slots=True)
class SyntheticPackage:
    authority_sha256: str = AUTHORITY
    package_id: str = "phase13-main-a-corrected-execution-freeze-v3"


@dataclass(frozen=True, slots=True)
class SyntheticProof:
    authority_sha256: str = AUTHORITY
    proof_id: str = "phase13-main-a-corrected-cost-proof-v3"

    @property
    def rate_card(self) -> RateCard:
        return RateCard(
            input_usd_per_million="0.20", cached_input_usd_per_million="0.02",
            output_usd_per_million="1.20", cache_write_planning_premium="1.25",
            cache_read_credit="none", long_context_threshold_tokens=272000,
            long_context_input_multiplier="2.0", long_context_output_multiplier="1.5",
            fx_planning_ceiling_krw_per_usd=1600,
        )


@dataclass(frozen=True, slots=True)
class SyntheticAuthorization:
    authority_sha256: str = AUTHORITY
    authorization_id: str = "phase13-main-a-corrected-authorized-execution-v3"


@dataclass(frozen=True, slots=True)
class SyntheticRequest:
    authority_sha256: str = AUTHORITY
    messages: tuple[tuple[str, str], ...] = (("user", "synthetic request"),)


@dataclass(frozen=True, slots=True)
class SyntheticFixture:
    authority: SyntheticAuthority = SyntheticAuthority()
    package: SyntheticPackage = SyntheticPackage()
    proof: SyntheticProof = SyntheticProof()
    authorization: SyntheticAuthorization = SyntheticAuthorization()
    request: SyntheticRequest = SyntheticRequest()

    def __post_init__(self) -> None:
        if any(bound.authority_sha256 != self.authority.identity for bound in (
            self.package, self.proof, self.authorization, self.request,
        )):
            raise MainRunError("MAIN_AUTHORITY_BINDING_MISMATCH")

    def binding(self) -> MainRunBinding:
        return MainRunBinding(
            self.package.package_id, "1" * 64, "2" * 64,
            self.authorization.authorization_id, "3" * 64, "4" * 64, "5" * 64,
        )

    def units(self) -> tuple[ProductionObject, ...]:
        return tuple(
            ProductionObject(
                sequence=seed,
                unit_id=hashlib.sha256(f"synthetic-prefix-{seed}".encode()).hexdigest(),
                kind="CLEAN_PREFIX", seed=seed, task="game24", memory_baseline="fh_bounded",
                arm="NOT_APPLICABLE", prefix_unit_id=None, projected_cost_krw=10,
                execution_template_id="game24|fh_bounded|prefix",
                ordered_sample_ids_sha256="6" * 64, registration_packet_sha256="7" * 64,
                checkpoint_registry_sha256="8" * 64,
            )
            for seed in (0, 1)
        )


@dataclass(frozen=True, slots=True)
class HistoricalEvidenceRegistry:
    registry_id: str = "CORE_EXECUTION_ENVELOPE_REGISTRY_V2"
    registry_hash: str = "41cd7e7310a961d0856e2020b05a3ae455811fb0660455b4c7dfbcb0a9aafd93"
    stages: tuple[StageEnvelope, ...] = (StageEnvelope(
        semantic_stage_id="full_history_generate", authority_stage_id="FH_generation",
        suffix_calls=0, prefix_calls=1, calls=1,
        maximum_input_tokens=9330, maximum_output_tokens=512,
    ),)


@dataclass(frozen=True, slots=True)
class HistoricalEvidenceRetry:
    contract_id: str = "CORE_TRANSPORT_ATTEMPT_CONTRACT_V2"
    contract_hash: str = "1ee66fcb795f97d483c2ef976133ee61dbd5108c9dae851c2c2786ff496d788f"
    terminal_failure_contract_id: str = "CORE_TERMINAL_TECHNICAL_MISSINGNESS_V1"
    terminal_failure_contract_sha256: str = "9bbcdd9dd1686af034f7c0d2114ac86d5837a07de0cc6ba8fef7940bbc822b75"


@dataclass(frozen=True, slots=True)
class HistoricalEvidencePolicy:
    registry: HistoricalEvidenceRegistry = HistoricalEvidenceRegistry()
    retry: HistoricalEvidenceRetry = HistoricalEvidenceRetry()
    proof: SyntheticProof = SyntheticProof()


def prefix_output(unit: ProductionObject, *, cost_usd: float = 0.01) -> MainUnitDispatchOutput:
    fixture = SyntheticFixture()
    policy = HistoricalEvidencePolicy()
    messages = [{"role": role, "content": text} for role, text in fixture.request.messages]
    return MainUnitDispatchOutput(
        evidence={
            "evidence_kind": "CLEAN_PREFIX", "prefix_unit_id": unit.unit_id,
            "checkpoint": {
                "schema_version": "phase13_main_prefix_checkpoint_v1",
                "baseline": unit.memory_baseline, "checkpoint_id": "checkpoint-1",
                "checkpoint_identity_sha256": "b" * 64, "canonical_sha256": "c" * 64,
                "canonical_state_utf8": "{}",
            },
            "runtime_evidence": {
                "unit_id": unit.unit_id, "task": unit.task, "seed": unit.seed,
                "memory_baseline": unit.memory_baseline, "arm": unit.arm,
                "production_identity": {
                    "execution_template_id": unit.execution_template_id, "trajectory_seed": unit.seed,
                    "concrete_seed_id": str(unit.seed), "scientific_result": False,
                    "ordered_sample_ids_sha256": unit.ordered_sample_ids_sha256,
                    "registration_packet_sha256": unit.registration_packet_sha256,
                    "checkpoint_registry_sha256": unit.checkpoint_registry_sha256,
                },
                "observability_registration_packet_sha256": unit.registration_packet_sha256,
                "request": {
                    "api": "OpenAI Responses API", "model": "gpt-5.6-luna",
                    "service_tier": "default", "reasoning_mode": "standard",
                    "reasoning_effort": "none", "reasoning_context": "current_turn",
                    "previous_response_id": None, "store": False, "timeout_seconds": 180,
                    "retries_after_initial_attempt": 0, "semantic_invalid_generic_retry": False,
                },
            },
        },
        provider_calls=(MethodCall(
            call_id="prefix-call", stage="full_history_generate", messages=messages,
            raw_response="offline", model="gpt-5.6-luna", temperature=0.0, top_p=1.0,
            token_usage={"prompt_tokens": 3, "completion_tokens": 2}, transport_attempts=1,
            provider_status="completed", provider_response_status="completed",
            provider_response_id="synthetic-response",
            provider_usage={"input_tokens": 3, "output_tokens": 2},
            provider_service_tier="default", provider_returned_model="gpt-5.6-luna",
            provider_request_contract={
                "model": "gpt-5.6-luna", "input_sha256": hashlib.sha256(json.dumps(
                    messages, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                ).encode()).hexdigest(), "temperature": 0.0, "top_p": 1.0,
                "reasoning": {"mode": "standard", "effort": "none", "context": "current_turn"},
                "previous_response_id": None, "service_tier": "default", "store": False,
                "tools": [], "max_output_tokens": 512,
            },
            provider_authority_contract={
                "maximum_input_tokens": 9330, "maximum_output_tokens": 512,
                "execution_envelope_id": policy.registry.registry_id,
                "execution_envelope_sha256": policy.registry.registry_hash,
                "failure_contract_id": policy.retry.contract_id,
                "failure_contract_sha256": policy.retry.contract_hash,
                "terminal_failure_contract_id": policy.retry.terminal_failure_contract_id,
                "terminal_failure_contract_sha256": policy.retry.terminal_failure_contract_sha256,
                "rate_card_sha256": hashlib.sha256(json.dumps(
                    fixture.proof.rate_card.model_dump(mode="json"),
                    sort_keys=True, separators=(",", ":"),
                ).encode()).hexdigest(),
            },
            provider_cost_usd=cost_usd, authoritative_provider_cost_usd=cost_usd,
            derived_cost_usd=cost_usd, provider_cost_source="AUTHORITATIVE_PROVIDER",
        ),),
        realized_cost_krw=int(cost_usd * 1600),
    )
