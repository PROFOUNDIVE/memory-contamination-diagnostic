from __future__ import annotations

from pydantic import ValidationError

from memcontam.readiness.phase13_v3_authority_models import AuthoritySnapshotV3

from .phase13_v3_cost_actual import reconcile_actual
from .phase13_v3_cost_law import calculate, decimal_string, exact_request_cost
from .phase13_v3_cost_models import (
    ActivatedPolicyV3,
    BaseCostInputsV3,
    CompleteCostInputsV3,
    CostError,
    CostProofV3,
    CostUnit,
    CostWitnessV3,
    FinalOrder,
    PrefreezeBindings,
    ProviderCostEvidence,
    ProviderUsage,
    RequestTokens,
    RetryReservation,
    StageOccurrences,
    canonical_bytes,
    check_hash,
    digest,
    seal,
)

__all__ = [
    "ActivatedPolicyV3",
    "BaseCostInputsV3",
    "CompleteCostInputsV3",
    "CostError",
    "CostProofV3",
    "CostUnit",
    "CostWitnessV3",
    "FinalOrder",
    "PrefreezeBindings",
    "ProviderCostEvidence",
    "ProviderUsage",
    "RequestTokens",
    "RetryReservation",
    "StageOccurrences",
    "activate_policy",
    "build_proof",
    "build_witness",
    "calculate",
    "canonical_bytes",
    "check_hash",
    "decimal_string",
    "digest",
    "exact_request_cost",
    "freeze_base",
    "freeze_complete",
    "reconcile_actual",
    "seal",
    "validate_base",
    "validate_complete",
    "validate_policy",
    "validate_policy_bytes",
    "validate_proof",
    "validate_witness",
]


def activate_policy(authority: AuthoritySnapshotV3) -> ActivatedPolicyV3:
    return seal(ActivatedPolicyV3(authority=authority, policy_hash="0" * 64))


def validate_policy(policy: ActivatedPolicyV3, authority: AuthoritySnapshotV3) -> None:
    validate_policy_bytes(canonical_bytes(policy), authority)


def validate_policy_bytes(raw: bytes, authority: AuthoritySnapshotV3) -> ActivatedPolicyV3:
    try:
        validated = ActivatedPolicyV3.model_validate_json(raw)
    except ValidationError as error:
        raise CostError("RATE_CARD_DRIFT_REQUIRES_REAPPROVAL") from error
    if raw != canonical_bytes(validated) or validated != activate_policy(authority):
        raise CostError("RATE_CARD_DRIFT_REQUIRES_REAPPROVAL")
    return validated


def freeze_base(policy: ActivatedPolicyV3, bindings: PrefreezeBindings, units: tuple[CostUnit, ...],
                *, retry_reservations: tuple[RetryReservation, ...] = ()) -> BaseCostInputsV3:
    validate_policy(policy, policy.authority)
    result = seal(BaseCostInputsV3(policy=policy, bindings=bindings,
                                  units=tuple(sorted(units, key=lambda unit: unit.unit_id)),
                                  retry_reservations=tuple(sorted(retry_reservations, key=lambda row: row.dispatch_id)),
                                  base_inputs_hash="0" * 64))
    _validate_base(result)
    return result


def _validate_base(base: BaseCostInputsV3) -> None:
    check_hash(base)
    validate_policy(base.policy, base.policy.authority)
    stage_ids = {stage.stage_id for stage in base.policy.authority.registry.stages}
    unit_ids = tuple(unit.unit_id for unit in base.units)
    if len(set(unit_ids)) != len(unit_ids) or unit_ids != tuple(sorted(unit_ids)):
        raise CostError("MAIN_COST_PROOF_MISMATCH")
    for unit in base.units:
        used = tuple(group.stage_id for group in unit.stages)
        if not set(used) <= stage_ids:
            raise CostError("MAIN_COST_PROOF_MISMATCH")
    retries = base.retry_reservations
    if (tuple(row.dispatch_id for row in retries) != tuple(sorted({row.dispatch_id for row in retries}))
        or sum(row.reservation_krw for row in retries) > base.policy.authority.retry.retry_budget_krw):
        raise CostError("MAIN_COST_PROOF_MISMATCH")
    by_unit = {unit.unit_id: {stage.stage_id for stage in unit.stages} for unit in base.units}
    from math import ceil
    for row in retries:
        if row.stage_id not in by_unit.get(row.unit_id, set()):
            raise CostError("MAIN_COST_PROOF_MISMATCH")
        envelope = next(stage for stage in base.policy.authority.registry.stages if stage.stage_id == row.stage_id)
        input_cost, output_cost = exact_request_cost(RequestTokens(
            input_tokens=envelope.maximum_input_tokens, output_tokens=envelope.maximum_output_tokens,
        ), base.policy.rate_card)
        if row.reservation_krw != ceil(input_cost) + ceil(output_cost):
            raise CostError("MAIN_COST_PROOF_MISMATCH")


def freeze_complete(base: BaseCostInputsV3, order: FinalOrder) -> CompleteCostInputsV3:
    _validate_base(base)
    if len(order.unit_ids) != len(base.units) or set(order.unit_ids) != {unit.unit_id for unit in base.units}:
        raise CostError("MAIN_COST_PROOF_MISMATCH")
    return seal(CompleteCostInputsV3(base=base, base_inputs_hash=base.base_inputs_hash,
                                    final_order=order, final_order_hash=digest(order), complete_inputs_hash="0" * 64))


def build_witness(base: BaseCostInputsV3) -> CostWitnessV3:
    _validate_base(base)
    totals, _ = calculate(base)
    return seal(CostWitnessV3(base_inputs_hash=base.base_inputs_hash, totals=totals, witness_hash="0" * 64))


def build_proof(complete: CompleteCostInputsV3, witness: CostWitnessV3, package_core_hash: str) -> CostProofV3:
    if complete != freeze_complete(complete.base, complete.final_order) or witness != build_witness(complete.base):
        raise CostError("MAIN_COST_PROOF_MISMATCH")
    totals, projections = calculate(complete.base, complete.final_order.unit_ids)
    if totals != witness.totals or sum(row.projected_krw for row in projections) != totals.cmax_main_krw:
        raise CostError("MAIN_COST_PROOF_MISMATCH")
    return seal(CostProofV3(proof_id=complete.base.policy.authority.identity.cost_proof_id,
                           complete_inputs_hash=complete.complete_inputs_hash,
                           witness_hash=witness.witness_hash, package_core_hash=package_core_hash,
                           totals=totals, projected_krw=projections, proof_hash="0" * 64))


def validate_witness(raw: bytes, base: BaseCostInputsV3) -> CostWitnessV3:
    try:
        result = CostWitnessV3.model_validate_json(raw)
    except ValidationError as error:
        raise CostError("MAIN_COST_PROOF_MISMATCH") from error
    if raw != canonical_bytes(result) or result != build_witness(base):
        raise CostError("MAIN_COST_PROOF_MISMATCH")
    return result


def validate_base(raw: bytes, expected: BaseCostInputsV3) -> BaseCostInputsV3:
    try:
        result = BaseCostInputsV3.model_validate_json(raw)
    except ValidationError as error:
        raise CostError("MAIN_COST_PROOF_MISMATCH") from error
    _validate_base(result)
    if raw != canonical_bytes(result) or result != expected:
        raise CostError("MAIN_COST_PROOF_MISMATCH")
    return result


def validate_complete(raw: bytes, expected: CompleteCostInputsV3) -> CompleteCostInputsV3:
    try:
        result = CompleteCostInputsV3.model_validate_json(raw)
    except ValidationError as error:
        raise CostError("MAIN_COST_PROOF_MISMATCH") from error
    if raw != canonical_bytes(result) or result != freeze_complete(expected.base, expected.final_order):
        raise CostError("MAIN_COST_PROOF_MISMATCH")
    return result


def validate_proof(raw: bytes, complete: CompleteCostInputsV3, package_core_hash: str) -> CostProofV3:
    try:
        result = CostProofV3.model_validate_json(raw)
    except ValidationError as error:
        raise CostError("MAIN_COST_PROOF_MISMATCH") from error
    expected = build_proof(complete, build_witness(complete.base), package_core_hash)
    if raw != canonical_bytes(result) or result != expected:
        raise CostError("MAIN_COST_PROOF_MISMATCH")
    return result
