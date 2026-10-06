from fractions import Fraction
from decimal import localcontext
from pathlib import Path

import pytest
from pydantic import ValidationError

from memcontam.readiness import phase13_v3_cost as cost
from memcontam.readiness.phase13_authority_files import load_authority_v3
from memcontam.readiness.phase13_v3_cost_binding import (
    CostBoundPackageV3, MRP4Costs, bind_package_costs, validate_package_costs,
)

from .phase13_corrective_identity import corrective_identity


@pytest.fixture(scope="module")
def base():
    authority = load_authority_v3(Path(
        "/home/hyunwoo/gdrive_undergrad_research/PeerJ fast-track/References/Theoretical Artifacts"
    ), identity=corrective_identity())
    bindings = cost.PrefreezeBindings(**{
        name: "a" * 64 for name in cost.PrefreezeBindings.model_fields
    })
    units = tuple(cost.CostUnit(unit_id=name, stages=(cost.StageOccurrences(
        stage_id="RAG_generation", calls=1, cache_write_tokens=378,
    ),)) for name in ("a", "b", "c"))
    return cost.freeze_base(cost.activate_policy(authority), bindings, units)


def pricing(rate="0.0004", operations=4):
    return cost.CountPricingV1(
        endpoint="https://count.invalid/v1/responses/input_tokens",
        deployment_sha256="1" * 64, billing_evidence_sha256="2" * 64,
        compatibility_evidence_sha256="3" * 64,
        maximum_usd_per_operation=rate, maximum_count_operations=operations,
    )


def priced_base(base, rate="0.0004", operations=4):
    return cost.freeze_base(base.policy, base.bindings, base.units,
        count_pricing=pricing(rate, operations), retry_reservations=(
            cost.RetryReservation(unit_id="a", dispatch_id="4" * 64,
                                  stage_id="RAG_generation", reservation_krw=2),
        ))


def bound_package(base):
    phase4 = MRP4Costs(policy=base.policy, base=base, witness=cost.build_witness(base))
    package = CostBoundPackageV3(
        package_id=base.policy.authority.identity.package_id,
        final_order=cost.FinalOrder(unit_ids=("c", "a", "b"), runtime_hash="5" * 64,
                                   request_hash="6" * 64, tokenizer_hash="7" * 64),
    )
    return bind_package_costs(package, phase4)


def test_unbound_count_pricing_cannot_build_current_witness(base):
    """Given no endpoint price, when building a witness, then fail closed."""
    with pytest.raises(cost.CostError, match="COUNT_PRICING_PROOF_UNBOUND"):
        cost.build_witness(base)


def test_count_reservation_includes_initial_and_entitled_retry_when_bound(base):
    """Given three initial calls and a retry, when bound, then reserve four counts."""
    inputs = priced_base(base)
    with localcontext() as context:
        context.prec = 2
        bound = bound_package(inputs)
    totals = bound.resources.proof.totals
    assert totals == bound.resources.phase4.witness.totals
    assert (totals.count_operations, totals.count_exact_krw, totals.count_krw_ceiling) == (4, "2.56", 3)
    assert (totals.retry_reserve_krw, totals.cmax_main_krw, totals.gate_margin_krw) == (3, 9, 449991)
    assert [(row.unit_id, row.projected_krw) for row in bound.resources.proof.projected_krw] == [
        ("c", 3), ("a", 5), ("b", 1),
    ]
    validate_package_costs(bound.package, bound.resources)


@pytest.mark.parametrize("rate", ["0", "0.0", "-1", "NaN", "1e-3", None, True])
def test_zero_or_unknown_count_rate_rejected_when_parsed(rate):
    """Given invalid external pricing, when parsed, then reject rather than free."""
    with pytest.raises((ValidationError, cost.CostError)):
        pricing(rate)


@pytest.mark.parametrize("operations", [0, 3, 5])
def test_count_operation_cap_rejected_when_not_exact_all_attempt_reservation(base, operations):
    """Given an incorrect count ceiling, when freezing, then reject the proof input."""
    with pytest.raises((ValidationError, cost.CostError)):
        priced_base(base, operations=operations)


@pytest.mark.parametrize("field", ["deployment_sha256", "billing_evidence_sha256", "compatibility_evidence_sha256"])
def test_count_evidence_must_be_bound_when_pricing_is_parsed(field):
    """Given an absent independent binding, when parsed, then reject pricing."""
    payload = pricing().model_dump()
    payload.pop(field)
    with pytest.raises(ValidationError):
        cost.CountPricingV1.model_validate(payload)


@pytest.mark.parametrize("field,value", [
    ("count_operations", 3), ("count_exact_krw", "0"), ("count_krw_ceiling", 0),
    ("cmax_main_krw", 6), ("retry_reserve_krw", 2),
])
def test_rehashed_count_proof_tamper_rejected_when_validated(base, field, value):
    """Given a resealed false projection, when recomputed, then reject it."""
    bound = bound_package(priced_base(base))
    proof = bound.resources.proof
    changed = cost.seal(proof.model_copy(update={
        "totals": proof.totals.model_copy(update={field: value}),
    }))
    with pytest.raises(cost.CostError, match="MAIN_COST_PROOF_MISMATCH"):
        cost.validate_proof(cost.canonical_bytes(changed), bound.resources.complete,
                            bound.package.package_core_hash)


def test_rehashed_count_rate_tamper_rejected_against_frozen_predecessor(base):
    """Given a changed rate with a fresh hash, when compared, then reject drift."""
    inputs = priced_base(base)
    changed = cost.seal(inputs.model_copy(update={"count_pricing": pricing("0.0005")}))
    with pytest.raises(cost.CostError, match="MAIN_COST_PROOF_MISMATCH"):
        cost.validate_base(cost.canonical_bytes(changed), inputs)


def test_count_cost_can_fail_gate_when_generation_alone_fits(base):
    """Given costly count operations, when proved, then use the 450k gate."""
    inputs = cost.freeze_base(base.policy, base.bindings, base.units,
                              count_pricing=pricing("100", 3))
    bound = bound_package(inputs)
    assert (bound.resources.proof.totals.cmax_main_krw,
            bound.resources.proof.totals.gate_result) == (480004, "FAIL")
    with pytest.raises(cost.CostError, match="MAIN_COST_PROOF_MISMATCH"):
        validate_package_costs(bound.package, bound.resources)


def test_retry_count_cost_cannot_escape_40000_reserve_when_generation_fits(base):
    """Given an excessive retry count price, when freezing, then reject reserve."""
    with pytest.raises(cost.CostError, match="MAIN_COST_PROOF_MISMATCH"):
        priced_base(base, rate="25")


def test_all_stages_count_once_when_calls_and_retry_slots_are_reserved(base):
    """Given the governed stage inventory, when proved, then count every request."""
    groups = tuple(cost.StageOccurrences(stage_id=stage.stage_id, calls=index + 1)
                   for index, stage in enumerate(base.policy.authority.registry.stages))
    units = (cost.CostUnit(unit_id="all", stages=groups),)
    inputs = cost.freeze_base(base.policy, base.bindings, units,
                              count_pricing=pricing("0.0000001", sum(row.calls for row in groups)))
    totals = cost.build_witness(inputs).totals
    assert totals.count_operations == totals.semantic_calls == 55
    assert Fraction(totals.count_exact_krw) == Fraction(55 * 1600, 10_000_000)
    assert totals.count_krw_ceiling == len(groups)
