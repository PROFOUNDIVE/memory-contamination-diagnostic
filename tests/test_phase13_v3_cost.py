import hashlib
from .phase13_corrective_identity import corrective_identity
import importlib
import importlib.util
import json
from collections.abc import Iterator
from decimal import Decimal, localcontext
from pathlib import Path
from types import ModuleType
from typing import assert_never

import pytest
from pydantic import JsonValue

from memcontam.readiness.phase13_cost_policy import _ceil, load_cost_policy_bundle
from memcontam.readiness.phase13_authority_files import load_authority_v3


def test_baseline_historical_stage_costs() -> None:
    """Given historical inputs, when loaded, then preserve historical arithmetic."""
    bundle = load_cost_policy_bundle(Path(__file__).resolve().parents[1])
    assert [(row.input_krw_ceiling, row.output_krw_ceiling) for row in bundle.proof.stage_costs] == [
        (37507, 9880), (830, 5928), (4732, 7410), (7835, 9880), (10231, 7410),
        (18302, 19710), (26859, 14783), (37033, 9880), (54355, 158073), (1160, 2458),
    ]
    assert bundle.proof.cmax_main_krw == 444256


def test_baseline_decimal_ceiling() -> None:
    """Given exact decimal components, when ceiled, then never round downward."""
    assert [_ceil(Decimal(value)) for value in ("0", "1", "1.00000000000000000000000001")] == [0, 1, 2]


def test_exact_law_known_answer_before_implementation() -> None:
    assert importlib.util.find_spec("memcontam.readiness.phase13_v3_cost") is not None, "V3 exact law is missing"
    module = importlib.import_module("memcontam.readiness.phase13_v3_cost")
    request = module.RequestTokens(input_tokens=272001, output_tokens=1000, cache_write_tokens=272001)
    assert tuple(module.decimal_string(value) for value in module.exact_request_cost(request)) == ("217.6008", "2.88")


@pytest.fixture
def cost() -> ModuleType:
    assert importlib.util.find_spec("memcontam.readiness.phase13_v3_cost") is not None, "V3 two-phase cost law is missing"
    return importlib.import_module("memcontam.readiness.phase13_v3_cost")


@pytest.fixture(scope="session")
def authority():
    return load_authority_v3(Path("/home/hyunwoo/gdrive_undergrad_research/PeerJ fast-track/References/Theoretical Artifacts"), identity=corrective_identity())


@pytest.fixture
def base(cost, authority):
    policy = cost.activate_policy(authority)
    bindings = cost.PrefreezeBindings(**{name: hashlib.sha256(name.encode()).hexdigest() for name in cost.PrefreezeBindings.model_fields})
    units = tuple(cost.CostUnit(unit_id=name, stages=(cost.StageOccurrences(stage_id="RAG_generation", calls=1, cache_write_tokens=378),)) for name in ("a", "b", "c"))
    return cost.freeze_base(policy, bindings, units)


def _complete(cost, base):
    order = cost.FinalOrder(unit_ids=("c", "a", "b"), runtime_hash="1" * 64, request_hash="2" * 64, tokenizer_hash="3" * 64)
    return cost.freeze_complete(base, order)


@pytest.mark.parametrize("stage,tokens,calls,expected", [
    ("FH_generation", (9330, 512), 10050, ("37506.6", "9879.552", 37507, 9880)),
    ("RAG_generation", (378, 512), 6030, ("911.736", "5927.7312", 912, 5928)),
    ("BoT_problem_distillation", (1177, 384), 10050, ("4731.54", "7409.664", 4732, 7410)),
    ("BoT_solve", (1949, 512), 10050, ("7834.98", "9879.552", 7835, 9880)),
    ("BoT_thought_distillation", (2545, 384), 10050, ("10230.9", "7409.664", 10231, 7410)),
    ("Reflexion_actor_generation", (2282, 512), 20050, ("18301.64", "19709.952", 18302, 19710)),
    ("Reflexion_reflection", (3349, 384), 20050, ("26858.98", "14782.464", 26859, 14783)),
    ("DC_RS_generation", (9212, 512), 10050, ("37032.24", "9879.552", 37033, 9880)),
    ("DC_RS_writer_synthesis", (13521, 8192), 10050, ("54354.42", "158072.832", 54355, 158073)),
    ("NoMem_generation", (1160, 512), 2500, ("1160", "2457.6", 1160, 2458)),
])
def test_each_stage_known_answer(cost, authority, stage, tokens, calls, expected):
    policy = cost.activate_policy(authority)
    unit = cost.CostUnit(unit_id="one", stages=(cost.StageOccurrences(stage_id=stage, calls=calls, cache_write_tokens=tokens[0]),))
    bindings = cost.PrefreezeBindings(**{name: "a" * 64 for name in cost.PrefreezeBindings.model_fields})
    witness = cost.build_witness(cost.freeze_base(policy, bindings, (unit,)))
    row = witness.totals.stage_costs[0]
    assert (row.input_exact_krw, row.output_exact_krw, row.input_krw_ceiling, row.output_krw_ceiling) == expected
    assert witness.totals.cmax_main_krw == expected[2] + expected[3]
    assert witness.totals.semantic_calls == calls


@pytest.mark.parametrize("inputs,outputs,written,expected", [
    (272000, 1000, 0, ("87.04", "1.92")),
    (272001, 1000, 0, ("174.08064", "2.88")),
    (272001, 1000, 272001, ("217.6008", "2.88")),
    (1000, 1000, 500, ("0.36", "1.92")),
    (1000, 1000, 0, ("0.32", "1.92")),
])
def test_modifiers_exact_before_ceiling(cost, inputs, outputs, written, expected):
    request = cost.RequestTokens(input_tokens=inputs, output_tokens=outputs, cache_write_tokens=written)
    with localcontext() as context:
        context.prec = 2
        actual = cost.exact_request_cost(request)
    assert tuple(cost.decimal_string(value) for value in actual) == expected


def test_two_phase_boundaries_and_telescoping(cost, base):
    witness = cost.build_witness(base)
    complete = _complete(cost, base)
    proof = cost.build_proof(complete, witness, "4" * 64)
    assert [(row.unit_id, row.projected_krw) for row in proof.projected_krw] == [("c", 2), ("a", 1), ("b", 1)]
    assert proof.totals == witness.totals
    assert (proof.totals.cmax_main_krw, proof.totals.gate_margin_krw, proof.totals.gate_result) == (4, 449996, "PASS")
    assert not {"final_order", "projected_krw", "proof_id"} & witness.model_dump().keys()
    assert complete.base_inputs_hash == base.base_inputs_hash
    assert complete.final_order_hash == hashlib.sha256(cost.canonical_bytes(complete.final_order)).hexdigest()
    cost.validate_proof(cost.canonical_bytes(proof), complete, proof.package_core_hash)


@pytest.mark.parametrize("field,value", [("input_tokens", -1), ("input_tokens", True), ("input_tokens", "1"), ("input_tokens", 1.5), ("cache_write_tokens", 3)])
def test_malformed_numeric_input(cost, field, value):
    with pytest.raises(ValueError):
        cost.RequestTokens.model_validate({"input_tokens": 2, "output_tokens": 1, field: value})


@pytest.mark.parametrize("money,usage,currency,expected", [
    ("0.001", None, "USD", ("AUTHORITATIVE_PROVIDER", "0.001", 2)),
    (None, (1000, 1000, 0), None, ("DERIVED_FROM_PROVIDER_USAGE", "0.0014", 3)),
    (None, (1000, 1000, 1000), None, ("DERIVED_FROM_PROVIDER_USAGE", "0.00122", 2)),
    ("0.0014", (1000, 1000, 0), "USD", ("AUTHORITATIVE_PROVIDER", "0.0014", 3)),
    ("0.00062500000000000000000000000000001", None, "USD", ("AUTHORITATIVE_PROVIDER", "0.00062500000000000000000000000000001", 2)),
    ("0", None, "USD", ("AUTHORITATIVE_PROVIDER", "0", 0)),
])
def test_realized_precedence_and_exact_fx(cost, money, usage, currency, expected):
    evidence = cost.ProviderCostEvidence(monetary_cost=money, currency=currency, usage=None if usage is None else cost.ProviderUsage(input_tokens=usage[0], output_tokens=usage[1], cached_input_tokens=usage[2]))
    result = cost.reconcile_actual(evidence)
    assert (result.source, result.selected_usd, result.realized_krw) == expected


@pytest.mark.parametrize("money,currency,code", [("0.01", None, "MAIN_COST_CURRENCY_INVALID"), ("0.01", "KRW", "MAIN_COST_CURRENCY_INVALID"), (None, None, "MAIN_TERMINAL_COST_UNKNOWN"), ("NaN", "USD", "MAIN_COST_NUMERIC_INVALID"), ("-1", "USD", "MAIN_COST_NUMERIC_INVALID")])
def test_actual_unknown_and_currency_fail_closed(cost, money, currency, code):
    with pytest.raises(cost.CostError, match=code):
        cost.reconcile_actual(cost.ProviderCostEvidence(monetary_cost=money, currency=currency))


def test_provider_disagreement_preserves_both(cost):
    evidence = cost.ProviderCostEvidence(monetary_cost="0.001", currency="USD", usage=cost.ProviderUsage(input_tokens=1000, output_tokens=1000))
    with pytest.raises(cost.CostError, match="MAIN_COST_RECONCILIATION_REQUIRED") as caught:
        cost.reconcile_actual(evidence)
    assert caught.value.evidence == evidence
    assert caught.value.derived_usd == "0.0014"


def test_rehashed_semantic_mutations_fail_cost_validation(cost, base):
    complete = _complete(cost, base)
    witness = cost.build_witness(base)
    proof = cost.build_proof(complete, witness, "4" * 64)
    artifacts = [(base.policy, lambda raw: cost.validate_policy_bytes(raw, base.policy.authority)),
                 (base, lambda raw: cost.validate_base(raw, base)),
                 (complete, lambda raw: cost.validate_complete(raw, complete)),
                 (witness, lambda raw: cost.validate_witness(raw, base)),
                 (proof, lambda raw: cost.validate_proof(raw, complete, "4" * 64))]
    checked = 0
    for artifact, validate in artifacts:
        original = json.loads(cost.canonical_bytes(artifact))
        assert _json_bytes(_rehash(original)) == cost.canonical_bytes(artifact)
        for mutation in _mutations(original):
            encoded = _json_bytes(_rehash(mutation))
            if encoded == cost.canonical_bytes(artifact):
                continue
            with pytest.raises(cost.CostError, match="MAIN_COST_PROOF_MISMATCH|RATE_CARD_DRIFT_REQUIRES_REAPPROVAL"):
                validate(encoded)
            checked += 1
    assert checked >= 250


def _json_bytes(value: JsonValue) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _rehash(value: JsonValue) -> JsonValue:
    match value:
        case dict():
            result = {key: _rehash(item) for key, item in value.items()}
            nested_base = result.get("base")
            if isinstance(nested_base, dict):
                result["base_inputs_hash"] = nested_base["base_inputs_hash"]
                result["final_order_hash"] = hashlib.sha256(_json_bytes(result["final_order"])).hexdigest()
            field = next((name for name in ("proof_hash", "witness_hash", "complete_inputs_hash", "base_inputs_hash", "policy_hash") if name in result), None)
            if field is not None:
                result[field] = hashlib.sha256(_json_bytes({key: item for key, item in result.items() if key != field})).hexdigest()
            return result
        case list():
            return [_rehash(item) for item in value]
        case str() | int() | float() | None:
            return value
        case unreachable:
            assert_never(unreachable)


def _mutations(value: JsonValue) -> Iterator[JsonValue]:
    match value:
        case dict():
            for key, item in value.items():
                for changed in _mutations(item):
                    yield {**value, key: changed}
        case list():
            for index, item in enumerate(value):
                for changed in _mutations(item):
                    yield [*value[:index], changed, *value[index + 1:]]
        case bool():
            yield not value
        case int() | float():
            yield value + 1
        case str():
            yield "0" * 64 if len(value) == 64 else value + "9"
        case None:
            yield "unexpected"
        case unreachable:
            assert_never(unreachable)


@pytest.mark.parametrize("field,value", [("input_usd_per_million", "0.21"), ("cache_read_credit", "yes"), ("fx_planning_ceiling_krw_per_usd", 1601)])
def test_rehashed_rate_drift_requires_reapproval(cost, base, field, value):
    rate = base.policy.rate_card.model_copy(update={field: value})
    policy = cost.seal(base.policy.model_copy(update={"rate_card": rate}))
    with pytest.raises(cost.CostError, match="RATE_CARD_DRIFT_REQUIRES_REAPPROVAL"):
        cost.validate_policy(policy, base.policy.authority)


def test_complete_inputs_validate_against_frozen_predecessor(cost, base):
    complete = _complete(cost, base)
    assert cost.validate_base(cost.canonical_bytes(base), base) == base
    assert cost.validate_complete(cost.canonical_bytes(complete), complete) == complete


def test_raw_self_hash_cannot_be_healed_by_schema_defaults(cost, base):
    witness = cost.build_witness(base)
    payload = json.loads(cost.canonical_bytes(witness))
    del payload["role"]
    with pytest.raises(cost.CostError, match="MAIN_COST_PROOF_MISMATCH"):
        cost.validate_witness(_json_bytes(payload), base)


def test_reordered_units_change_attribution_not_stage_totals(cost, base):
    complete = _complete(cost, base)
    reordered = cost.freeze_complete(base, complete.final_order.model_copy(update={"unit_ids": ("a", "b", "c")}))
    proof = cost.build_proof(reordered, cost.build_witness(base), "4" * 64)
    assert [(row.unit_id, row.projected_krw) for row in proof.projected_krw] == [("a", 2), ("b", 1), ("c", 1)]
    assert reordered.complete_inputs_hash != complete.complete_inputs_hash
    with pytest.raises(cost.CostError, match="MAIN_COST_PROOF_MISMATCH"):
        cost.validate_proof(cost.canonical_bytes(proof), complete, "4" * 64)


def test_stage_occurrences_with_distinct_cache_plans(cost, base):
    groups = (cost.StageOccurrences(stage_id="RAG_generation", calls=1), cost.StageOccurrences(stage_id="RAG_generation", calls=1, cache_write_tokens=378))
    inputs = cost.freeze_base(base.policy, base.bindings, (cost.CostUnit(unit_id="synthetic", stages=groups),))
    witness = cost.build_witness(inputs)
    row = witness.totals.stage_costs[0]
    assert (row.semantic_calls, row.input_exact_krw, row.output_exact_krw, witness.totals.cmax_main_krw) == (2, "0.27216", "1.96608", 3)


def test_all_stage_final_ceiling_from_explicit_multiplicities(cost, base):
    counts = (10050, 6030, 10050, 10050, 10050, 20050, 20050, 10050, 10050, 2500)
    groups = tuple(cost.StageOccurrences(stage_id=stage.stage_id, calls=count, cache_write_tokens=stage.maximum_input_tokens) for stage, count in zip(base.policy.authority.registry.stages, counts, strict=True))
    inputs = cost.freeze_base(base.policy, base.bindings, (cost.CostUnit(unit_id="synthetic", stages=groups),))
    totals = cost.build_witness(inputs).totals
    assert (totals.semantic_calls, totals.cmax_main_krw, totals.gate_margin_krw) == (108930, 444338, 5662)


@pytest.mark.parametrize("count,expected", [(250000, (283560, "PASS")), (400000, (453696, "FAIL"))])
def test_gate_uses_core_not_total_budget(cost, base, count, expected):
    group = cost.StageOccurrences(stage_id="RAG_generation", calls=count, cache_write_tokens=378)
    inputs = cost.freeze_base(base.policy, base.bindings, (cost.CostUnit(unit_id="synthetic", stages=(group,)),))
    totals = cost.build_witness(inputs).totals
    assert (totals.cmax_main_krw, totals.gate_result) == expected


@pytest.mark.parametrize("change", ["rate", "scale", "order", "duplicate", "base", "witness"])
def test_stale_or_incomplete_inputs_rejected(cost, base, change):
    complete = _complete(cost, base)
    witness = cost.build_witness(base)
    with pytest.raises((ValueError, cost.CostError)):
        if change == "rate":
            cost.validate_policy(base.policy.model_copy(update={"model": "other"}), base.policy.authority)
        elif change == "scale":
            cost.ActivatedPolicyV3.model_validate({**base.policy.model_dump(), "decimal_scale": 2})
        elif change in {"order", "duplicate"}:
            cost.freeze_complete(base, complete.final_order.model_copy(update={"unit_ids": ("a", "a") if change == "duplicate" else ("a",)}))
        elif change == "base":
            cost.build_proof(complete.model_copy(update={"base_inputs_hash": "9" * 64}), witness, "4" * 64)
        else:
            cost.build_proof(complete, witness.model_copy(update={"base_inputs_hash": "9" * 64}), "4" * 64)
