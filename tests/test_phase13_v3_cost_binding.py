from __future__ import annotations
from .phase13_corrective_identity import corrective_identity
from .phase13_count_fake import CountedProvider, fake_count_pricing

import hashlib
import importlib
import importlib.util
import json
from pathlib import Path

import pytest

from memcontam.clients.base import LLMResponse
from memcontam.readiness.phase13_authority_files import load_authority_v3
from memcontam.readiness.phase13_main_request_dispatch import _response_cost, _realized
from memcontam.readiness.phase13_main_production import ProductionObject
from memcontam.readiness.phase13_v3_request import MessageV3
from memcontam.readiness.phase13_v3_cost import activate_policy, freeze_base, build_witness
from memcontam.readiness.phase13_v3_cost_models import (
    CostError, CostUnit, FinalOrder, PrefreezeBindings, StageOccurrences,
    canonical_bytes, digest, seal,
)


MODULE = "memcontam.readiness.phase13_v3_cost_binding"


@pytest.fixture
def api():
    assert importlib.util.find_spec(MODULE), "V3 package-bound live cost seam is missing"
    return importlib.import_module(MODULE)


@pytest.fixture(scope="session")
def base():
    authority = load_authority_v3(Path(
        "/home/hyunwoo/gdrive_undergrad_research/PeerJ fast-track/References/Theoretical Artifacts"
    ), identity=corrective_identity())
    bindings = PrefreezeBindings(**{
        name: hashlib.sha256(name.encode()).hexdigest() for name in PrefreezeBindings.model_fields
    })
    units = tuple(CostUnit(unit_id=name * 64, stages=(StageOccurrences(
        stage_id="RAG_generation", calls=1, cache_write_tokens=378,
    ),)) for name in ("a", "b", "c"))
    return freeze_base(activate_policy(authority), bindings, units, count_pricing=fake_count_pricing(3, "0.001"))


@pytest.fixture
def bound(api, base):
    phase4 = api.MRP4Costs(policy=base.policy, base=base, witness=build_witness(base))
    package = api.CostBoundPackageV3(package_id=base.policy.authority.identity.package_id, final_order=FinalOrder(
        unit_ids=("c" * 64, "a" * 64, "b" * 64), runtime_hash="1" * 64,
        request_hash="2" * 64, tokenizer_hash="3" * 64,
    ))
    return api.bind_package_costs(package, phase4)


@pytest.fixture
def dispatch_fake(api, tmp_path, monkeypatch):
    from memcontam.readiness import phase13_main_request_dispatch as dispatch
    from memcontam.readiness.phase13_v3_terminal_ledger import TerminalLedgerV3

    counts = {"constructors": 0, "requests": 0}

    class Provider(CountedProvider):
        def send_compiled_v3(self, compiled, before_request):
            before_request()
            counts["requests"] += 1
            return LLMResponse("answer", {"status": "completed", "model": "gpt-5.6-luna", "service_tier": "default", "usage": {"input_tokens": 0, "output_tokens": 0}}, {}, 0)

    def factory(binding):
        counts["constructors"] += 1
        return Provider()

    def execute(package, resources, drift=None):
        parent_id = package.final_order.unit_ids[0]
        binding = dispatch.PackageBindingV3(package_sha256=("f" * 64 if drift == "raw_hash" else digest(package)),
                                             identity=(corrective_identity("foreign") if drift == "identity"
                                                       else resources.phase4.policy.authority.identity),
                                             authorization_sha256="b" * 64)
        key = dispatch.RequestKeyV3(parent_id=parent_id, stage="rag_generate", ordinal=0)
        ledger = TerminalLedgerV3.create(tmp_path / "request.sqlite3", {
            "schema_version": "phase13_main_run_ledger_v3", "unit_ids": [key.dispatch_id],
            "identity": binding.identity.model_dump(mode="json"),
            "package_sha256": binding.package_sha256, "authorization_sha256": binding.authorization_sha256,
        })
        dispatcher = dispatch.ProductionRequestDispatcherV3(ledger, binding, (
            dispatch.ParentTrajectoryV3(parent_id=parent_id, kind="NO_MEMORY_SINGLETON"),
        ), provider_factory=factory)
        live = dispatch.CostBoundRequestDispatcherV3(dispatcher, api.LiveCosts(package, resources),
                                                     "f" * 64 if drift == "package_hash" else package.package_hash)
        return live.dispatch(key, lambda: dispatch.RequestMaterialV3(
            messages=(MessageV3(role="user", content="fixture"),), native_state=b"state",
        ), lambda response: response.content)

    monkeypatch.setattr(dispatch, "count_prompt_tokens", lambda *_args: 1)
    return execute, counts


def test_mr_p4_rejects_complete_inputs_and_contains_no_table(api, bound):
    payload = bound.resources.phase4.model_dump(mode="json")
    assert set(payload) == {"policy", "base", "witness"}
    assert not {"final_order", "projected_krw"} & payload["witness"].keys()
    with pytest.raises(ValueError):
        api.MRP4Costs.model_validate({**payload, "complete": bound.resources.complete})


def test_hash_chain_uses_exact_projections(api, bound):
    package, resources = bound.package, bound.resources
    assert resources.complete.final_order_hash == digest(package.final_order)
    assert package.complete_inputs_hash == resources.complete.complete_inputs_hash
    assert resources.proof.package_core_hash == api.package_core_hash(package)
    assert package.cost_proof_hash == resources.proof.proof_hash
    assert package.package_hash == digest(package, "package_hash")
    assert package.package_hash != package.package_core_hash


def test_production_live_projection_uses_ordered_bound_table(api, bound):
    from memcontam.readiness.phase13_main_production import attribute_v3_projected_cost

    units = tuple(ProductionObject(index, name * 64, "NO_MEMORY_SINGLETON", 0,
                                  "game24", None, "NOT_APPLICABLE", None, 999)
                  for index, name in enumerate(("c", "a", "b")))
    costs = api.LiveCosts(bound.package, bound.resources)
    actual = attribute_v3_projected_cost(units, costs, bound.package.package_hash)
    assert [unit.projected_cost_krw for unit in actual] == [4, 3, 2]
    assert sum(unit.projected_cost_krw for unit in actual) == 9


@pytest.mark.parametrize("field", ["runtime_hash", "request_hash", "tokenizer_hash"])
def test_final_execution_identity_changes_every_successor(api, bound, field):
    order = bound.package.final_order.model_copy(update={field: "9" * 64})
    changed = api.bind_package_costs(bound.package.model_copy(update={"final_order": order}),
                                     bound.resources.phase4)
    assert changed.resources.complete.complete_inputs_hash != bound.resources.complete.complete_inputs_hash
    assert changed.package.package_core_hash != bound.package.package_core_hash
    assert changed.resources.proof.proof_hash != bound.resources.proof.proof_hash
    assert changed.package.package_hash != bound.package.package_hash


def test_live_rejects_self_rehashed_table_from_other_package(api, bound, dispatch_fake):
    other_order = bound.package.final_order.model_copy(update={"unit_ids": ("a" * 64, "b" * 64, "c" * 64)})
    other = api.bind_package_costs(bound.package.model_copy(update={"final_order": other_order}),
                                  bound.resources.phase4)
    foreign_rows = {row.unit_id: row for row in other.resources.proof.projected_krw}
    forged_proof = seal(bound.resources.proof.model_copy(update={
        "projected_krw": tuple(foreign_rows[name] for name in bound.package.final_order.unit_ids),
    }))
    package = bound.package.model_copy(update={"cost_proof_hash": forged_proof.proof_hash})
    package = package.model_copy(update={"package_hash": digest(package, "package_hash")})
    forged = bound.resources.model_copy(update={"proof": forged_proof})
    execute, counts = dispatch_fake
    with pytest.raises(CostError, match="MAIN_COST_PROOF_MISMATCH"):
        execute(package, forged)
    assert counts == {"constructors": 0, "requests": 0}
    assert package.package_hash == digest(package, "package_hash")
    assert forged_proof.proof_hash == digest(forged_proof, "proof_hash")


def test_cost_bound_dispatch_reaches_one_fake_request(api, bound, dispatch_fake):
    base = bound.resources.phase4.base
    unit = base.units[0].model_copy(update={"unit_id": "a" * 64})
    base = freeze_base(base.policy, base.bindings, (unit,), count_pricing=fake_count_pricing(1, "0.001"))
    phase4 = api.MRP4Costs(policy=base.policy, base=base, witness=build_witness(base))
    order = bound.package.final_order.model_copy(update={"unit_ids": ("a" * 64,)})
    package = api.bind_package_costs(bound.package.model_copy(update={"final_order": order}), phase4)
    execute, counts = dispatch_fake
    assert execute(package.package, package.resources) == "answer"
    assert counts == {"constructors": 1, "requests": 1}


@pytest.mark.parametrize("package_hash,proof_hash,unit_id", [
    ("f" * 64, None, "c" * 64), (None, "f" * 64, "c" * 64), (None, None, "missing"),
])
def test_live_requires_exact_package_proof_and_unit(api, bound, package_hash, proof_hash, unit_id):
    costs = api.LiveCosts(bound.package, bound.resources)
    with pytest.raises(CostError, match="MAIN_COST_PROOF_MISMATCH"):
        costs.projected(package_hash or bound.package.package_hash, api.TableKey(
            proof_hash=proof_hash or bound.resources.proof.proof_hash, unit_id=unit_id))


def test_package_core_includes_added_execution_fields(api, bound):
    class ExtendedPackage(api.CostBoundPackageV3):
        governed_tree_hash: str

    extended = ExtendedPackage(**bound.package.model_dump(), governed_tree_hash="a" * 64)
    changed = extended.model_copy(update={"governed_tree_hash": "b" * 64})
    assert api.package_core_hash(extended) != api.package_core_hash(changed)


@pytest.mark.parametrize("field", ["live_contract_hash", "generated_closure_hash"])
def test_successor_references_do_not_enter_package_core(api, bound, field):
    changed = bound.package.model_copy(update={field: "f" * 64})
    assert api.package_core_hash(changed) == bound.package.package_core_hash
    assert digest(changed, "package_hash") != bound.package.package_hash


def test_live_rejects_self_rehashed_complete_input_change(api, bound):
    order = bound.resources.complete.final_order.model_copy(update={"runtime_hash": "f" * 64})
    complete = seal(bound.resources.complete.model_copy(update={
        "final_order": order, "final_order_hash": digest(order),
    }))
    with pytest.raises(CostError, match="MAIN_COST_PROOF_MISMATCH"):
        api.LiveCosts(bound.package, bound.resources.model_copy(update={"complete": complete}))


def test_phase4_rejects_self_rehashed_witness(api, bound):
    witness = bound.resources.phase4.witness
    totals = witness.totals.model_copy(update={"cmax_main_krw": 0})
    forged = seal(witness.model_copy(update={"totals": totals}))
    with pytest.raises(ValueError, match="MAIN_COST_PROOF_MISMATCH"):
        api.MRP4Costs(policy=bound.resources.phase4.policy, base=bound.resources.phase4.base,
                     witness=forged)


def test_live_rate_drift_requires_reapproval(api, bound):
    phase4 = bound.resources.phase4
    policy = seal(phase4.policy.model_copy(update={"currency": "KRW"}))
    resources = bound.resources.model_copy(update={
        "phase4": phase4.model_copy(update={"policy": policy}),
    })
    with pytest.raises(CostError, match="RATE_CARD_DRIFT_REQUIRES_REAPPROVAL"):
        api.LiveCosts(bound.package, resources)


@pytest.mark.parametrize("usage", [None, {"input_tokens": 1000, "output_tokens": 1000}])
def test_response_missing_currency_reaches_exact_rejection(usage):
    response = LLMResponse("", {"authoritative_provider_cost_usd": "0.001", "usage": usage}, {}, 0)
    evidence = _response_cost(response)
    assert evidence.currency is None
    with pytest.raises(CostError, match="MAIN_COST_CURRENCY_INVALID"):
        _realized(evidence)


def test_response_unknown_actual_remains_nullable():
    evidence = _response_cost(LLMResponse("", {}, {}, 0))
    assert evidence.monetary_cost is None
    assert evidence.currency is None
    assert _realized(evidence) is None


@pytest.mark.parametrize("money,expected", [("0.0014", 3), ("0.001", None)])
def test_response_provider_precedence_and_disagreement(money, expected):
    response = LLMResponse("", {"authoritative_provider_cost_usd": money, "currency": "USD",
                               "usage": {"input_tokens": 1000, "output_tokens": 1000}}, {}, 0)
    evidence = _response_cost(response)
    if expected is None:
        with pytest.raises(CostError, match="MAIN_COST_RECONCILIATION_REQUIRED"):
            _realized(evidence)
    else:
        assert _realized(evidence) == expected


def test_proof_canonical_bytes_are_preserved(api, bound):
    costs = api.LiveCosts(bound.package, bound.resources)
    assert canonical_bytes(costs.resources.proof) == canonical_bytes(bound.resources.proof)


@pytest.mark.parametrize("field", ["phase4", "complete", "proof"])
def test_live_cost_resources_require_every_phase_output(api, bound, field):
    payload = bound.resources.model_dump(mode="json")
    del payload[field]
    with pytest.raises(ValueError):
        api.CostResourcesV3.model_validate_json(json.dumps(payload))


def test_over_budget_witness_cannot_supply_live_costs(api, bound):
    base = bound.resources.phase4.base
    stage = base.units[0].stages[0].model_copy(update={"calls": 1_000_000})
    unit = base.units[0].model_copy(update={"stages": (stage,)})
    base = freeze_base(base.policy, base.bindings, (unit,), count_pricing=fake_count_pricing(1_000_000, "0.001"))
    phase4 = api.MRP4Costs(policy=base.policy, base=base, witness=build_witness(base))
    order = bound.package.final_order.model_copy(update={"unit_ids": (unit.unit_id,)})
    over = api.bind_package_costs(bound.package.model_copy(update={"final_order": order}), phase4)
    with pytest.raises(CostError, match="MAIN_COST_PROOF_MISMATCH"):
        api.LiveCosts(over.package, over.resources)


def test_production_projection_rejects_wrong_unit_order(api, bound):
    units = tuple(ProductionObject(index, name * 64, "NO_MEMORY_SINGLETON", 0,
                                  "game24", None, "NOT_APPLICABLE", None, 999)
                  for index, name in enumerate(("a", "b", "c")))
    with pytest.raises(CostError, match="MAIN_COST_PROOF_MISMATCH"):
        api.attribute_v3_projected_cost(units, api.LiveCosts(bound.package, bound.resources),
                                        bound.package.package_hash)


@pytest.mark.parametrize("drift", ["raw_hash", "package_hash", "identity"])
def test_dispatch_rejects_wrong_package_before_fake_constructor(bound, dispatch_fake, drift):
    execute, counts = dispatch_fake
    with pytest.raises(CostError, match="MAIN_COST_PROOF_MISMATCH"):
        execute(bound.package, bound.resources, drift)
    assert counts == {"constructors": 0, "requests": 0}


@pytest.mark.parametrize("field", ["base_inputs_hash", "witness_hash", "complete_inputs_hash", "cost_proof_hash"])
def test_rehashed_package_binding_fields_fail_semantically(api, bound, field):
    package = bound.package.model_copy(update={field: "f" * 64})
    package = package.model_copy(update={"package_hash": digest(package, "package_hash")})
    with pytest.raises(CostError, match="MAIN_COST_PROOF_MISMATCH"):
        api.LiveCosts(package, bound.resources)
