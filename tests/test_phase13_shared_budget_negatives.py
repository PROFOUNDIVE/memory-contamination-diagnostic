from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from .phase13_budget_fixtures import budget_api, fixture, operation, proof, save, signed


@pytest.mark.parametrize("mutation", ["unknown_lane", "missing_operation", "zero_unknown", "unbounded", "expiry"])
def test_prior_unknown_or_invalid_pricing_refuses(tmp_path: Path, mutation: str) -> None:
    api = budget_api()
    binding = fixture(tmp_path)
    binding_raw = json.loads(binding.read_bytes())
    reconciliation_path = tmp_path / "reconciliation.json"
    raw = json.loads(reconciliation_path.read_bytes())
    if mutation == "unknown_lane":
        raw["lanes"][0]["funding"] = "unresolved"
    elif mutation == "missing_operation":
        raw["operations"] = []
    elif mutation == "zero_unknown":
        raw["operations"][0].update(state="ambiguous", hold_usd_micros=0)
        raw["lanes"][0]["unknown_hold_usd_micros"] = 0
    elif mutation == "unbounded":
        raw["lanes"][0]["unknown_hold_usd_micros"] = None
    else:
        pricing_path = tmp_path / "pricing.json"
        pricing = json.loads(pricing_path.read_bytes())
        pricing["entries"][0]["valid_until"] = "2000-01-01"
        binding_raw["pricing"] = save(pricing_path, pricing)
    binding_raw["reconciliation"] = save(reconciliation_path, raw)
    save(binding, binding_raw)

    with pytest.raises(ValueError):
        api.SharedProviderBudget.create(binding).reserve(api.BudgetOperation(**operation()))


def test_reopen_rejects_tampered_hold_even_with_consistent_operation(tmp_path: Path) -> None:
    api = budget_api()
    binding = fixture(tmp_path)
    budget = api.SharedProviderBudget.create(binding)
    budget.reserve(api.BudgetOperation(**operation()))
    with sqlite3.connect(tmp_path / "aggregate.sqlite3") as connection:
        raw = operation(bound=1)
        connection.execute("UPDATE reservations SET operation=?,held=1", (json.dumps(raw),))

    with pytest.raises(ValueError):
        api.SharedProviderBudget.open(binding)


def test_actual_over_bound_charge_is_preserved_and_blocks_next_request(tmp_path: Path) -> None:
    api = budget_api()
    binding = fixture(tmp_path)
    budget = api.SharedProviderBudget.create(binding)
    intent = budget.reserve(api.BudgetOperation(**operation()))
    budget.mark_intent(intent)

    budget.reconcile(api.FilePin(**proof(tmp_path, budget, intent, amount=50_000_000)))

    status = api.SharedProviderBudget.open(binding).status()
    assert status.known_usd_micros == 55_000_000
    assert status.paid_eligible is False
    with pytest.raises(ValueError):
        budget.reserve(api.BudgetOperation(**operation(request="new")))


def test_charge_cannot_be_released_by_later_no_charge_proof(tmp_path: Path) -> None:
    api = budget_api()
    budget = api.SharedProviderBudget.create(fixture(tmp_path))
    intent = budget.reserve(api.BudgetOperation(**operation()))
    budget.mark_intent(intent)
    pointer = proof(tmp_path, budget, intent, kind="ambiguous", amount=3_000_000)
    budget.reconcile(api.FilePin(**pointer))
    raw = json.loads(Path(pointer["path"]).read_bytes())
    raw.update(kind="no_charge", realized_usd_micros=0)
    raw.pop("signature")
    second = save(tmp_path / "second-proof.json", signed(raw))

    with pytest.raises(ValueError, match="KNOWN_CHARGE"):
        budget.reconcile(api.FilePin(**second))


def test_missing_authority_is_not_silently_recreated(tmp_path: Path) -> None:
    api = budget_api()
    binding = fixture(tmp_path)
    budget = api.SharedProviderBudget.create(binding)
    (tmp_path / "aggregate.sqlite3").unlink()

    with pytest.raises(ValueError, match="MISSING"):
        budget.reserve(api.BudgetOperation(**operation()))

    assert not (tmp_path / "aggregate.sqlite3").exists()


def test_missing_ledger_cannot_reset_existing_authorization(tmp_path: Path) -> None:
    api = budget_api()
    binding = fixture(tmp_path)
    budget = api.SharedProviderBudget.create(binding)
    budget.reserve(api.BudgetOperation(**operation()))
    (tmp_path / "aggregate.sqlite3").unlink()

    with pytest.raises(ValueError):
        api.SharedProviderBudget.create(binding)

    assert not (tmp_path / "aggregate.sqlite3").exists()


def test_expired_only_prices_do_not_report_paid_eligibility(tmp_path: Path) -> None:
    api = budget_api()
    binding = fixture(tmp_path)
    pricing_path = tmp_path / "pricing.json"
    pricing = json.loads(pricing_path.read_bytes())
    for row in pricing["entries"]:
        row["valid_until"] = "2000-01-01"
    raw = json.loads(binding.read_bytes())
    raw["pricing"] = save(pricing_path, pricing)
    save(binding, raw)
    api.SharedProviderBudget.create(binding)

    assert api.verify(binding).paid_eligible is False


def test_self_asserted_owner_without_signature_cannot_release(tmp_path: Path) -> None:
    api = budget_api()
    budget = api.SharedProviderBudget.create(fixture(tmp_path))
    intent = budget.reserve(api.BudgetOperation(**operation()))
    pointer = proof(tmp_path, budget, intent, kind="no_request", amount=0)
    raw = json.loads(Path(pointer["path"]).read_bytes())
    raw.pop("signature", None)
    forged = save(tmp_path / "unsigned.json", raw)

    with pytest.raises(ValueError):
        budget.reconcile(api.FilePin(**forged))

    assert budget.status().held_usd_micros == 27_000_000


def test_crash_after_possible_egress_retains_hold_and_refuses_redispatch(tmp_path: Path) -> None:
    api = budget_api()
    binding = fixture(tmp_path)
    api.SharedProviderBudget.create(binding)
    child = subprocess.run([sys.executable, "-B", "-m", "tests.phase13_recovery_offline",
        str(Path(__file__).with_name("phase13_budget_fixtures.py"))], input="go\n",
        env={**os.environ, "BUDGET_BINDING": str(binding), "BUDGET_CRASH": "after_intent",
             "BUDGET_OPERATION": json.dumps(operation())}, capture_output=True, text=True, timeout=30)

    assert child.returncode == 73, child.stderr
    from memcontam.readiness.phase13_live_diagnostic_models import DiagnosticRequestIntent
    intent = DiagnosticRequestIntent.model_validate_json(child.stdout.splitlines()[1])
    budget = api.SharedProviderBudget.open(binding)
    with pytest.raises(ValueError, match="REPLAY"):
        budget.mark_intent(intent)
    assert budget.status().held_usd_micros == 27_000_000
    save(tmp_path / "crash-results.json", {"exit": child.returncode, "stdout": child.stdout,
        "stderr": child.stderr, "status": budget.status().model_dump(mode="json")})


@pytest.mark.parametrize("kind", ["charge", "ambiguous"])
def test_breached_bound_blocks_even_below_aggregate_cap(tmp_path: Path, kind: str) -> None:
    api = budget_api()
    budget = api.SharedProviderBudget.create(fixture(tmp_path))
    intent = budget.reserve(api.BudgetOperation(**operation()))
    budget.mark_intent(intent)

    budget.reconcile(api.FilePin(**proof(tmp_path, budget, intent, kind=kind, amount=30_000_000)))

    status = budget.status()
    assert status.known_usd_micros == 35_000_000
    assert status.paid_eligible is False
    if kind == "ambiguous":
        assert status.held_usd_micros is None
        assert status.available_usd_micros is None


def test_terminal_release_refuses_later_signed_proof(tmp_path: Path) -> None:
    api = budget_api()
    budget = api.SharedProviderBudget.create(fixture(tmp_path))
    intent = budget.reserve(api.BudgetOperation(**operation()))
    pointer = proof(tmp_path, budget, intent, kind="no_request", amount=0)
    budget.reconcile(api.FilePin(**pointer))
    raw = json.loads(Path(pointer["path"]).read_bytes())
    raw.update(kind="ambiguous", realized_usd_micros=None)
    raw.pop("signature")
    second = save(tmp_path / "second-proof.json", signed(raw))

    with pytest.raises(ValueError, match="PROOF_REPLAY"):
        budget.reconcile(api.FilePin(**second))


def test_exclusive_marker_allows_only_one_concurrent_creator(tmp_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    api = budget_api()
    binding = fixture(tmp_path)
    barrier = Barrier(2)

    def create_once():
        barrier.wait(timeout=10)
        try:
            api.SharedProviderBudget.create(binding)
        except ValueError:
            return "refused"
        return "created"

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(create_once) for _ in range(2)]
        results = [future.result(timeout=15) for future in futures]

    assert sorted(results) == ["created", "refused"]
    assert api.SharedProviderBudget.open(binding).status().reservations == 0
