from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from .phase13_budget_fixtures import budget_api, fixture, operation, pin, proof, save, signed
from .test_phase13_snapshot_bot import capacity_fixture as capacity_fixture


def test_aggregate_cap_includes_prior_spend_and_unknown_holds(tmp_path: Path) -> None:
    api = budget_api()
    binding = fixture(tmp_path)
    budget = api.SharedProviderBudget.create(binding)
    budget.reserve(api.BudgetOperation(**operation()))

    with pytest.raises(ValueError, match="CAP"):
        budget.reserve(api.BudgetOperation(**operation(request="second", lane="M")))

    status = api.SharedProviderBudget.open(binding).status()
    assert (status.known_usd_micros, status.held_usd_micros, status.available_usd_micros) == (
        5_000_000, 27_000_000, 13_000_000)


def test_remaining_run_allowance_is_not_reset_to_original_cap(tmp_path: Path) -> None:
    api = budget_api()
    budget = api.SharedProviderBudget.create(fixture(tmp_path, allowance=20_000_000))

    with pytest.raises(ValueError, match="CAP"):
        budget.reserve(api.BudgetOperation(**operation()))

    assert budget.status().held_usd_micros == 2_000_000


def test_incomplete_reconciliation_never_creates_authority(tmp_path: Path) -> None:
    api = budget_api()
    binding = fixture(tmp_path, complete=False)

    with pytest.raises(ValueError, match="RECONCILIATION"):
        api.SharedProviderBudget.create(binding)

    assert not (tmp_path / "aggregate.sqlite3").exists()
    assert api.verify(binding).paid_eligible is False


@pytest.mark.parametrize("field,value", [("currency", "KRW"), ("cap_usd_micros", 45_000_001),
    ("cap_usd_micros", 45.0), ("cap_usd_micros", True), ("account_id", "foreign"),
    ("ledger_path", "/tmp/alternate-authority.sqlite3"), ("lanes", ["D"])])
def test_schema_scope_and_reset_refuse(tmp_path: Path, field: str, value) -> None:
    api = budget_api()
    binding = fixture(tmp_path)
    raw = json.loads(binding.read_bytes())
    raw[field] = value
    save(binding, raw)

    with pytest.raises(ValueError):
        api.SharedProviderBudget.create(binding)

    assert not (tmp_path / "aggregate.sqlite3").exists()


@pytest.mark.parametrize("field,value", [("bound_usd_micros", 0), ("bound_usd_micros", None),
    ("bound_usd_micros", 1.0), ("bound_usd_micros", True), ("bound_usd_micros", 24_999_999),
    ("price_id", "unknown"), ("endpoint", "wrong"), ("model", "wrong"), ("lane_id", "foreign"),
    ("run_id", "old-run")])
def test_unbounded_underpriced_or_foreign_operation_refuses(tmp_path: Path, field: str, value) -> None:
    api = budget_api()
    budget = api.SharedProviderBudget.create(fixture(tmp_path))
    raw = {**operation(), field: value}

    with pytest.raises(ValueError):
        budget.reserve(api.BudgetOperation(**raw))

    assert budget.status().held_usd_micros == 2_000_000


def test_replay_refuses_even_after_proven_release(tmp_path: Path) -> None:
    api = budget_api()
    budget = api.SharedProviderBudget.create(fixture(tmp_path))
    intent = budget.reserve(api.BudgetOperation(**operation()))
    budget.reconcile(api.FilePin(**proof(tmp_path, budget, intent, kind="no_request", amount=0)))

    with pytest.raises(ValueError, match="REPLAY"):
        budget.reserve(api.BudgetOperation(**operation()))

    assert budget.status().available_usd_micros == 38_000_000


@pytest.mark.parametrize("kind,amount,known,held", [("charge", 3_000_000, 8_000_000, 2_000_000),
    ("ambiguous", None, 5_000_000, 27_000_000), ("ambiguous", 3_000_000, 8_000_000, 24_000_000),
    ("no_charge", 0, 5_000_000, 2_000_000)])
def test_reconciliation_preserves_known_and_ambiguous_liability(tmp_path: Path, kind, amount, known, held) -> None:
    api = budget_api()
    binding = fixture(tmp_path)
    budget = api.SharedProviderBudget.create(binding)
    intent = budget.reserve(api.BudgetOperation(**operation()))
    budget.mark_intent(intent)

    budget.reconcile(api.FilePin(**proof(tmp_path, budget, intent, kind=kind, amount=amount)))

    status = api.SharedProviderBudget.open(binding).status()
    assert (status.known_usd_micros, status.held_usd_micros) == (known, held)
    with pytest.raises(ValueError):
        budget.mark_intent(intent)


@pytest.mark.parametrize("mutation", ["absent", "foreign_intent", "issuer", "missing_evidence", "zero_charge"])
def test_release_requires_exact_attributable_proof(tmp_path: Path, mutation: str) -> None:
    api = budget_api()
    budget = api.SharedProviderBudget.create(fixture(tmp_path))
    intent = budget.reserve(api.BudgetOperation(**operation()))
    pointer = proof(tmp_path, budget, intent, kind="no_request", amount=0)
    path = Path(pointer["path"])
    raw = json.loads(path.read_bytes())
    if mutation == "absent":
        path.unlink()
    else:
        if mutation == "foreign_intent":
            raw["intent"]["request_sha256"] = "f" * 64
        elif mutation == "issuer":
            raw["issuer"] = "unrecognized"
        elif mutation == "missing_evidence":
            raw["evidence"] = []
        else:
            raw.update(kind="charge", realized_usd_micros=0)
        pointer = save(path, raw)

    with pytest.raises((ValueError, OSError)):
        budget.reconcile(api.FilePin(**pointer))

    assert budget.status().held_usd_micros == 27_000_000


@pytest.mark.parametrize("valid", [False, True])
def test_unknown_count_requires_current_exact_decision(tmp_path: Path, valid: bool) -> None:
    api = budget_api()
    budget = api.SharedProviderBudget.create(fixture(tmp_path))
    raw = operation(kind="count")
    decision = {"schema_version": "phase13_unknown_count_decision_v1", "issuer": "qa-accounting-owner",
                "budget_binding_sha256": budget.binding.sha256(), "run_id": "qa-run" if valid else "old-run",
                "request_id": raw["request_id"], "attempt_id": raw["attempt_id"],
                "request_sha256": raw["request_sha256"], "bound_usd_micros": 25_000_000,
                "price_id": "qa-count", "evidence": [pin(tmp_path / "qa-evidence.json")]}
    raw["unknown_count_decision"] = api.FilePin(**save(tmp_path / "decision.json", signed(decision)))

    if valid:
        budget.reserve(api.BudgetOperation(**raw))
        assert budget.status().held_usd_micros == 27_000_000
    else:
        with pytest.raises(ValueError, match="COUNT"):
            budget.reserve(api.BudgetOperation(**raw))


def test_unknown_count_without_decision_refuses(tmp_path: Path) -> None:
    api = budget_api()
    budget = api.SharedProviderBudget.create(fixture(tmp_path))

    with pytest.raises(ValueError, match="COUNT"):
        budget.reserve(api.BudgetOperation(**operation(kind="count")))


@pytest.mark.parametrize("target", ["qa-evidence.json", "pricing.json", "reconciliation.json", "binding.json"])
def test_stale_authority_refuses_after_reopen(tmp_path: Path, target: str) -> None:
    api = budget_api()
    binding = fixture(tmp_path)
    budget = api.SharedProviderBudget.create(binding)
    path = tmp_path / target
    if target == "binding.json":
        raw = json.loads(path.read_bytes())
        raw["run_id"] = "reset-run"
        save(path, raw)
    else:
        path.write_bytes(path.read_bytes() + b" ")

    with pytest.raises(ValueError):
        api.SharedProviderBudget.open(binding).reserve(api.BudgetOperation(**operation()))

    if target == "binding.json":
        with pytest.raises(ValueError):
            budget.reserve(api.BudgetOperation(**operation()))


def test_two_process_reservation_race_and_crash_hold(tmp_path: Path) -> None:
    api = budget_api()
    binding = fixture(tmp_path)
    api.SharedProviderBudget.create(binding)
    children = []
    results = []
    try:
        for index in range(2):
            children.append(subprocess.Popen([sys.executable, "-B", "-m", "tests.phase13_recovery_offline",
                str(Path(__file__).with_name("phase13_budget_fixtures.py"))],
                env={**os.environ, "BUDGET_BINDING": str(binding), "BUDGET_CRASH": "1",
                     "BUDGET_OPERATION": json.dumps(operation(request=f"race-{index}", lane=("D", "M")[index]))},
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
        for child in children:
            assert child.stdout is not None
            assert child.stdout.readline().strip() == "READY"
        for child in children:
            assert child.stdin is not None
            child.stdin.write("go\n")
            child.stdin.flush()
        for child in children:
            stdout, stderr = child.communicate(timeout=30)
            results.append({"pid": child.pid, "exit": child.returncode, "stdout": stdout, "stderr": stderr})
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=10)

    save(tmp_path / "race-results.json", results)
    assert sorted(row["exit"] for row in results) == [3, 73], results
    status = api.SharedProviderBudget.open(binding).status()
    assert (status.reservations, status.held_usd_micros) == (1, 27_000_000)


def test_budget_claim_binds_existing_diagnostic_intent(tmp_path: Path) -> None:
    import hashlib
    from memcontam.readiness.phase13_live_diagnostic import LiveDiagnosticLedger
    from .phase13_live_diagnostic_fixtures import bundle

    api = budget_api()
    diagnostic_root = tmp_path / "diagnostic"
    diagnostic_root.mkdir()
    diagnostic_binding, units, _ = bundle(diagnostic_root)
    ledger = LiveDiagnosticLedger.create(diagnostic_binding, units)
    ledger.claim(units[0].unit_id)
    binding_path = fixture(tmp_path / "budget", prior=0, hold=0, allowance=45_000_000)
    raw = json.loads(binding_path.read_bytes())
    raw["allowed_runs"] = [diagnostic_binding.observations[0].run_id]
    save(binding_path, raw)
    budget = api.SharedProviderBudget.create(binding_path)
    request = diagnostic_binding.observations[0].requests[0].request_id
    raw_operation = {**operation(request=request), "run_id": diagnostic_binding.observations[0].run_id,
        "unit_id": units[0].unit_id, "execution_binding_sha256": diagnostic_binding.sha256(),
        "attempt_id": hashlib.sha256((diagnostic_binding.sha256() + request).encode()).hexdigest()}
    intent = budget.reserve(api.BudgetOperation(**raw_operation))

    budget.claim_diagnostic(ledger, intent)

    assert ledger.status()["unresolved_requests"] == 1
    assert api.SharedProviderBudget.open(binding_path).status().held_usd_micros == 25_000_000
    with pytest.raises(ValueError, match="REPLAY"):
        budget.claim_diagnostic(ledger, intent)
