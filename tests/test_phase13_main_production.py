from __future__ import annotations

import json
from pathlib import Path

from memcontam.readiness.phase13_main_execution_models import MainExecutionFreeze
from memcontam.readiness.phase13_main_live_dispatch import persist_reconciliation_evidence
from memcontam.readiness.phase13_main_runner import run_pending
from memcontam.readiness.phase13_main_runner_ledger import MainRunLedger
from memcontam.readiness.phase13_main_runner_models import (
    DispatchTechnicalFailure,
    MainRunBinding,
    enumerate_execution_units,
)


ROOT = Path(__file__).resolve().parents[1]
P5 = ROOT / "data/phase13/main/mr_p5/execution_package_v1.json"


def _candidate_package() -> MainExecutionFreeze:
    payload = json.loads(P5.read_text(encoding="utf-8"))
    payload["cost_guard"].update(
        semantic_calls=108930,
        cmax_main_krw=444256,
        margin_krw=5744,
    )
    payload["runtime"]["tools"] = ()
    return MainExecutionFreeze.model_validate(payload)


def test_production_order_contains_prefixes_consumers_and_nomem_once() -> None:
    units = enumerate_execution_units(_candidate_package())

    assert len(units) == 1200
    assert sum(unit.kind == "CLEAN_PREFIX" for unit in units) == 230
    assert sum(unit.kind == "MEMORY_BEARING" for unit in units) == 920
    assert sum(unit.kind == "NO_MEMORY_SINGLETON" for unit in units) == 50
    assert tuple(unit.sequence for unit in units) == tuple(range(1200))
    assert len({unit.unit_id for unit in units}) == 1200


def test_each_prefix_immediately_precedes_its_four_compatible_consumers() -> None:
    units = enumerate_execution_units(_candidate_package())

    for index, prefix in enumerate(units):
        if prefix.kind != "CLEAN_PREFIX":
            continue
        consumers = units[index + 1 : index + 5]
        assert len(consumers) == 4
        assert all(unit.kind == "MEMORY_BEARING" for unit in consumers)
        assert all(unit.prefix_unit_id == prefix.unit_id for unit in consumers)
        assert all(unit.seed == prefix.seed for unit in consumers)
        assert all(unit.task == prefix.task for unit in consumers)
        assert all(unit.memory_baseline == prefix.memory_baseline for unit in consumers)
        assert prefix.arm == "NOT_APPLICABLE"
        assert prefix.prefix_unit_id is None


def test_nomem_has_no_prefix_and_projection_telescopes_to_frozen_total() -> None:
    units = enumerate_execution_units(_candidate_package())
    nomem = tuple(unit for unit in units if unit.kind == "NO_MEMORY_SINGLETON")

    assert all(unit.prefix_unit_id is None for unit in nomem)
    assert all(unit.projected_cost_krw > 0 for unit in units)
    assert sum(unit.projected_cost_krw for unit in units) == 444256
    assert units[52].projected_cost_krw == 5


def test_production_order_and_projection_are_deterministic() -> None:
    first = enumerate_execution_units(_candidate_package())
    second = enumerate_execution_units(_candidate_package())

    assert first == second


def test_prefix_terminal_failure_marks_all_four_consumers_without_dispatch(
    tmp_path: Path,
) -> None:
    units = enumerate_execution_units(_candidate_package())
    ledger = MainRunLedger.create(
        tmp_path / "main-run.sqlite3",
        MainRunBinding(
            package_id="phase13-main-a-execution-freeze-v1",
            package_sha256="1" * 64,
            package_hash="2" * 64,
            authorization_id="phase13-main-a-authorized-execution-v1",
            authorization_sha256="3" * 64,
            authorization_hash="4" * 64,
            runner_sha256="5" * 64,
        ),
        units,
    )
    dispatched: list[str] = []

    def fail_prefix(unit):
        dispatched.append(unit.unit_id)
        evidence = persist_reconciliation_evidence(
            tmp_path,
            "TERMINAL_FAILURE",
            ledger.in_flight_context(unit.unit_id),
            failure_code="PROVIDER_QUOTA",
        )
        raise DispatchTechnicalFailure(
            "PROVIDER_QUOTA",
            evidence.evidence_sha256,
        )

    report = run_pending(ledger, fail_prefix, tranche_ceiling_krw=444126)

    assert dispatched == [units[0].unit_id]
    assert report.terminal_technical_missing_count == 5
    assert ledger.status().pending_count == 1195
    assert ledger.next_pending() == units[5]


def test_reconciled_prefix_terminal_failure_also_marks_all_four_consumers(
    tmp_path: Path,
) -> None:
    units = enumerate_execution_units(_candidate_package())
    ledger = MainRunLedger.create(
        tmp_path / "main-run.sqlite3",
        MainRunBinding(
            "phase13-main-a-execution-freeze-v1",
            "1" * 64,
            "2" * 64,
            "phase13-main-a-authorized-execution-v1",
            "3" * 64,
            "4" * 64,
            "5" * 64,
        ),
        units,
    )
    prefix = units[0]
    ledger.persist_dispatch_intent(prefix.unit_id)
    context = ledger.in_flight_context(prefix.unit_id)

    ledger.reconcile(
        prefix.unit_id,
        persist_reconciliation_evidence(
            tmp_path,
            "TERMINAL_FAILURE",
            context,
            failure_code="PROVIDER_QUOTA",
        ),
    )

    assert ledger.status().terminal_technical_missing_count == 5
    assert ledger.next_pending() == units[5]
