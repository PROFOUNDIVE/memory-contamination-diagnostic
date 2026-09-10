from __future__ import annotations

import contextlib
import hashlib
import json
import sqlite3
from contextlib import closing
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import JsonValue

from memcontam.readiness.phase13_main_request_recovery import recover_requests
from memcontam.readiness.phase13_main_v3_runner import V3MainRun, V3RunStatus
from memcontam.readiness.phase13_v3_terminal_ledger import TerminalLedgerV3

from .test_phase13_v3_envelope_gate import api as api
from .test_phase13_v3_envelope_gate import rig as rig
from .test_phase13_v3_entrypoint_fixture import entrypoint_bytes as entrypoint_bytes
from .test_phase13_v3_entrypoint_fixture import entrypoint_fixture as entrypoint_fixture
from .test_phase13_v3_entrypoint_integration import deny_external as deny_external


def _binding(unit_count: int) -> dict[str, JsonValue]:
    return {
        "schema_version": "phase13_main_run_ledger_v3",
        "unit_ids": [hashlib.sha256(str(index).encode()).hexdigest() for index in range(unit_count)],
        "package_sha256": "b" * 64,
        "authorization_sha256": "c" * 64,
    }


def test_t01_replay_hashes_large_binding_once_and_preserves_genesis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import memcontam.readiness.phase13_v3_terminal_ledger as ledger_api

    ledger = TerminalLedgerV3.create(tmp_path / "ledger.sqlite3", _binding(64))
    original = ledger_api.digest
    calls = 0

    def counted(value, excluded_field=None):
        nonlocal calls
        if value == ledger.binding:
            calls += 1
        return original(value, excluded_field)

    monkeypatch.setattr(ledger_api, "digest", counted)
    states = TerminalLedgerV3.open(ledger.path, ledger.binding).states()
    binding_hash = original(ledger.binding)

    assert calls == 1
    assert tuple(states) == ledger.binding.unit_ids
    assert all(
        state.event_hash == hashlib.sha256(f"{binding_hash}:{unit_id}".encode()).hexdigest()
        for unit_id, state in states.items()
    )


def test_t02_fresh_recovery_uses_one_state_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = TerminalLedgerV3.create(tmp_path / "ledger.sqlite3", _binding(64))
    original = TerminalLedgerV3._replay
    calls = 0

    def counted(self, connection):
        nonlocal calls
        calls += 1
        return original(self, connection)

    monkeypatch.setattr(TerminalLedgerV3, "_replay", counted)
    recover_requests(ledger)

    assert calls == 1
    assert ledger.rows() == ()


def test_t03_receive_and_acknowledge_process_only_new_events(
    rig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = rig.dispatcher.receive(rig.keys[0], rig.material)
    rig.dispatcher.acknowledge(rig.keys[0], first, semantic_success=True)
    import memcontam.readiness.phase13_v3_terminal_ledger as ledger_api

    original_parse = ledger_api.parse_event
    original_read = TerminalLedgerV3.read_record
    parsed = 0
    receipts = 0

    def counted_parse(raw: bytes):
        nonlocal parsed
        parsed += 1
        return original_parse(raw)

    def counted_read(self, name: str) -> bytes:
        nonlocal receipts
        receipts += 1
        return original_read(self, name)

    monkeypatch.setattr(ledger_api, "parse_event", counted_parse)
    monkeypatch.setattr(TerminalLedgerV3, "read_record", counted_read)
    second = rig.dispatcher.receive(rig.keys[1], rig.material)
    rig.dispatcher.acknowledge(rig.keys[1], second, semantic_success=True)

    assert parsed == 4
    assert receipts == 0
    assert rig.seen.requests == rig.seen.constructors == 2


def test_t03_second_dispatcher_observes_terminal_parent(rig) -> None:
    second = rig.api.ProductionRequestDispatcherV3(
        rig.ledger, rig.binding, rig.parents, provider_factory=lambda _binding: None,
    )
    assert second.terminal_parents == frozenset()
    rig.seen.count = 379
    with pytest.raises(rig.api.DispatchTechnicalFailureV3):
        rig.dispatcher.dispatch(rig.keys[0], rig.material, rig.semantic)

    assert second.terminal_parents == frozenset({"a" * 64, "d" * 64})


@pytest.mark.parametrize("mutation", ["event", "metadata", "delete"])
def test_t03_cached_ledger_rejects_direct_sql_mutation(tmp_path: Path, mutation: str) -> None:
    ledger = TerminalLedgerV3.create(tmp_path / "ledger.sqlite3", _binding(1))
    unit_id = ledger.binding.unit_ids[0]
    state = ledger.state(unit_id)
    ledger.append({
        "schema_version": "phase13_main_dispatch_evidence_v3",
        "unit_id": unit_id,
        "revision": 1,
        "previous_hash": state.event_hash,
        "kind": "DISPATCH_INTENT",
        "compiled": None,
    })
    ledger.rows()
    with closing(sqlite3.connect(ledger.path)) as connection, connection:
        if mutation == "event":
            raw = json.loads(ledger.rows()[0])
            raw["kind"] = "FORGED"
            connection.execute("UPDATE events SET raw=? WHERE sequence=1", (
                (json.dumps(raw, sort_keys=True, separators=(",", ":")) + "\n").encode(),
            ))
        elif mutation == "metadata":
            connection.execute("UPDATE metadata SET raw=?", (b"{}\n",))
        else:
            connection.execute("DELETE FROM events WHERE sequence=1")

    with pytest.raises(ValueError, match="MAIN_TERMINAL_EVIDENCE_CONFLICT"):
        ledger.rows()


def test_t05_live_selector_invokes_governed_source_gate(
    entrypoint_fixture, monkeypatch: pytest.MonkeyPatch, deny_external,
) -> None:
    import memcontam.readiness.phase13_v3_entrypoint as entrypoint_api
    from memcontam.readiness.phase13_v3_resource_files import ClosureError

    request = entrypoint_fixture

    def reject(_root: Path, _inventory) -> None:
        raise ClosureError("MAIN_GOVERNED_SOURCE_DRIFT")

    monkeypatch.setattr(entrypoint_api, "validate_governed", reject, raising=False)
    with pytest.raises(ValueError, match="MAIN_GOVERNED_SOURCE_DRIFT"):
        entrypoint_api.select_execution(request, "run")


@pytest.mark.parametrize("target", [
    "src/memcontam/__init__.py",
    "data/phase13/main/mr_p4/corrected_v3/manifest_v3.json",
    "data/phase13/main/main_live_contract_v3.json",
])
def test_t05_live_selector_rejects_actual_closure_member_drift(
    entrypoint_fixture, deny_external, target: str,
) -> None:
    from memcontam.readiness.phase13_v3_entrypoint import select_execution

    path = entrypoint_fixture.repository_root / target
    path.write_bytes(path.read_bytes() + b"drift")
    with pytest.raises(ValueError, match="MAIN_GOVERNED_SOURCE_DRIFT"):
        select_execution(entrypoint_fixture, "run")


def test_t05_execute_rejects_governed_source_drift_after_open(
    entrypoint_fixture, deny_external,
) -> None:
    from memcontam.readiness.phase13_v3_entrypoint import select_execution

    selected = select_execution(entrypoint_fixture, "run")
    from memcontam.readiness.phase13_v3_entrypoint import SelectedExecutionV3
    assert isinstance(selected, SelectedExecutionV3)
    run = V3MainRun.open(selected, entrypoint_fixture.repository_root / "run", create=True, seed=0)
    try:
        path = entrypoint_fixture.repository_root / "src/memcontam/__init__.py"
        path.write_bytes(path.read_bytes() + b"drift")
        with pytest.raises(ValueError, match="MAIN_(GOVERNED_SOURCE_DRIFT|PATH_UNSAFE)"):
            run.execute(entrypoint_fixture.repository_root / "cache", max_units=0, tranche_ceiling_krw=450000)
    finally:
        run.close()


class _Rows:
    def __iter__(self):
        return iter(())

    def fetchone(self) -> tuple[None]:
        return (None,)


class _Connection:
    def execute(self, _sql: str, _parameters: tuple[str, ...] = ()) -> _Rows:
        return _Rows()


class _Private:
    @contextlib.contextmanager
    def connect(self):
        yield _Connection()


@pytest.mark.parametrize("seed", [0, 4, 9])
def test_t06_selected_seed_boundary_precedes_skip_and_runtime_construction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, seed: int,
) -> None:
    units = tuple(
        SimpleNamespace(sequence=sequence, seed=sequence // 120, unit_id=f"unit-{sequence}", projected_cost_krw=1)
        for sequence in range(1200)
    )
    selected = SimpleNamespace(
        package=SimpleNamespace(production=units, tranche_unit_count=120),
        costs=SimpleNamespace(resources=SimpleNamespace(phase4=SimpleNamespace(base=SimpleNamespace(units=())))),
        repository_root=tmp_path,
        preflight=lambda _root: None,
    )
    terminal = frozenset(unit.unit_id for unit in units if unit.seed <= seed)
    dispatcher = SimpleNamespace(terminal_parents=terminal, recover=lambda: None)
    ledger = SimpleNamespace(states=lambda: {})
    run = object.__new__(V3MainRun)
    object.__setattr__(run, "selected", selected)
    object.__setattr__(run, "private", _Private())
    object.__setattr__(run, "ledger", ledger)
    object.__setattr__(run, "lease", ExitStack())
    object.__setattr__(run, "seed", seed)
    terminal_count = (seed + 1) * 120
    status = V3RunStatus("READY" if seed < 9 else "COMPLETED", 0, terminal_count,
                         1200 - terminal_count, 0)
    constructions: list[int] = []

    monkeypatch.setattr(V3MainRun, "dispatcher", lambda *_args, **_kwargs: dispatcher)
    monkeypatch.setattr(V3MainRun, "status", lambda _self: status)

    class Runtime:
        def __init__(self, *_args, **_kwargs) -> None:
            constructions.append(1)

    import memcontam.readiness.phase13_main_live_runtime as runtime_api

    monkeypatch.setattr(runtime_api, "ProductionMainRuntime", Runtime)
    assert run.execute(tmp_path / "cache", max_units=None, tranche_ceiling_krw=120) == status
    assert constructions == []


def test_t06_middle_seed_rejects_pending_preceding_tranches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    units = tuple(
        SimpleNamespace(sequence=sequence, seed=sequence // 120, unit_id=f"unit-{sequence}", projected_cost_krw=1)
        for sequence in range(1200)
    )
    selected = SimpleNamespace(
        package=SimpleNamespace(production=units, tranche_unit_count=120),
        costs=SimpleNamespace(resources=SimpleNamespace(phase4=SimpleNamespace(base=SimpleNamespace(units=())))),
        repository_root=tmp_path,
        preflight=lambda _root: None,
    )
    dispatcher = SimpleNamespace(terminal_parents=frozenset(), recover=lambda: None)
    run = object.__new__(V3MainRun)
    object.__setattr__(run, "selected", selected)
    object.__setattr__(run, "private", _Private())
    object.__setattr__(run, "ledger", SimpleNamespace(states=lambda: {}))
    object.__setattr__(run, "lease", ExitStack())
    object.__setattr__(run, "seed", 4)
    monkeypatch.setattr(V3MainRun, "dispatcher", lambda *_args, **_kwargs: dispatcher)
    monkeypatch.setattr(V3MainRun, "status", lambda _self: V3RunStatus("READY", 0, 0, 1200, 0))

    with pytest.raises(ValueError, match="MAIN_TRANCHE_ORDER_MISMATCH"):
        run.execute(tmp_path / "cache", max_units=0, tranche_ceiling_krw=450000)


@pytest.mark.parametrize("terminal_count,session_state,pending", [
    (120, "READY", 1080),
    (1200, "COMPLETED", 0),
])
def test_t06_global_status_completes_only_after_seed_nine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    terminal_count: int, session_state: str, pending: int,
) -> None:
    units = tuple(SimpleNamespace(unit_id=f"unit-{sequence}") for sequence in range(1200))
    selected = SimpleNamespace(
        package=SimpleNamespace(production=units, tranche_unit_count=120),
        repository_root=tmp_path,
    )

    class StatusRows:
        def __iter__(self):
            return iter(())

        def fetchone(self) -> tuple[int]:
            return (120,)

    class StatusConnection:
        def execute(self, _sql: str) -> StatusRows:
            return StatusRows()

    class StatusPrivate:
        @contextlib.contextmanager
        def connect(self):
            yield StatusConnection()

    run = object.__new__(V3MainRun)
    object.__setattr__(run, "selected", selected)
    object.__setattr__(run, "private", StatusPrivate())
    object.__setattr__(run, "ledger", SimpleNamespace(rows=lambda: ()))
    object.__setattr__(run, "lease", ExitStack())
    object.__setattr__(run, "seed", 0)
    monkeypatch.setattr(
        V3MainRun, "dispatcher", lambda *_args, **_kwargs: SimpleNamespace(
            terminal_parents=frozenset(unit.unit_id for unit in units[:terminal_count]),
        ),
    )

    assert run.status() == V3RunStatus(session_state, 0, terminal_count, pending, 0)


@pytest.mark.parametrize("sequence,seed", [(119, 1), (120, 0)])
def test_t06_package_rejects_seed_outside_exact_seed_zero_partition(
    sequence: int, seed: int,
) -> None:
    from memcontam.readiness.phase13_v3_entrypoint_models import MainExecutionPackageV3

    path = Path("data/phase13/main/mr_p5/execution_package_v3.json")
    payload = json.loads(path.read_bytes())
    payload["tranche_unit_count"] = 120
    payload["production"][sequence]["seed"] = seed

    with pytest.raises(ValueError, match="MAIN_AUTHORIZATION_BINDING_MISMATCH"):
        MainExecutionPackageV3.model_validate_json(json.dumps(payload))
