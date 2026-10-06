from __future__ import annotations
from .phase13_corrective_identity import corrective_identity

import importlib
import importlib.util
import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path
from types import ModuleType

import pytest

from memcontam.readiness.phase13_v3_terminal_models import attempt_identity


def test_v3_durable_contract_is_available() -> None:
    assert importlib.util.find_spec("memcontam.readiness.phase13_v3_terminal_ledger") is not None


@pytest.fixture
def api() -> ModuleType:
    name = "memcontam.readiness.phase13_v3_terminal_ledger"
    assert importlib.util.find_spec(name) is not None, "V3 durable terminal evidence seam absent"
    return importlib.import_module(name)


@pytest.fixture
def ledger(api: ModuleType, tmp_path: Path):
    return api.TerminalLedgerV3.create(tmp_path / "ledger.sqlite3", {
        "schema_version": "phase13_main_run_ledger_v3", "unit_ids": ["a" * 64, "b" * 64],
        "identity": corrective_identity().model_dump(mode="json"),
        "package_sha256": "c" * 64, "authorization_sha256": "d" * 64,
    })


COMPILED = {"stage": "rag_generate", "token_count": 379, "compiled_request_hash": "1" * 64,
            "immutable_input_hash": "2" * 64, "native_state_hash": "3" * 64}
PATHS = {
    "PENDING": (), "DISPATCH_INTENT_PERSISTED": ("DISPATCH_INTENT",),
    "REQUEST_COMPILED": ("DISPATCH_INTENT", "REQUEST_COMPILED"),
    "INPUT_ENVELOPE_OVERFLOW": ("DISPATCH_INTENT", "REQUEST_COMPILED", "INPUT_ENVELOPE_OVERFLOW"),
    "TERMINAL_TECHNICAL_MISSING": ("DISPATCH_INTENT", "REQUEST_COMPILED", "INPUT_ENVELOPE_OVERFLOW", "TERMINAL_TECHNICAL_MISSING"),
    "ATTEMPT_STARTED": ("DISPATCH_INTENT", "REQUEST_COMPILED", "ATTEMPT_STARTED"),
    **{kind: ("DISPATCH_INTENT", "REQUEST_COMPILED", "ATTEMPT_STARTED", kind)
       for kind in ("COMPLETED", "ATTEMPTED_PROVIDER_FAILURE", "AMBIGUOUS_ATTEMPT")},
}
EDGES = {
    ("PENDING", "DISPATCH_INTENT"): "DISPATCH_INTENT_PERSISTED",
    ("DISPATCH_INTENT_PERSISTED", "REQUEST_COMPILED"): "REQUEST_COMPILED",
    ("DISPATCH_INTENT_PERSISTED", "NO_REQUEST"): "PENDING",
    ("REQUEST_COMPILED", "NO_REQUEST"): "PENDING",
    ("REQUEST_COMPILED", "INPUT_ENVELOPE_OVERFLOW"): "INPUT_ENVELOPE_OVERFLOW",
    ("INPUT_ENVELOPE_OVERFLOW", "TERMINAL_TECHNICAL_MISSING"): "TERMINAL_TECHNICAL_MISSING",
    ("REQUEST_COMPILED", "ATTEMPT_STARTED"): "ATTEMPT_STARTED",
    **{("ATTEMPT_STARTED", kind): kind for kind in
       ("COMPLETED", "ATTEMPTED_PROVIDER_FAILURE", "AMBIGUOUS_ATTEMPT")},
}
KINDS = tuple(dict.fromkeys(kind for _, kind in EDGES))


def event(ledger, kind: str, unit: str = "a" * 64) -> dict:
    state = ledger.state(unit)
    raw = {"schema_version": "phase13_main_dispatch_evidence_v3", "unit_id": unit,
           "revision": state.revision + 1, "previous_hash": state.event_hash,
           "kind": kind, "compiled": state.compiled.model_dump() if state.compiled else None}
    if kind == "REQUEST_COMPILED":
        raw["compiled"] = COMPILED.copy()
    if kind == "ATTEMPT_STARTED":
        raw.update(attempt_index=0, attempt_id=attempt_identity(unit, 0))
    if kind in ("INPUT_ENVELOPE_OVERFLOW", "TERMINAL_TECHNICAL_MISSING"):
        raw.update(failure_code="MAIN_INPUT_ENVELOPE_EXCEEDED", transport_attempts=0, realized_cost_krw=0)
    if kind in ("COMPLETED", "ATTEMPTED_PROVIDER_FAILURE", "AMBIGUOUS_ATTEMPT"):
        raw.update(transport_attempts=1, cost={"usage": None, "monetary_cost": None, "currency": None},
                   realized_cost_krw=None)
    if kind == "COMPLETED":
        raw.update(result_hash="4" * 64)
    if kind == "ATTEMPTED_PROVIDER_FAILURE":
        raw.update(failure_code="PROVIDER_TIMEOUT", observation_hash="5" * 64)
    if kind in ("NO_REQUEST", "AMBIGUOUS_ATTEMPT"):
        raw.update(schema_version="phase13_main_reconciliation_v3", proof_hash="6" * 64)
    if kind == "AMBIGUOUS_ATTEMPT":
        raw.update(failure_code="MAIN_AMBIGUOUS_ATTEMPT")
    return raw


def reach(ledger, state: str) -> None:
    for kind in PATHS[state]:
        ledger.append(event(ledger, kind))


@pytest.mark.parametrize("state,kind", [(state, kind) for state in PATHS for kind in KINDS])
def test_exact_transition_matrix_when_event_is_fresh(ledger, state: str, kind: str) -> None:
    reach(ledger, state)
    supplied = event(ledger, kind)
    expected = EDGES.get((state, kind))
    before = ledger.rows()
    if expected is None:
        with pytest.raises(ValueError, match="MAIN_TERMINAL_EVIDENCE_CONFLICT"):
            ledger.append(supplied)
        assert ledger.rows() == before
    else:
        ledger.append(supplied)
        assert ledger.state("a" * 64).kind == expected


@pytest.mark.parametrize("state", tuple(PATHS))
def test_reopen_preserves_state_and_canonical_bytes(api: ModuleType, ledger, state: str) -> None:
    reach(ledger, state)
    reopened = api.TerminalLedgerV3.open(ledger.path, ledger.binding)
    assert reopened.state("a" * 64) == ledger.state("a" * 64)
    assert reopened.rows() == ledger.rows()
    for raw in reopened.rows():
        assert raw == (json.dumps(json.loads(raw), sort_keys=True, separators=(",", ":")) + "\n").encode()


@pytest.mark.parametrize("field,value", [(field, None) for field in COMPILED] + [
    ("token_count", -1), ("token_count", True), ("compiled_request_hash", "bad"), ("stage", "")])
def test_malformed_compilation_fails_without_write(ledger, field: str, value) -> None:
    reach(ledger, "DISPATCH_INTENT_PERSISTED")
    supplied = event(ledger, "REQUEST_COMPILED")
    supplied["compiled"][field] = value
    before = ledger.rows()
    with pytest.raises(ValueError, match="MAIN_TERMINAL_EVIDENCE_CONFLICT"):
        ledger.append(supplied)
    assert ledger.rows() == before


@pytest.mark.parametrize("kind", ("ATTEMPT_STARTED", "INPUT_ENVELOPE_OVERFLOW", "NO_REQUEST"))
def test_compiled_binding_cannot_be_erased_or_changed(ledger, kind: str) -> None:
    reach(ledger, "REQUEST_COMPILED")
    supplied = event(ledger, kind)
    supplied["compiled"] = None
    with pytest.raises(ValueError, match="MAIN_TERMINAL_EVIDENCE_CONFLICT"):
        ledger.append(supplied)


@pytest.mark.parametrize("state", tuple(PATHS)[1:])
def test_duplicate_writes_are_idempotent_after_reopen(api: ModuleType, ledger, state: str) -> None:
    reach(ledger, state)
    before = ledger.rows()
    api.TerminalLedgerV3.open(ledger.path, ledger.binding).append(json.loads(before[-1]))
    assert ledger.rows() == before


def test_conflicting_terminalization_is_rejected(ledger) -> None:
    reach(ledger, "ATTEMPTED_PROVIDER_FAILURE")
    before = ledger.rows()
    changed = json.loads(before[-1])
    changed["failure_code"] = "PROVIDER_5XX"
    with pytest.raises(ValueError) as caught:
        ledger.append(changed)
    assert getattr(caught.value, "code") == "MAIN_TERMINAL_EVIDENCE_CONFLICT"
    assert ledger.rows() == before


def test_unknown_attempted_cost_fails_closed(ledger) -> None:
    reach(ledger, "ATTEMPTED_PROVIDER_FAILURE")
    before = ledger.rows()
    with pytest.raises(ValueError) as caught:
        ledger.append(event(ledger, "DISPATCH_INTENT", "b" * 64))
    assert getattr(caught.value, "code") == "MAIN_TERMINAL_COST_UNKNOWN"
    assert ledger.rows() == before
    assert json.loads(before[-1])["realized_cost_krw"] is None


@pytest.mark.parametrize("state,expected", [("DISPATCH_INTENT_PERSISTED", "PENDING"),
    ("REQUEST_COMPILED", "PENDING"), ("ATTEMPT_STARTED", "AMBIGUOUS_ATTEMPT")])
def test_recovery_uses_durable_marker_not_claimed_no_request(api: ModuleType, ledger, state: str, expected: str) -> None:
    reach(ledger, state)
    reopened = api.TerminalLedgerV3.open(ledger.path, ledger.binding)
    reopened.recover("a" * 64, "6" * 64)
    assert reopened.state("a" * 64).kind == expected
    if expected == "PENDING":
        reopened.append(event(reopened, "DISPATCH_INTENT"))
        assert reopened.state("a" * 64).kind == "DISPATCH_INTENT_PERSISTED"
    else:
        assert json.loads(reopened.rows()[-1])["failure_code"] == "MAIN_AMBIGUOUS_ATTEMPT"


@pytest.mark.parametrize("cost,realized", [({"monetary_cost": "0.001", "currency": "USD"}, 2),
    ({"usage": {"input_tokens": 100, "output_tokens": 0}}, 1)])
def test_known_cost_allows_independent_dispatch(ledger, cost: dict, realized: int) -> None:
    reach(ledger, "ATTEMPT_STARTED")
    supplied = event(ledger, "ATTEMPTED_PROVIDER_FAILURE")
    supplied.update(cost=cost, realized_cost_krw=realized)
    ledger.append(supplied)
    ledger.append(event(ledger, "DISPATCH_INTENT", "b" * 64))
    assert ledger.state("b" * 64).kind == "DISPATCH_INTENT_PERSISTED"


@pytest.mark.parametrize("realized", [0, 1, -1, True])
def test_unknown_cost_cannot_be_coerced_to_number(ledger, realized) -> None:
    reach(ledger, "ATTEMPT_STARTED")
    supplied = event(ledger, "ATTEMPTED_PROVIDER_FAILURE")
    supplied["realized_cost_krw"] = realized
    with pytest.raises(ValueError, match="MAIN_TERMINAL_EVIDENCE_CONFLICT"):
        ledger.append(supplied)


@pytest.mark.parametrize("boundary", [1, 2])
def test_fsync_interruption_reopen_preserves_committed_marker(api: ModuleType, ledger, monkeypatch, boundary: int) -> None:
    reach(ledger, "REQUEST_COMPILED")
    supplied = event(ledger, "ATTEMPT_STARTED")
    original, calls = os.fsync, 0
    def interrupted(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == boundary:
            raise OSError("injected fsync interruption")
        original(descriptor)
    with monkeypatch.context() as patch:
        patch.setattr(api.os, "fsync", interrupted)
        with pytest.raises(OSError, match="injected fsync interruption"):
            ledger.append(supplied)
    reopened = api.TerminalLedgerV3.open(ledger.path, ledger.binding)
    reopened.recover("a" * 64, "6" * 64)
    assert reopened.state("a" * 64).kind == "AMBIGUOUS_ATTEMPT"


def test_uncommitted_sqlite_interruption_rolls_back_on_reopen(api: ModuleType, ledger) -> None:
    reach(ledger, "REQUEST_COMPILED")
    before = ledger.rows()
    with closing(sqlite3.connect(ledger.path)) as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("INSERT INTO events(raw, event_hash) VALUES (?, ?)", (b"partial", "0" * 64))
    reopened = api.TerminalLedgerV3.open(ledger.path, ledger.binding)
    assert reopened.rows() == before


def test_replayed_old_event_does_not_rewind(ledger) -> None:
    reach(ledger, "COMPLETED")
    before = ledger.rows()
    ledger.append(json.loads(before[0]))
    assert ledger.state("a" * 64).kind == "COMPLETED"
    assert ledger.rows() == before


@pytest.mark.parametrize("field,value", [("transport_attempts", False), ("realized_cost_krw", False),
    ("transport_attempts", 0.0), ("realized_cost_krw", 0.0)])
def test_zero_attempt_evidence_rejects_boolean_and_float(ledger, field: str, value) -> None:
    reach(ledger, "REQUEST_COMPILED")
    supplied = event(ledger, "INPUT_ENVELOPE_OVERFLOW")
    supplied[field] = value
    with pytest.raises(ValueError, match="MAIN_TERMINAL_EVIDENCE_CONFLICT"):
        ledger.append(supplied)


@pytest.mark.parametrize("field,value", [("schema_version", "phase13_main_dispatch_evidence_v1"),
    ("schema_version", "phase13_main_dispatch_evidence_v2"), ("unit_id", "9" * 64),
    ("revision", 20), ("previous_hash", "9" * 64), ("kind", "BOGUS"), ("extra", 1)])
def test_untrusted_event_identity_fails_closed(ledger, field: str, value) -> None:
    supplied = event(ledger, "DISPATCH_INTENT")
    supplied[field] = value
    with pytest.raises(ValueError, match="MAIN_TERMINAL_EVIDENCE_CONFLICT"):
        ledger.append(supplied)
    assert ledger.rows() == ()


@pytest.mark.parametrize("field", tuple(COMPILED))
def test_post_compile_mutation_preserves_first_bytes(ledger, field: str) -> None:
    reach(ledger, "REQUEST_COMPILED")
    before = ledger.rows()
    supplied = event(ledger, "ATTEMPT_STARTED")
    supplied["compiled"][field] = 380 if field == "token_count" else "8" * 64
    with pytest.raises(ValueError, match="MAIN_TERMINAL_EVIDENCE_CONFLICT"):
        ledger.append(supplied)
    assert ledger.rows() == before


def test_overflow_interruption_finishes_terminal_on_recovery(api: ModuleType, ledger) -> None:
    reach(ledger, "INPUT_ENVELOPE_OVERFLOW")
    reopened = api.TerminalLedgerV3.open(ledger.path, ledger.binding)
    reopened.recover("a" * 64, "6" * 64)
    assert reopened.state("a" * 64).kind == "TERMINAL_TECHNICAL_MISSING"
    assert json.loads(reopened.rows()[-1])["realized_cost_krw"] == 0


def test_changed_last_evidence_row_rejected_on_reopen(api: ModuleType, ledger) -> None:
    reach(ledger, "ATTEMPTED_PROVIDER_FAILURE")
    changed = json.loads(ledger.rows()[-1])
    changed["failure_code"] = "FORGED"
    raw = (json.dumps(changed, sort_keys=True, separators=(",", ":")) + "\n").encode()
    with closing(sqlite3.connect(ledger.path)) as connection, connection:
        connection.execute("UPDATE events SET raw = ? WHERE sequence = (SELECT MAX(sequence) FROM events)", (raw,))
    with pytest.raises(ValueError, match="MAIN_TERMINAL_EVIDENCE_CONFLICT"):
        api.TerminalLedgerV3.open(ledger.path, ledger.binding)
