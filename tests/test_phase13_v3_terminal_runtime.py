from __future__ import annotations

import json
import importlib

import pytest

from memcontam.readiness.phase13_v3_terminal_ledger import TerminalLedgerV3

_fixtures = importlib.import_module("tests.test_phase13_v3_envelope_gate")
api, rig = _fixtures.api, _fixtures.rig


class Crash(BaseException):
    pass


def restart(rig):
    ledger = TerminalLedgerV3.open(rig.ledger.path, rig.ledger.binding)
    return rig.api.ProductionRequestDispatcherV3(
        ledger, rig.binding, rig.parents, provider_factory=rig.dispatcher._factory,
    )


@pytest.mark.parametrize("outcome", ["transport", "max_output_tokens", "parse", "unknown"])
def test_attempted_failure_restart_never_redispatches(rig, outcome):
    rig.seen.outcome = outcome
    with pytest.raises(rig.api.DispatchTechnicalFailureV3):
        rig.dispatcher.dispatch(rig.keys[0], rig.material, rig.semantic)
    resumed = restart(rig)
    assert resumed.terminal_parents == frozenset({"a" * 64, "d" * 64})
    for key in rig.keys[:4]:
        with pytest.raises(rig.api.DispatchTechnicalFailureV3, match="MAIN_TRAJECTORY_TERMINAL"):
            resumed.dispatch(key, rig.material, rig.semantic)
    assert rig.seen.requests == rig.seen.constructors == 1
    assert rig.ledger.state(rig.keys[0].dispatch_id).kind == "ATTEMPTED_PROVIDER_FAILURE"


@pytest.mark.parametrize("point", ["DISPATCH_INTENT", "REQUEST_COMPILED"])
def test_pre_marker_no_request_returns_pending(rig, monkeypatch, point):
    original = rig.dispatcher._append

    def append(key, kind, extra=None):
        original(key, kind, extra)
        if kind == point:
            raise Crash()

    monkeypatch.setattr(rig.dispatcher, "_append", append)
    with pytest.raises(Crash):
        rig.dispatcher.dispatch(rig.keys[0], rig.material, rig.semantic)
    before = rig.ledger.state(rig.keys[0].dispatch_id)
    assert (before.compiled is None) == (point == "DISPATCH_INTENT")
    resumed = restart(rig)
    resumed.recover()
    assert rig.ledger.state(rig.keys[0].dispatch_id).kind == "PENDING"
    assert json.loads(rig.ledger.rows()[-1])["kind"] == "NO_REQUEST"
    assert resumed.dispatch(rig.keys[0], rig.material, rig.semantic) == "final: 24"
    assert rig.seen.requests == 1


@pytest.mark.parametrize("point", ["marker", "response", "observation"])
def test_post_marker_crash_becomes_ambiguous(rig, monkeypatch, point):
    if point == "marker":
        original = rig.dispatcher._append

        def append(key, kind, extra=None):
            original(key, kind, extra)
            if kind == "ATTEMPT_STARTED":
                raise Crash()

        monkeypatch.setattr(rig.dispatcher, "_append", append)
    elif point == "response":
        def semantic(response):
            raise Crash()

        rig.semantic = semantic
    else:
        rig.seen.outcome = "transport"
        original_publish = rig.dispatcher._publish_bytes

        def publish(key, role, raw):
            original_publish(key, role, raw)
            if role == "observation":
                raise Crash()

        monkeypatch.setattr(rig.dispatcher, "_publish_bytes", publish)
    with pytest.raises(Crash):
        rig.dispatcher.dispatch(rig.keys[0], rig.material, rig.semantic)
    calls = rig.seen.requests
    resumed = restart(rig)
    resumed.recover()
    state = rig.ledger.state(rig.keys[0].dispatch_id)
    assert state.kind == "AMBIGUOUS_ATTEMPT"
    assert state.compiled is not None
    assert state.attempted_cost.monetary_cost is None
    for key in rig.keys[:4]:
        with pytest.raises(rig.api.DispatchTechnicalFailureV3):
            resumed.dispatch(key, rig.material, rig.semantic)
    assert rig.seen.requests == calls


def test_unknown_cost_blocks_independent_units(rig):
    rig.seen.outcome = "transport"
    with pytest.raises(rig.api.DispatchTechnicalFailureV3):
        rig.dispatcher.dispatch(rig.keys[0], rig.material, rig.semantic)
    resumed = restart(rig)
    resumed.recover()
    before = rig.ledger.rows()
    with pytest.raises(ValueError, match="MAIN_TERMINAL_COST_UNKNOWN"):
        resumed.dispatch(rig.keys[4], rig.material, rig.semantic)
    assert rig.ledger.rows() == before
    assert rig.seen.requests == rig.seen.constructors == 1


@pytest.mark.parametrize("point", ["INPUT_ENVELOPE_OVERFLOW", "TERMINAL_TECHNICAL_MISSING"])
def test_overflow_restart_allows_only_independent_progress(rig, monkeypatch, point):
    rig.seen.count = 379
    original = rig.dispatcher._append

    def append(key, kind, extra=None):
        original(key, kind, extra)
        if kind == point:
            raise Crash()

    monkeypatch.setattr(rig.dispatcher, "_append", append)
    with pytest.raises(Crash):
        rig.dispatcher.dispatch(rig.keys[0], rig.material, rig.semantic)
    resumed = restart(rig)
    resumed.recover()
    assert rig.ledger.state(rig.keys[0].dispatch_id).kind == "TERMINAL_TECHNICAL_MISSING"
    for key in rig.keys[:4]:
        with pytest.raises(rig.api.DispatchTechnicalFailureV3):
            resumed.dispatch(key, rig.material, rig.semantic)
    assert rig.seen.requests == rig.seen.constructors == 0
    rig.seen.count = 378
    assert resumed.dispatch(rig.keys[4], rig.material, rig.semantic) == "final: 24"
    assert rig.seen.requests == 1


@pytest.mark.parametrize("after_sync", [False, True])
def test_terminal_fsync_interruption_preserves_first_evidence(rig, monkeypatch, after_sync):
    rig.seen.outcome = "transport"
    original = TerminalLedgerV3._sync

    def sync(ledger):
        if ledger.state(rig.keys[0].dispatch_id).kind == "ATTEMPTED_PROVIDER_FAILURE":
            if after_sync:
                original(ledger)
            raise Crash()
        original(ledger)

    with monkeypatch.context() as patch:
        patch.setattr(TerminalLedgerV3, "_sync", sync)
        with pytest.raises(Crash):
            rig.dispatcher.dispatch(rig.keys[0], rig.material, rig.semantic)
    before = rig.ledger.rows()
    resumed = restart(rig)
    resumed.recover()
    assert rig.ledger.rows() == before
    assert rig.ledger.state(rig.keys[0].dispatch_id).kind == "ATTEMPTED_PROVIDER_FAILURE"
    with pytest.raises(rig.api.DispatchTechnicalFailureV3):
        resumed.dispatch(rig.keys[1], rig.material, rig.semantic)
    assert rig.seen.requests == 1


def test_failure_exposes_durable_evidence_hash_before_raise(rig):
    rig.seen.outcome = "transport"
    with pytest.raises(rig.api.DispatchTechnicalFailureV3) as failure:
        rig.dispatcher.dispatch(rig.keys[0], rig.material, rig.semantic)
    reopened = TerminalLedgerV3.open(rig.ledger.path, rig.ledger.binding)
    assert failure.value.evidence_sha256 == reopened.state(rig.keys[0].dispatch_id).event_hash
    assert rig.seen.trace[-1] == "fsync"


def test_runner_recovers_and_skips_failed_parent_requests(rig):
    from memcontam.readiness.phase13_main_runner import run_pending_requests_v3

    rig.seen.count = 379
    with pytest.raises(rig.api.DispatchTechnicalFailureV3):
        rig.dispatcher.dispatch(rig.keys[0], rig.material, rig.semantic)
    rig.seen.count = 378
    resumed = restart(rig)
    completed = run_pending_requests_v3(resumed, rig.keys, lambda key: resumed.dispatch(
        key, rig.material, rig.semantic,
    ))
    assert completed == tuple(key.dispatch_id for key in rig.keys[4:])
    assert rig.seen.requests == 2


def test_same_dispatcher_recovers_published_compile_before_event(rig, monkeypatch):
    original = rig.dispatcher._append

    def append(key, kind, extra=None):
        if kind == "REQUEST_COMPILED":
            raise Crash()
        original(key, kind, extra)

    with monkeypatch.context() as patch:
        patch.setattr(rig.dispatcher, "_append", append)
        with pytest.raises(Crash):
            rig.dispatcher.dispatch(rig.keys[0], rig.material, rig.semantic)
    rig.dispatcher.recover()
    assert rig.dispatcher.dispatch(rig.keys[0], rig.material, rig.semantic) == "final: 24"


def test_compiled_overflow_cannot_recover_as_pending(rig, monkeypatch):
    rig.seen.count = 379
    original = rig.dispatcher._append

    def append(key, kind, extra=None):
        original(key, kind, extra)
        if kind == "REQUEST_COMPILED":
            raise Crash()

    monkeypatch.setattr(rig.dispatcher, "_append", append)
    with pytest.raises(Crash):
        rig.dispatcher.dispatch(rig.keys[0], rig.material, rig.semantic)
    resumed = restart(rig)
    resumed.recover()
    assert rig.ledger.state(rig.keys[0].dispatch_id).kind == "TERMINAL_TECHNICAL_MISSING"
    assert rig.seen.requests == rig.seen.constructors == 0


def test_known_attempted_cost_allows_independent_runner_progress(rig):
    from memcontam.readiness.phase13_main_runner import run_pending_requests_v3

    def semantic(response):
        raise ValueError("synthetic semantic failure with complete usage")

    with pytest.raises(rig.api.DispatchTechnicalFailureV3) as failure:
        rig.dispatcher.dispatch(rig.keys[0], rig.material, semantic)
    assert failure.value.realized_cost_krw == 0
    resumed = restart(rig)
    completed = run_pending_requests_v3(resumed, rig.keys, lambda key: resumed.dispatch(
        key, rig.material, rig.semantic,
    ))
    assert completed == tuple(key.dispatch_id for key in rig.keys[4:])
    assert rig.seen.requests == 3


def test_runner_unknown_cost_blocks_even_callback(rig):
    from memcontam.readiness.phase13_main_runner import run_pending_requests_v3

    rig.seen.outcome = "transport"
    with pytest.raises(rig.api.DispatchTechnicalFailureV3):
        rig.dispatcher.dispatch(rig.keys[0], rig.material, rig.semantic)
    callbacks = []
    with pytest.raises(ValueError, match="MAIN_TERMINAL_COST_UNKNOWN"):
        run_pending_requests_v3(restart(rig), rig.keys, lambda key: callbacks.append(key))
    assert callbacks == []


@pytest.mark.parametrize("ambiguous", [False, True])
def test_cost_reconciliation_unlocks_independent_not_failed_parents(rig, ambiguous):
    def crash(response):
        raise Crash()

    rig.seen.outcome = "ok" if ambiguous else "transport"
    with pytest.raises(Crash if ambiguous else rig.api.DispatchTechnicalFailureV3):
        rig.dispatcher.dispatch(rig.keys[0], rig.material, crash if ambiguous else rig.semantic)
    restart(rig).recover()
    before = rig.ledger.rows()
    state = rig.ledger.state(rig.keys[0].dispatch_id)
    rig.ledger.reconcile_cost(rig.keys[0].dispatch_id, {
        "usage": {"input_tokens": 1, "output_tokens": 1, "cached_input_tokens": 0},
    }, "f" * 64)
    resumed = restart(rig)
    assert rig.ledger.rows()[:len(before)] == before
    assert json.loads(rig.ledger.rows()[-1])["realized_cost_krw"] == 1
    rig.ledger.append(json.loads(rig.ledger.rows()[-1]))
    assert rig.ledger.state(rig.keys[0].dispatch_id).kind == state.kind
    for key in rig.keys[:4]:
        with pytest.raises(rig.api.DispatchTechnicalFailureV3):
            resumed.dispatch(key, rig.material, rig.semantic)
    rig.seen.outcome = "ok"
    assert resumed.dispatch(rig.keys[4], rig.material, rig.semantic) == "final: 24"
    assert rig.seen.requests == 2


@pytest.mark.parametrize("after_sync", [False, True])
@pytest.mark.parametrize("point,expected", [
    ("DISPATCH_INTENT_PERSISTED", "PENDING"), ("REQUEST_COMPILED", "PENDING"),
    ("ATTEMPT_STARTED", "AMBIGUOUS_ATTEMPT"), ("COMPLETED", "COMPLETED"),
])
def test_restart_at_each_ledger_fsync(rig, monkeypatch, point, expected, after_sync):
    original = TerminalLedgerV3._sync

    def sync(ledger):
        if ledger.state(rig.keys[0].dispatch_id).kind == point:
            if after_sync:
                original(ledger)
            raise Crash()
        original(ledger)

    with monkeypatch.context() as patch:
        patch.setattr(TerminalLedgerV3, "_sync", sync)
        with pytest.raises(Crash):
            rig.dispatcher.dispatch(rig.keys[0], rig.material, rig.semantic)
    calls = rig.seen.requests
    resumed = restart(rig)
    resumed.recover()
    assert rig.ledger.state(rig.keys[0].dispatch_id).kind == expected
    assert rig.seen.requests == calls
    for raw in rig.ledger.rows():
        row = json.loads(raw)
        if row["kind"] not in ("DISPATCH_INTENT", "NO_REQUEST"):
            assert row["compiled"]["compiled_request_hash"]


__all__ = ["api", "rig"]
