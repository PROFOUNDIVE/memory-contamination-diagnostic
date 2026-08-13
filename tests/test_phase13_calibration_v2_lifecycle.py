from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from memcontam.readiness import phase13_cli
from memcontam.readiness.phase13_calibration_v2_lifecycle import (
    AuthorizedLifecycleEvidence,
    LifecycleInvocation,
    SourceTrajectoryEvidence,
    run_calibration_v2_lifecycle,
)
from .test_phase13_calibration_v2_authorization import CONFIG, NOW, _verify, complete_bundle


TASKS = ("game24", "math_equation_balancer", "word_sorting")
SEEDS = tuple(range(10000, 10012))


def _invocation(tmp_path: Path) -> LifecycleInvocation:
    return LifecycleInvocation(
        config_path=CONFIG,
        report_path=tmp_path / "calibration-v2-report.json",
        request_path=None,
        authorization_path=None,
        expected_authorization_sha256=None,
        allow_live_calls=False,
        environment={},
        now=NOW,
    )


def _source(task: str, seed: int) -> SourceTrajectoryEvidence:
    calls = tuple(f"{task}-{seed}-call-{index}" for index in range(250))
    return SourceTrajectoryEvidence(
        source_run_id=f"{task}-seed-{seed}",
        task=task,
        seed_id=seed,
        event_times=tuple(range(10)),
        state_lineage=tuple((f"state-{index}", f"state-{index + 1}") for index in range(10)),
        source_execution_count=1,
        accounting_status="closed_complete",
        provider_owner_id="phase13-h10-execution-owner-v1",
        offline_owner_id="phase13-offline-compute-owner-v1",
        provider_call_ids=calls,
        settled_call_ids=calls,
        transport_attempt_ids=tuple(f"transport-{call_id}" for call_id in calls),
        short_window_provider_calls=0,
        derived_window_executions=0,
        archive_valid=True,
        claim_status="synthetic_qa_only",
    )


def _authorized_evidence() -> AuthorizedLifecycleEvidence:
    return AuthorizedLifecycleEvidence(
        sources=tuple(_source(task, seed) for task in TASKS for seed in SEEDS),
        support_successes=tuple((task, 12) for task in TASKS),
        attempted_seeds=tuple((task, 13) for task in TASKS),
        observed_input_tokens=9_000,
        observed_output_tokens=4_500,
        observed_cost_microusd=36_000,
        observed_latency_milliseconds=3_600,
        observed_storage_bytes=72_000,
        observed_wall_clock_seconds=36,
    )


def test_external_block_is_sealed_provider_free_and_creates_no_run_root(
    tmp_path: Path,
) -> None:
    constructions = dispatches = 0

    def forbidden_factory() -> AuthorizedLifecycleEvidence:
        nonlocal constructions
        constructions += 1
        raise AssertionError("provider construction forbidden")

    invocation = _invocation(tmp_path)
    run_root = CONFIG.parents[2] / "runs/phase13-calibration-v2"

    report = run_calibration_v2_lifecycle(invocation, forbidden_factory)

    payload = json.loads(invocation.report_path.read_text(encoding="utf-8"))
    sealed = dict(payload)
    report_hash = sealed.pop("report_sha256")
    assert report.terminal == "CALIBRATION_V2_EXTERNAL_BLOCK"
    assert report.main_terminal == "MAIN_A_EXECUTION_FORBIDDEN"
    assert payload["deterministic_evidence"]["validate"] == "DETERMINISTIC_AUTHORITY_SYNC_COMPLETE"
    assert payload["deterministic_evidence"]["prepare"] == "AWAITING_CALIBRATION_V2_AUTHORIZATION"
    assert set(payload["unavailable_fields"]) >= {
        "authorization_path",
        "authorization_sha256",
        "credential:OPENAI_API_KEY",
        "cache_runtime_access",
        "operator_capacity:maximum_cost_microusd",
        "operator_capacity:provider_concurrency",
    }
    assert report_hash == hashlib.sha256(
        json.dumps(sealed, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert payload["provider_construction_count"] == payload["provider_dispatch_count"] == 0
    assert payload["run_root"] is None
    assert payload["archive_status"] == payload["claim_status"] == "absent"
    assert constructions == dispatches == 0
    assert not run_root.exists()


def test_cli_external_block_revalidation_does_not_create_run_root(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report = tmp_path / "external-block.json"
    args = argparse.Namespace(
        phase13_command="run-calibration-v2",
        config=CONFIG,
        report=report,
        request=None,
        authorization=None,
        expected_authorization_sha256=None,
        allow_live_calls=False,
    )

    for _ in range(2):
        phase13_cli.run(args)

    lines = capsys.readouterr().out.splitlines()
    assert sum(line == "CALIBRATION_V2_EXTERNAL_BLOCK" for line in lines) == 2
    assert sum(line == "MAIN_A_EXECUTION_FORBIDDEN" for line in lines) == 2
    assert json.loads(report.read_text(encoding="utf-8"))["run_root"] is None


def test_unbound_synthetic_batch_cannot_emit_completed_or_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, permit, digest, _ = complete_bundle(tmp_path)
    verified = _verify(tmp_path, monkeypatch)
    constructions = 0

    def factory() -> AuthorizedLifecycleEvidence:
        nonlocal constructions
        constructions += 1
        return _authorized_evidence()

    invocation = replace(
        _invocation(tmp_path),
        request_path=request,
        authorization_path=permit,
        expected_authorization_sha256=digest,
        allow_live_calls=True,
        environment={"OPENAI_API_KEY": "synthetic", "MEMCONTAM_BGE_CACHE_DIR": "/synthetic"},
    )
    monkeypatch.setattr(
        "memcontam.readiness.phase13_calibration_v2_lifecycle.verify_calibration_v2_authorization",
        lambda **_kwargs: verified,
    )

    report = run_calibration_v2_lifecycle(invocation, factory)

    assert report.terminal == "CALIBRATION_V2_INVALIDATED"
    assert report.reason == "UNBOUND_SYNTHETIC_QA_EVIDENCE"
    assert report.main_terminal == "MAIN_A_EXECUTION_FORBIDDEN"
    assert report.synthetic_qa_only is True
    assert report.trajectory_count == 36
    assert report.tasks == TASKS
    assert report.seeds == SEEDS
    assert report.events_per_source == 10
    assert report.source_execution_count == 36
    assert report.short_window_provider_calls == 0
    assert report.derived_window_executions == 0
    assert report.settled_semantic_calls == 9_000
    assert report.provider_owner_id == "phase13-h10-execution-owner-v1"
    assert report.offline_owner_id == "phase13-offline-compute-owner-v1"
    assert report.archive_valid is False
    assert report.archive_status == "invalid"
    assert report.claim_status == "absent"
    assert report.support_cp95 == tuple((task, "0.779") for task in TASKS)
    assert report.attempted_seeds == tuple((task, 13) for task in TASKS)
    assert report.observed_resources.maximum_input_tokens == 9_000
    assert report.capacity_ceilings.maximum_input_tokens == 234_733_568
    assert report.observed_resources.maximum_input_tokens != report.capacity_ceilings.maximum_input_tokens
    assert constructions == 1


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("observed_input_tokens", -1, "OBSERVED_RESOURCE_INVALID"),
        ("observed_output_tokens", True, "OBSERVED_RESOURCE_INVALID"),
        ("observed_input_tokens", 234_733_568, "CAPACITY_CEILING_REACHED"),
        ("observed_input_tokens", 234_733_569, "CAPACITY_OVERRUN"),
    ],
)
def test_observed_resources_are_nonnegative_integers_strictly_below_ceiling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: int | bool,
    code: str,
) -> None:
    request, permit, digest, _ = complete_bundle(tmp_path)
    verified = _verify(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "memcontam.readiness.phase13_calibration_v2_lifecycle.verify_calibration_v2_authorization",
        lambda **_kwargs: verified,
    )
    invocation = replace(
        _invocation(tmp_path), request_path=request, authorization_path=permit,
        expected_authorization_sha256=digest, allow_live_calls=True,
        environment={"OPENAI_API_KEY": "synthetic", "MEMCONTAM_BGE_CACHE_DIR": "/synthetic"},
    )

    report = run_calibration_v2_lifecycle(
        invocation, lambda: replace(_authorized_evidence(), **{field: value})
    )

    assert report.terminal == "CALIBRATION_V2_INVALIDATED"
    assert report.reason == code
    assert report.claim_status == "absent"


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        (lambda rows: replace(rows, sources=rows.sources[:-1]), "TRAJECTORY_INVENTORY_INVALID"),
        (lambda rows: replace(rows, sources=(replace(rows.sources[0], event_times=tuple(range(9))), *rows.sources[1:])), "SOURCE_H10_INCOMPLETE"),
        (lambda rows: replace(rows, sources=(replace(rows.sources[0], settled_call_ids=rows.sources[0].settled_call_ids[:-1]), *rows.sources[1:])), "OWNER_SETTLEMENT_INCOMPLETE"),
        (lambda rows: replace(rows, sources=(replace(rows.sources[0], provider_call_ids=(rows.sources[0].provider_call_ids[0], *rows.sources[0].provider_call_ids[:-1])), *rows.sources[1:])), "OWNER_SETTLEMENT_INCOMPLETE"),
        (lambda rows: replace(rows, sources=(replace(rows.sources[0], provider_owner_id="phase13-offline-compute-owner-v1"), *rows.sources[1:])), "OWNER_RECONCILIATION_INVALID"),
        (lambda rows: replace(rows, sources=(replace(rows.sources[0], derived_window_executions=1), *rows.sources[1:])), "SHORT_WINDOW_EXECUTION_FORBIDDEN"),
        (lambda rows: replace(rows, sources=(replace(rows.sources[0], seed_id=99999), *rows.sources[1:])), "TRAJECTORY_SEED_INVENTORY_INVALID"),
        (lambda rows: replace(rows, sources=(replace(rows.sources[0], archive_valid=False), *rows.sources[1:])), "ARCHIVE_INVALID"),
    ],
)
def test_authorized_mutations_invalidate_without_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation,
    code: str,
) -> None:
    request, permit, digest, _ = complete_bundle(tmp_path)
    verified = _verify(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "memcontam.readiness.phase13_calibration_v2_lifecycle.verify_calibration_v2_authorization",
        lambda **_kwargs: verified,
    )
    invocation = replace(
        _invocation(tmp_path), request_path=request, authorization_path=permit,
        expected_authorization_sha256=digest, allow_live_calls=True,
        environment={"OPENAI_API_KEY": "synthetic", "MEMCONTAM_BGE_CACHE_DIR": "/synthetic"},
    )

    report = run_calibration_v2_lifecycle(
        invocation, lambda: mutation(_authorized_evidence())
    )

    assert report.terminal == "CALIBRATION_V2_INVALIDATED"
    assert report.reason == code
    assert report.claim_status == "absent"
    assert report.main_terminal == "MAIN_A_EXECUTION_FORBIDDEN"
    assert invocation.report_path.exists()
