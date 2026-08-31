from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

import memcontam.readiness.phase13_main_runner as runner_module
from memcontam.readiness.phase13_main_runner import (
    DispatchCompleted,
    MainRunError,
    MainRunRequest,
    open_main_run,
    prepare_main_run,
    resume_main,
    run_main,
)


ROOT = Path(__file__).resolve().parents[1]
P5 = ROOT / "data/phase13/main/mr_p5/execution_package_v1.json"
P6 = ROOT / "data/phase13/main/mr_p6/authorized_execution_v1.json"


def _authorization_sha256() -> str:
    return hashlib.sha256(P6.read_bytes()).hexdigest()


def _request(tmp_path: Path, expected_sha256: str | None = None) -> MainRunRequest:
    return MainRunRequest(
        repository_root=ROOT,
        package_path=P5,
        authorization_path=P6,
        expected_authorization_sha256=expected_sha256 or _authorization_sha256(),
        run_root=tmp_path,
        run_id="offline-qa",
    )


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        (sys.executable, "-m", "memcontam.readiness.phase13_main_runner_cli", *arguments),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def _common(tmp_path: Path) -> tuple[str, ...]:
    return (
        "--repository-root",
        str(ROOT),
        "--package",
        str(P5),
        "--authorization",
        str(P6),
        "--expected-authorization-sha256",
        _authorization_sha256(),
        "--run-root",
        str(tmp_path),
        "--run-id",
        "offline-qa",
    )


def test_authorized_run_creation_binds_exact_frozen_inputs(tmp_path: Path) -> None:
    ledger = prepare_main_run(_request(tmp_path))

    status = ledger.status()
    assert status.session_state == "NOT_STARTED"
    assert status.total_count == 970
    assert status.pending_count == 970


def test_authorized_run_reopen_revalidates_package_and_authorization(tmp_path: Path) -> None:
    prepare_main_run(_request(tmp_path))

    reopened = open_main_run(_request(tmp_path))

    assert reopened.status().total_count == 970


def test_authorized_run_and_resume_dispatch_distinct_pending_units(tmp_path: Path) -> None:
    calls: list[str] = []

    def dispatch(unit) -> DispatchCompleted:
        calls.append(unit.unit_id)
        return DispatchCompleted(hashlib.sha256(unit.unit_id.encode()).hexdigest(), 1)

    run_main(
        _request(tmp_path),
        dispatch,
        projected_cost_krw=lambda _unit: 1,
        tranche_ceiling_krw=10,
        max_units=1,
    )
    resume_main(
        _request(tmp_path),
        dispatch,
        projected_cost_krw=lambda _unit: 1,
        tranche_ceiling_krw=10,
        max_units=1,
    )

    assert len(calls) == len(set(calls)) == 2


def test_authorized_run_rejects_authorization_hash_tampering(tmp_path: Path) -> None:
    with pytest.raises(MainRunError, match="MAIN_AUTHORIZATION_FILE_HASH_MISMATCH"):
        prepare_main_run(_request(tmp_path, "0" * 64))


def test_authorized_run_rejects_package_changed_after_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed = json.loads(P5.read_text())
    changed["dispatch"]["task_order"] = list(reversed(changed["dispatch"]["task_order"]))
    changed_raw = json.dumps(changed).encode()
    monkeypatch.setattr(runner_module, "read_regular_nofollow", lambda _path: changed_raw)

    with pytest.raises(MainRunError, match="MAIN_RUN_PACKAGE_BYTES_CHANGED"):
        prepare_main_run(_request(tmp_path))


def test_frozen_runner_binds_direct_authorization_trust_base() -> None:
    package = json.loads(P5.read_text())
    roles = {binding["role"] for binding in package["artifacts"]}

    assert {
        "main_execution",
        "main_execution_models",
        "main_execution_bindings",
    } <= roles


def test_main_run_id_must_be_one_component(tmp_path: Path) -> None:
    request = _request(tmp_path)
    changed = MainRunRequest(
        request.repository_root,
        request.package_path,
        request.authorization_path,
        request.expected_authorization_sha256,
        request.run_root,
        "../escape",
    )

    with pytest.raises(MainRunError, match="MAIN_RUN_ID_INVALID"):
        prepare_main_run(changed)


def test_phase13_help_exposes_main_execution_control_surface() -> None:
    result = _run("--help")

    assert result.returncode == 0
    assert "run" in result.stdout
    assert "status" in result.stdout
    assert "resume" in result.stdout


def test_main_cli_run_status_resume_are_offline_and_stable(tmp_path: Path) -> None:
    started = _run("run", *_common(tmp_path))
    status = _run("status", *_common(tmp_path))
    resumed = _run("resume", *_common(tmp_path))

    assert started.returncode == status.returncode == resumed.returncode == 0
    reports = tuple(json.loads(result.stdout) for result in (started, status, resumed))
    assert {report["session_state"] for report in reports} == {"NOT_STARTED"}
    assert {report["total_count"] for report in reports} == {970}
    assert {report["pending_count"] for report in reports} == {970}


def test_main_cli_reports_bad_authorization_without_traceback(tmp_path: Path) -> None:
    arguments = list(_common(tmp_path))
    hash_index = arguments.index("--expected-authorization-sha256") + 1
    arguments[hash_index] = "0" * 64

    result = _run("run", *arguments)

    assert result.returncode != 0
    assert "MAIN_AUTHORIZATION_FILE_HASH_MISMATCH" in result.stderr
    assert "Traceback" not in result.stderr
