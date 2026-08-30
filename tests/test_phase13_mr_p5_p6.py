from __future__ import annotations

import json
import hashlib
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
P5 = ROOT / "data/phase13/main/mr_p5/execution_package_v1.json"
P6 = ROOT / "data/phase13/main/mr_p6/authorized_execution_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "memcontam.readiness.phase13_main_execution_cli", *arguments],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_main_execution_freeze_cli_validates_exact_package() -> None:
    result = _run(
        "validate-freeze",
        "--repository-root",
        str(ROOT),
        "--package",
        str(P5),
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "FROZEN"
    assert report["mr_p4_status"] == "CLOSED"
    assert report["mr_p5_status"] == "FROZEN"
    assert report["measured_main_a_trajectory_count"] == 0
    assert len(report["package_sha256"]) == 64


def test_main_authorization_cli_binds_exact_frozen_package() -> None:
    freeze = _run(
        "validate-freeze",
        "--repository-root",
        str(ROOT),
        "--package",
        str(P5),
    )
    authorization = _run(
        "validate-authorization",
        "--repository-root",
        str(ROOT),
        "--package",
        str(P5),
        "--authorization",
        str(P6),
        "--expected-authorization-sha256",
        _sha256(P6),
    )

    assert freeze.returncode == authorization.returncode == 0
    freeze_report = json.loads(freeze.stdout)
    authorization_report = json.loads(authorization.stdout)
    assert authorization_report["status"] == "AUTHORIZED_EXECUTION"
    assert authorization_report["mr_p6_status"] == "PASS"
    assert authorization_report["execution_package_sha256"] == freeze_report["package_sha256"]
    assert authorization_report["main_a_status"] == "NOT_STARTED"
    assert authorization_report["measured_main_a_trajectory_count"] == 0


def test_main_authorization_rejects_package_drift(tmp_path: Path) -> None:
    package = json.loads(P5.read_text())
    package["H_run"] = 49
    changed = tmp_path / "execution_package_v1.json"
    changed.write_text(json.dumps(package))

    result = _run(
        "validate-authorization",
        "--repository-root",
        str(ROOT),
        "--package",
        str(changed),
        "--authorization",
        str(P6),
        "--expected-authorization-sha256",
        _sha256(P6),
    )

    assert result.returncode != 0
    assert "MAIN_EXECUTION_PACKAGE_INVALID" in result.stderr


def test_main_freeze_rejects_rehashed_observability_detachment(tmp_path: Path) -> None:
    package = json.loads(P5.read_text())
    package["observability"]["packet_sha256"] = "0" * 64
    payload = {key: value for key, value in package.items() if key != "package_hash"}
    package["package_hash"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    changed = tmp_path / "execution_package_v1.json"
    changed.write_text(json.dumps(package))

    result = _run(
        "validate-freeze",
        "--repository-root",
        str(ROOT),
        "--package",
        str(changed),
    )

    assert result.returncode != 0
    assert "MAIN_EXECUTION_OBSERVABILITY_BINDING_INVALID" in result.stderr


def test_main_freeze_rejects_rehashed_artifact_role_detachment(tmp_path: Path) -> None:
    package = json.loads(P5.read_text())
    binding = next(row for row in package["artifacts"] if row["role"] == "task_seed_orders")
    binding["path"] = "README.md"
    binding["sha256"] = _sha256(ROOT / "README.md")
    payload = {key: value for key, value in package.items() if key != "package_hash"}
    package["package_hash"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    changed = tmp_path / "execution_package_v1.json"
    changed.write_text(json.dumps(package))

    result = _run(
        "validate-freeze",
        "--repository-root",
        str(ROOT),
        "--package",
        str(changed),
    )

    assert result.returncode != 0
    assert "MAIN_EXECUTION_ARTIFACT_PATH_INVALID" in result.stderr


def test_main_authorization_requires_expected_raw_hash() -> None:
    result = _run(
        "validate-authorization",
        "--repository-root",
        str(ROOT),
        "--package",
        str(P5),
        "--authorization",
        str(P6),
        "--expected-authorization-sha256",
        "0" * 64,
    )

    assert result.returncode != 0
    assert "MAIN_AUTHORIZATION_FILE_HASH_MISMATCH" in result.stderr


def test_main_freeze_rejects_rehashed_duplicate_authority(tmp_path: Path) -> None:
    package = json.loads(P5.read_text())
    package["authority"].insert(
        0,
        {"role": "authority_router", "sha256": "0" * 64},
    )
    payload = {key: value for key, value in package.items() if key != "package_hash"}
    package["package_hash"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    changed = tmp_path / "execution_package_v1.json"
    changed.write_text(json.dumps(package))

    result = _run(
        "validate-freeze",
        "--repository-root",
        str(ROOT),
        "--package",
        str(changed),
    )

    assert result.returncode != 0
    assert "MAIN_EXECUTION_AUTHORITY_MISMATCH" in result.stderr
