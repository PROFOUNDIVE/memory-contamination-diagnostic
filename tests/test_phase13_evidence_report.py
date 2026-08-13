from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from memcontam.readiness.phase13_evidence import EvidenceError, verify_evidence_report


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs/evidence/phase13-authority-sync-calibration-v2-v1.json"
PROTECTED_ROOTS = (
    "$tmp_dir/",
    "data/embedding_cache/bfv2-source-contract-replay-test/",
    "data/embedding_cache/bfv2-structural-replay-test/",
    "oh-my-opencode.json",
)
BLOCKERS = (
    "authenticated_structural_checkpoint_authority_incomplete",
    "runtime_archive_cardinality_contract_incompatible",
)
COMMAND_RESULTS = (
    (
        "PYTHONPATH=src python -m memcontam.cli phase13 validate-calibration-v2 --config configs/phase13/pre_main_calibration_v2.yaml",
        0,
        ("DETERMINISTIC_AUTHORITY_SYNC_COMPLETE",),
    ),
    (
        "PYTHONPATH=src python -m memcontam.cli phase13 prepare-calibration-v2 --config configs/phase13/pre_main_calibration_v2.yaml",
        0,
        ("DETERMINISTIC_AUTHORITY_SYNC_COMPLETE", "AWAITING_CALIBRATION_V2_AUTHORIZATION"),
    ),
    (
        "PYTHONPATH=src python -m memcontam.cli phase13 run-calibration-v2 --config configs/phase13/pre_main_calibration_v2.yaml",
        1,
        ("CALIBRATION_V2_EXTERNAL_BLOCK", "MAIN_A_EXECUTION_FORBIDDEN"),
    ),
    (
        "PYTHONPATH=src python -m memcontam.cli phase13 main-a",
        1,
        ("MAIN_A_EXECUTION_FORBIDDEN",),
    ),
)


def _write_report(path: Path, payload: dict[str, object]) -> None:
    unsealed = {key: value for key, value in payload.items() if key != "report_sha256"}
    payload["report_sha256"] = hashlib.sha256(
        json.dumps(unsealed, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _valid_report(tmp_path: Path) -> Path:
    snapshot = {
        "entry_count": 1,
        "content_sha256": "1" * 64,
        "metadata_sha256": "2" * 64,
        "combined_sha256": "3" * 64,
        "untracked": True,
        "staged": False,
    }
    payload: dict[str, object] = {
        "schema_version": "phase13_authority_sync_calibration_v2_evidence_v1",
        "build_terminal": "DETERMINISTIC_AUTHORITY_SYNC_COMPLETE",
        "calibration_terminal": "CALIBRATION_V2_EXTERNAL_BLOCK",
        "main_terminal": "MAIN_A_EXECUTION_FORBIDDEN",
        "provider_calls": 0,
        "calibration_ran": False,
        "scientific_evidence": False,
        "archive_status": "absent",
        "claim_status": "absent",
        "external_blockers": list(BLOCKERS),
        "tracked_artifacts": [
            {
                "path": "configs/phase13/pre_main_calibration_v2.yaml",
                "sha256": hashlib.sha256(
                    (ROOT / "configs/phase13/pre_main_calibration_v2.yaml").read_bytes()
                ).hexdigest(),
            }
        ],
        "command_results": [
            {
                "command": command,
                "exit_code": exit_code,
                "stdout_sha256": hashlib.sha256(
                    ("\n".join(lines) + "\n").encode()
                ).hexdigest(),
                "terminal_lines": list(lines),
            }
            for command, exit_code, lines in COMMAND_RESULTS
        ],
        "protected_dirty_roots": [
            {"path": path, "before": snapshot, "after": snapshot} for path in PROTECTED_ROOTS
        ],
        "main_a_artifact_paths": [],
    }
    report = tmp_path / REPORT.name
    _write_report(report, payload)
    return report


def test_valid_report_reconstructs_tracked_hashes_and_terminal(tmp_path: Path) -> None:
    verified = verify_evidence_report(ROOT, _valid_report(tmp_path))

    assert verified.terminal == "CALIBRATION_V2_EXTERNAL_BLOCK"
    assert verified.provider_calls == 0


def test_copied_report_tampering_fails_closed(tmp_path: Path) -> None:
    report = _valid_report(tmp_path)
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["report_sha256"] = "0" * 64
    report.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(EvidenceError, match="REPORT_HASH_MISMATCH"):
        verify_evidence_report(ROOT, report)


def test_tracked_input_tampering_fails_closed(tmp_path: Path) -> None:
    report = _valid_report(tmp_path)
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["tracked_artifacts"][0]["sha256"] = "0" * 64
    _write_report(report, payload)

    with pytest.raises(EvidenceError, match="TRACKED_ARTIFACT_HASH_MISMATCH"):
        verify_evidence_report(ROOT, report)


def test_dirty_snapshot_mismatch_fails_closed(tmp_path: Path) -> None:
    report = _valid_report(tmp_path)
    payload = json.loads(report.read_text(encoding="utf-8"))
    roots = payload["protected_dirty_roots"]
    assert isinstance(roots, list)
    root = roots[0]
    assert isinstance(root, dict)
    root["after"]["combined_sha256"] = "0" * 64
    _write_report(report, payload)

    with pytest.raises(EvidenceError, match="PROTECTED_DIRTY_STATE_MISMATCH"):
        verify_evidence_report(ROOT, report)


def test_main_a_artifact_injection_fails_closed(tmp_path: Path) -> None:
    report = _valid_report(tmp_path)
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["main_a_artifact_paths"] = ["runs/phase13-main-a/request.json"]
    _write_report(report, payload)

    with pytest.raises(EvidenceError, match="MAIN_A_ARTIFACT_FORBIDDEN"):
        verify_evidence_report(ROOT, report)


def test_external_block_cannot_be_relabelled_completed(tmp_path: Path) -> None:
    report = _valid_report(tmp_path)
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["calibration_terminal"] = "CALIBRATION_V2_COMPLETED"
    _write_report(report, payload)

    with pytest.raises(EvidenceError, match="TERMINAL_INTERPRETATION_INVALID"):
        verify_evidence_report(ROOT, report)


@pytest.mark.parametrize(
    "blockers",
    [
        list(reversed(BLOCKERS)),
        ["operator_runtime_capacities_not_issued", BLOCKERS[1]],
        ["native_production_dispatch_orchestration_unavailable", BLOCKERS[0]],
        [BLOCKERS[0]],
    ],
)
def test_external_blockers_are_exact_and_ordered(tmp_path: Path, blockers: list[str]) -> None:
    report = _valid_report(tmp_path)
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["external_blockers"] = blockers
    _write_report(report, payload)

    with pytest.raises(EvidenceError, match="EXTERNAL_BLOCKERS_INVALID"):
        verify_evidence_report(ROOT, report)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("provider_calls", 1),
        ("calibration_ran", True),
        ("archive_status", "present"),
        ("claim_status", "present"),
        ("scientific_evidence", True),
    ],
)
def test_live_calibration_or_evidence_claim_fails_closed(
    tmp_path: Path, field: str, value: str | int | bool
) -> None:
    report = _valid_report(tmp_path)
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload[field] = value
    _write_report(report, payload)

    with pytest.raises(EvidenceError, match="EVIDENCE_REPORT_INVALID"):
        verify_evidence_report(ROOT, report)


def test_cli_observation_mutation_fails_closed(tmp_path: Path) -> None:
    report = _valid_report(tmp_path)
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["command_results"][2]["terminal_lines"] = ["CALIBRATION_V2_COMPLETED"]
    _write_report(report, payload)

    with pytest.raises(EvidenceError, match="COMMAND_OBSERVATION_INVALID"):
        verify_evidence_report(ROOT, report)


@pytest.mark.parametrize(
    "private_path",
    [
        ".env",
        ".omo/operator/request.json",
        "credentials/provider.json",
        "/private/cache/model",
        "../authorization.json",
    ],
)
def test_private_or_untracked_artifact_path_fails_closed(
    tmp_path: Path, private_path: str
) -> None:
    report = _valid_report(tmp_path)
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["tracked_artifacts"][0]["path"] = private_path
    _write_report(report, payload)

    with pytest.raises(EvidenceError, match="TRACKED_ARTIFACT_PATH_INVALID"):
        verify_evidence_report(ROOT, report)
