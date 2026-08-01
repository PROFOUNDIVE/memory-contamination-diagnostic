from __future__ import annotations

import json
import shutil
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import pytest

from memcontam.experiment.phase12.filter_challenge.bct_archive import validate_evidence_bundle
from memcontam.experiment.phase12.filter_challenge.evidence_contract import (
    approval_descriptor_path,
    approved_plan_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "docs" / "evidence" / "phase12-filter-v5-bct-v1"
PLAN = ROOT / ".omo" / "plans" / "phase12-post-filter-v5-calibration-readiness.md"
VERIFY = ROOT / "scripts" / "verify_phase12_filter_v5_bct_evidence.py"


def _verify(bundle: Path, cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(VERIFY),
            "--through",
            "readiness",
            "--bundle",
            str(bundle),
            "--plan",
            str(PLAN),
            "--artifact-root",
            str(ROOT / "runs" / "phase12-filter-v5-bct-live-v1"),
        ],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def _copied_bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "bundle"
    shutil.copytree(BUNDLE, bundle)
    return bundle


def _copied_repository(tmp_path: Path) -> tuple[Path, Path]:
    repository = tmp_path / "repository"
    bundle = repository / "docs/evidence/phase12-filter-v5-bct-v1"
    bundle.parent.mkdir(parents=True)
    shutil.copytree(BUNDLE, bundle)
    source_universe = json.loads(
        (ROOT / "data/phase12/filter_v5_bct_v1/source_universe_v1.json").read_text(
            encoding="utf-8"
        )
    )
    source_files = source_universe["source_files"]
    assert isinstance(source_files, dict)
    paths = (
        *source_files,
        "data/phase12/filter_v5_bct_v1/source_universe_v1.json",
        "data/phase12/filter_v5_bct_v1/freeze_a.json",
        "data/phase12/filter_v5_bct_v1/screening_authorization_request.json",
        "configs/phase12/filter_v5_bct_calibration.yaml",
        "configs/phase12/exploratory_code_source_fidelity_v2.yaml",
        "containers/python-sandbox/image.lock.json",
        "docs/evidence/phase12-filter-v5-bct-v1/authority_transition_manifest.json",
        "docs/phase12-filter-v5-bct-methods-lock.md",
        ".omo/evidence/phase12-post-filter-v5-calibration-readiness/task-3-screening-stage-result.json",
        ".omo/evidence/phase12-post-filter-v5-calibration-readiness/task-5-bct-stage-result.json",
        ".omo/evidence/phase12-post-filter-v5-calibration-readiness/task-6-pilot-b-readiness-stage-result.json",
    )
    for relative_path in paths:
        assert isinstance(relative_path, str)
        destination = repository / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative_path, destination)
    return repository, bundle


def _write_report(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def _write_sealed_report(path: Path, payload: dict[str, object]) -> None:
    payload["output_seal"] = sha256(
        json.dumps(
            {key: value for key, value in payload.items() if key != "output_seal"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    _write_report(path, payload)


def _reseal_backward_hashes(bundle: Path) -> None:
    report_ids = (
        "authority-transition",
        "methods-lock",
        "freeze-a",
        "screening",
        "freeze-b-search-config",
        "bct-execution",
        "archive-validation",
        "claim-scope",
    )
    for index, report_id in enumerate(report_ids[4:], start=4):
        path = bundle / f"{report_id.replace('-', '_')}_report.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        upstream_ids = report_ids[:4] if report_id == "freeze-b-search-config" else report_ids[:5]
        payload["upstream_report_sha256"] = {
            prior: sha256((bundle / f"{prior.replace('-', '_')}_report.json").read_bytes()).hexdigest()
            for prior in upstream_ids
        }
        _write_report(path, payload)
    readiness = bundle / "pilot_b_readiness_report.json"
    payload = json.loads(readiness.read_text(encoding="utf-8"))
    payload["prior_report_sha256"] = {
        report_id: sha256((bundle / f"{report_id.replace('-', '_')}_report.json").read_bytes()).hexdigest()
        for report_id in report_ids
    }
    _write_report(readiness, payload)


def test_readiness_verifier_rejects_report_one_provider_counter_reseal(tmp_path: Path) -> None:
    # Given: a copied report chain with a false paid-call counter in report 1.
    bundle = _copied_bundle(tmp_path)
    report = bundle / "authority_transition_report.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["provider_calls_issued"] = 1
    _write_report(report, payload)
    _reseal_backward_hashes(bundle)

    # When: the final readiness verifier examines the modified chain.
    result = _verify(bundle)

    # Then: a downstream-resealed self-assertion cannot turn a zero-call branch into success.
    assert result.returncode != 0
    assert result.stdout == "EVIDENCE_REPORT_CONTRACT_INVALID\n"


def test_readiness_verifier_rejects_misleading_report_nine_success_fields(tmp_path: Path) -> None:
    # Given: the raw readiness stage remains waiting but its report claims completion.
    bundle = _copied_bundle(tmp_path)
    report = bundle / "pilot_b_readiness_report.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["stage_disposition"] = "completed"
    payload["all_passed"] = True
    _write_report(report, payload)

    # When: the final verifier recomputes report 9 from the stage result.
    result = _verify(bundle)

    # Then: undeclared success fields and a mismatched disposition are rejected.
    assert result.returncode != 0
    assert result.stdout == "EVIDENCE_REPORT_CONTRACT_INVALID\n"


def test_readiness_verifier_rejects_symlinked_report_substitution(tmp_path: Path) -> None:
    # Given: a required report path is replaced by a symlink to an otherwise valid report.
    bundle = _copied_bundle(tmp_path)
    report = bundle / "claim_scope_report.json"
    report.unlink()
    report.symlink_to(bundle / "archive_validation_report.json")

    # When: the final verifier opens every report.
    result = _verify(bundle)

    # Then: descriptor no-follow opening rejects the substitution.
    assert result.returncode != 0
    assert result.stdout == "EVIDENCE_REPORT_INVALID\n"


def test_evidence_validator_rejects_current_source_universe_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "data" / "phase12" / "filter_v5_bct_v1" / "source_universe_v1.json"
    source.parent.mkdir(parents=True)
    source.write_bytes((ROOT / source.relative_to(tmp_path)).read_bytes().replace(b"7a7a39", b"000000", 1))
    monkeypatch.chdir(tmp_path)
    digest = approved_plan_sha256(PLAN, approval_descriptor_path(PLAN))
    result = validate_evidence_bundle(_copied_bundle(tmp_path), digest, "freeze-a")
    assert result.valid is False
    assert result.reason_code == "EVIDENCE_SOURCE_UNIVERSE_INVALID"


@pytest.mark.parametrize(
    ("relative_path", "expected", "replacement"),
    (
        (
            "configs/phase12/filter_v5_bct_calibration.yaml",
            b"max_output_tokens: 640",
            b"max_output_tokens: 641",
        ),
        (
            "data/phase12/filter_v5_bct_v1/freeze_a.json",
            b'"approved_plan_sha256":"e8d44600fb3a9177ae691fd8f49ac1c06305b004db7ccd50d391c9876356a230"',
            b'"approved_plan_sha256":"0000000000000000000000000000000000000000000000000000000000000000"',
        ),
    ),
)
def test_readiness_verifier_rejects_current_frozen_input_mutation(
    tmp_path: Path, relative_path: str, expected: bytes, replacement: bytes
) -> None:
    # Given: a complete isolated repository with one report-bound frozen input changed.
    repository, bundle = _copied_repository(tmp_path)
    path = repository / relative_path
    path.write_bytes(path.read_bytes().replace(expected, replacement, 1))

    # When: the fresh readiness verifier resolves all paths from the isolated root.
    result = _verify(bundle, repository)

    # Then: the changed current input cannot retain the prior report's approval.
    assert result.returncode != 0
    assert result.stdout == "EVIDENCE_FROZEN_INPUT_INVALID\n"


def test_readiness_verifier_rejects_source_universe_member_mutation(tmp_path: Path) -> None:
    # Given: a source-universe member changed after the source-universe report was sealed.
    repository, bundle = _copied_repository(tmp_path)
    source = repository / "data/tasks/game24_pilot.jsonl"
    source.write_bytes(source.read_bytes() + b"\n")

    # When: the fresh readiness verifier reads the isolated repository.
    result = _verify(bundle, repository)

    # Then: the source-universe's declared member digest is recomputed and rejects it.
    assert result.returncode != 0
    assert result.stdout == "EVIDENCE_SOURCE_UNIVERSE_INVALID\n"


def test_readiness_verifier_reports_resealed_terminal_tampering_with_a_code(tmp_path: Path) -> None:
    # Given: a resealed report that claims a terminal different from its raw stage result.
    bundle = _copied_bundle(tmp_path)
    report = bundle / "pilot_b_readiness_report.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["terminal_status"] = "READY_FOR_SEPARATE_FILTER_V5_PILOT_B_AUTHORIZATION"
    _write_sealed_report(report, payload)

    # When: the command-line verifier validates readiness.
    result = _verify(bundle)

    # Then: the caller receives the deterministic readiness rejection code.
    assert result.returncode != 0
    assert result.stdout == "EVIDENCE_READINESS_INVALID\n"
