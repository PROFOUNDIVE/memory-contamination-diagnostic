from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import memcontam.cli as root_cli
import memcontam.clients.factory as client_factory
from memcontam.experiment.phase12.filter_challenge.mft_state_models import JsonValue


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "phase12" / "filter_v5"
SEARCH = FIXTURES / "FilterChallengeSearchConfig.yaml"
POLICY = FIXTURES / "FilterChallengeSelectedPolicy.yaml"
PREREQUISITES = FIXTURES / "bct_execution_prerequisites.json"
COMMIT = "a" * 40
SEARCH_HASH = "6883cd37e997ae041fe9151bcb228d5530ed4cc7fca95813a06bd94cbe6c7eae"
MFT_IDS = (
    "MFT-FV5-01-PAIR-MATCH", "MFT-FV5-02-EXPOSURE-REQUIRED",
    "MFT-FV5-03-TRISTATE", "MFT-FV5-04-FAIL-OPEN",
    "MFT-FV5-05-ROUTE-INVARIANCE", "MFT-FV5-06-SCRIPTED-CORRECT",
    "MFT-FV5-07-SCRIPTED-IRRELEVANT", "MFT-FV5-08-NO-WRITEBACK",
    "MFT-FV5-09-CONTAM-SHADOW-SHARE", "MFT-FV5-10-PARSER-BOUNDARY",
    "MFT-FV5-11-CONTROL-CACHE", "MFT-FV5-12-PROBE-KEY-INVARIANCE",
    "MFT-FV5-13-ANSWER-CALL-PROVENANCE", "MFT-FV5-14-ACTIVATION-DOMAIN",
    "MFT-FV5-15-ELIGIBILITY-STATES", "MFT-FV5-16-COVERAGE-NOT-ESTIMABLE",
)
COMMANDS = (
    "validate-search-config", "validate-selected-policy", "mft", "build-archive",
    "validate-archive", "cost-preview", "bct-readiness",
)


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "memcontam.cli", "phase12", "filter-v5", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _payload(path: Path) -> dict[str, JsonValue]:
    text = path.read_text(encoding="utf-8")
    payload = json.loads(text)
    assert text == json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    return payload


def _archive_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_filter_v5_help_exposes_only_exact_offline_commands() -> None:
    result = _run("--help")

    assert result.returncode == 0
    assert all(command in result.stdout for command in COMMANDS)
    assert "bct-run" not in result.stdout
    assert "pilot-b" not in result.stdout
    assert "main" not in result.stdout


def test_validation_commands_write_hash_bound_canonical_reports(tmp_path: Path) -> None:
    search_output = tmp_path / "search.json"
    result = _run(
        "validate-search-config", "--config", str(SEARCH), "--output", str(search_output)
    )

    assert result.returncode == 0, result.stderr
    search = _payload(search_output)
    assert search["valid"] is True
    assert search["search_config_hash"] == SEARCH_HASH
    assert search["fixture_only"] is True
    assert search["provider_calls_issued"] == 0

    for stage, required in (("build", False), ("pilot_b", False), ("main", True)):
        output = tmp_path / f"selected-{stage}.json"
        result = _run(
            "validate-selected-policy",
            "--search-config", str(SEARCH),
            "--selected-policy", str(POLICY),
            "--stage", stage,
            "--output", str(output),
        )
        assert result.returncode == 0, result.stderr
        report = _payload(output)
        assert report["selected_policy_required"] is required
        assert report["selected_policy_reference_valid"] is True
        assert report["execution_authorized"] is False


def test_mft_and_cost_preview_are_exact_deterministic_zero_call_reports(tmp_path: Path) -> None:
    mft_output = tmp_path / "mft.json"
    result = _run(
        "mft",
        "--search-config", str(SEARCH),
        "--fixture-root", str(FIXTURES),
        "--output", str(mft_output),
    )

    assert result.returncode == 0, result.stderr
    mft = _payload(mft_output)
    ordered_ids = mft["ordered_test_ids"]
    counts = mft["execution_counts"]
    assert isinstance(ordered_ids, list)
    assert isinstance(counts, list)
    assert tuple(ordered_ids) == MFT_IDS
    observed_counts = []
    for item in counts:
        assert isinstance(item, dict)
        observed_counts.append((item["test_id"], item["count"]))
    assert observed_counts == [(test_id, 1) for test_id in MFT_IDS]
    assert mft["all_passed"] is True
    assert mft["fixture_only"] is True
    assert mft["provider_calls_issued"] == 0

    cost_output = tmp_path / "cost.json"
    result = _run(
        "cost-preview", "--search-config", str(SEARCH), "--output", str(cost_output)
    )
    assert result.returncode == 0, result.stderr
    cost = _payload(cost_output)
    assert cost["status"] == "not_estimated"
    assert cost["price_registry_id"] is None
    assert cost["candidate_estimates"] == []


def test_archive_is_exactly_targeted_reproducible_and_binding_validated(tmp_path: Path) -> None:
    archives: list[Path] = []
    for external in (tmp_path / "one", tmp_path / "two"):
        report = external / "build.json"
        result = _run(
            "build-archive",
            "--search-config", str(SEARCH),
            "--fixture-root", str(FIXTURES),
            "--implementation-commit", COMMIT,
            "--freeze-id", "filter-v5-freeze-v1",
            "--run-id", "filter-v5-build",
            "--output-root", str(external / "archives"),
            "--output", str(report),
        )
        assert result.returncode == 0, result.stderr
        archive = external / "archives" / "filter-v5-build"
        archives.append(archive)
        assert archive.is_dir()
        assert _payload(report)["archive_valid"] is True

    assert _archive_bytes(archives[0]) == _archive_bytes(archives[1])
    validation = tmp_path / "validation.json"
    result = _run(
        "validate-archive",
        "--archive", str(archives[0]),
        "--expected-implementation-commit", COMMIT,
        "--expected-search-config-hash", SEARCH_HASH,
        "--output", str(validation),
    )
    assert result.returncode == 0, result.stderr
    assert _payload(validation)["archive_valid"] is True

    for option, value, code in (
        ("--expected-implementation-commit", "b" * 40, "IMPLEMENTATION_COMMIT_MISMATCH"),
        ("--expected-search-config-hash", "0" * 64, "SEARCH_CONFIG_HASH_MISMATCH"),
    ):
        arguments = [
            "validate-archive", "--archive", str(archives[0]),
            "--expected-implementation-commit", COMMIT,
            "--expected-search-config-hash", SEARCH_HASH,
            "--output", str(tmp_path / f"invalid-{option[11:]}.json"),
        ]
        arguments[arguments.index(option) + 1] = value
        rejected = _run(*arguments)
        assert rejected.returncode != 0
        assert code in rejected.stderr


@pytest.mark.parametrize(
    ("option", "value", "code"),
    (
        ("--implementation-commit", "abc", "IMPLEMENTATION_COMMIT_INVALID"),
        ("--run-id", "../escape", "RUN_ID_INVALID"),
        ("--freeze-id", "nested/path", "FREEZE_ID_INVALID"),
    ),
)
def test_archive_rejects_invalid_commit_and_identifiers(
    tmp_path: Path, option: str, value: str, code: str
) -> None:
    arguments = [
        "build-archive", "--search-config", str(SEARCH), "--fixture-root", str(FIXTURES),
        "--implementation-commit", COMMIT, "--freeze-id", "freeze-v1",
        "--run-id", "run-v1", "--output-root", str(tmp_path / "archives"),
        "--output", str(tmp_path / "report.json"),
    ]
    arguments[arguments.index(option) + 1] = value

    result = _run(*arguments)

    assert result.returncode != 0
    assert code in result.stderr


def test_bct_readiness_consumes_explicit_reports_without_constructing_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mft = tmp_path / "mft.json"
    assert _run(
        "mft", "--search-config", str(SEARCH), "--fixture-root", str(FIXTURES),
        "--output", str(mft),
    ).returncode == 0
    build = tmp_path / "build.json"
    assert _run(
        "build-archive", "--search-config", str(SEARCH), "--fixture-root", str(FIXTURES),
        "--implementation-commit", COMMIT, "--freeze-id", "freeze-v1",
        "--run-id", "run-v1", "--output-root", str(tmp_path / "archives"),
        "--output", str(build),
    ).returncode == 0
    archive = tmp_path / "archive.json"
    assert _run(
        "validate-archive", "--archive", str(tmp_path / "archives" / "run-v1"),
        "--expected-implementation-commit", COMMIT,
        "--expected-search-config-hash", SEARCH_HASH, "--output", str(archive),
    ).returncode == 0

    output = tmp_path / "readiness.json"
    result = _run(
        "bct-readiness", "--search-config", str(SEARCH), "--mft-report", str(mft),
        "--archive-report", str(archive), "--execution-prerequisites", str(PREREQUISITES),
        "--output", str(output),
    )
    assert result.returncode == 0, result.stderr
    readiness = _payload(output)
    assert readiness["software_interface_status"] == "ready"
    assert readiness["execution_status"] == "blocked"
    assert readiness["provider_calls_issued"] == 0
    family_statuses = readiness["family_statuses"]
    assert isinstance(family_statuses, list)
    assert all(isinstance(item, dict) and item["status"] == "not_executed" for item in family_statuses)

    malformed = json.loads(mft.read_text(encoding="utf-8"))
    malformed["safety_report"]["cases"] = malformed["safety_report"]["cases"][:-4]
    mft.write_text(json.dumps(malformed, sort_keys=True, separators=(",", ":")) + "\n")
    rejected = _run(
        "bct-readiness", "--search-config", str(SEARCH), "--mft-report", str(mft),
        "--archive-report", str(archive), "--execution-prerequisites", str(PREREQUISITES),
        "--output", str(output),
    )
    assert rejected.returncode != 0 and "MFT_REGISTRY_MISMATCH" in rejected.stderr

    monkeypatch.setattr(client_factory, "build_llm_client", pytest.fail)
    monkeypatch.setattr(sys, "argv", ["memcontam", "phase12", "filter-v5", "bct-run"])
    with pytest.raises(SystemExit) as error:
        root_cli.main()
    assert error.value.code == 2


def test_bct_readiness_rejects_authorized_prerequisites(tmp_path: Path) -> None:
    mft = tmp_path / "mft.json"
    assert _run(
        "mft", "--search-config", str(SEARCH), "--fixture-root", str(FIXTURES),
        "--output", str(mft),
    ).returncode == 0
    archive_report = tmp_path / "archive.json"
    assert _run(
        "build-archive", "--search-config", str(SEARCH), "--fixture-root", str(FIXTURES),
        "--implementation-commit", COMMIT, "--freeze-id", "freeze-v1",
        "--run-id", "run-v1", "--output-root", str(tmp_path / "archives"),
        "--output", str(archive_report),
    ).returncode == 0
    assert _run(
        "validate-archive", "--archive", str(tmp_path / "archives" / "run-v1"),
        "--expected-implementation-commit", COMMIT,
        "--expected-search-config-hash", SEARCH_HASH, "--output", str(archive_report),
    ).returncode == 0

    prerequisites = tmp_path / "authorized_prerequisites.json"
    payload = json.loads(PREREQUISITES.read_text(encoding="utf-8"))
    payload.update(
        {
            "search_config_frozen": True,
            "inventory_frozen": True,
            "canonical_patch_status": "applied",
            "provider_config_enabled": True,
            "runtime_authorization_present": True,
        }
    )
    prerequisites.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")

    output = tmp_path / "readiness.json"
    result = _run(
        "bct-readiness", "--search-config", str(SEARCH), "--mft-report", str(mft),
        "--archive-report", str(archive_report), "--execution-prerequisites", str(prerequisites),
        "--output", str(output),
    )

    assert result.returncode != 0
    assert "BCT_EXECUTION_AUTHORIZATION_FORBIDDEN" in result.stderr
    assert not output.exists()


def test_build_archive_rejects_existing_target(tmp_path: Path) -> None:
    target = tmp_path / "archives" / "run-v1"
    target.mkdir(parents=True)

    result = _run(
        "build-archive", "--search-config", str(SEARCH), "--fixture-root", str(FIXTURES),
        "--implementation-commit", COMMIT, "--freeze-id", "freeze-v1",
        "--run-id", "run-v1", "--output-root", str(tmp_path / "archives"),
        "--output", str(tmp_path / "build.json"),
    )

    assert result.returncode != 0
    assert "ARCHIVE_ROOT_EXISTS" in result.stderr


def test_commands_require_declared_output_and_existing_input_paths(tmp_path: Path) -> None:
    missing_output = _run("validate-search-config", "--config", str(SEARCH))
    missing_input = _run(
        "validate-search-config",
        "--config", str(tmp_path / "missing.yaml"),
        "--output", str(tmp_path / "report.json"),
    )

    assert missing_output.returncode == 2 and "--output" in missing_output.stderr
    assert missing_input.returncode != 0 and "missing.yaml" in missing_input.stderr
