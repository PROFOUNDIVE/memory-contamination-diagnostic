from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "phase12" / "pilot_a_game24_minimal.yaml"


def _module() -> Any:
    assert importlib.util.find_spec("memcontam.readiness.pilot_a_invariants") is not None
    return importlib.import_module("memcontam.readiness.pilot_a_invariants")


def _rewrite_manifest(run_dir: Path) -> None:
    manifest_path = run_dir / "public_artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for filename, artifact in manifest["artifacts"].items():
        path = run_dir / filename
        artifact["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        if filename.endswith(".jsonl"):
            artifact["count"] = len(path.read_text(encoding="utf-8").splitlines())
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    (run_dir / "archive_seal.json").write_text(
        json.dumps(
            {"public_artifact_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()},
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )


def _run_dir(tmp_path: Path) -> tuple[Any, Path]:
    module = _module()
    run_dir = module.run_replay(CONFIG, "pilot-a", artifact_root=tmp_path)
    return module, run_dir


def test_invariant_inspector_accepts_a_valid_wrong_answer(tmp_path: Path) -> None:
    module, run_dir = _run_dir(tmp_path)

    report = module.inspect_run(run_dir)

    assert report["overall"] == "pass"
    assert report["scientific_result"] is False
    assert report["live_provider_calls"] == 0
    assert [result["name"] for result in report["results"]] == list(module.INVARIANT_NAMES)
    assert all(result["status"] == "pass" for result in report["results"])


def test_rejects_dc_style_pre_generation_ordering_in_other_baselines(tmp_path: Path) -> None:
    module, run_dir = _run_dir(tmp_path)
    calls_path = run_dir / "calls.jsonl"
    calls = [json.loads(line) for line in calls_path.read_text(encoding="utf-8").splitlines()]
    calls[0]["baseline"] = "rag_frozen"
    calls_path.write_text(
        "".join(json.dumps(call, sort_keys=True, separators=(",", ":")) + "\n" for call in calls),
        encoding="utf-8",
    )
    _rewrite_manifest(run_dir)

    report = module.inspect_run(run_dir)

    assert report["overall"] == "fail"
    assert report["reason_code"] == "BASELINE_PRE_GENERATION_ORDERING_FAILED"


def test_rejects_unresolved_context_event_reference(tmp_path: Path) -> None:
    module, run_dir = _run_dir(tmp_path)
    trials_path = run_dir / "trials.jsonl"
    trials = [json.loads(line) for line in trials_path.read_text(encoding="utf-8").splitlines()]
    trials[1]["context_event_id_or_none"] = "context:missing"
    trials_path.write_text(
        "".join(json.dumps(trial, sort_keys=True, separators=(",", ":")) + "\n" for trial in trials),
        encoding="utf-8",
    )
    _rewrite_manifest(run_dir)

    report = module.inspect_run(run_dir)

    assert report["overall"] == "fail"
    assert report["reason_code"] == "UNRESOLVED_REFERENCE"


def test_python_candidate_validation_never_executes_archive_code(
    tmp_path: Path, monkeypatch
) -> None:
    module, run_dir = _run_dir(tmp_path)
    called = False

    def unexpected_run(*_args, **_kwargs):
        nonlocal called
        called = True
        return subprocess.CompletedProcess([], 0)

    monkeypatch.setattr(subprocess, "run", unexpected_run)

    report = module.inspect_run(run_dir)

    assert report["overall"] == "pass"
    assert called is False


def test_python_candidate_must_have_static_parser_runtime_timeout_and_semantic_statuses(
    tmp_path: Path,
) -> None:
    module, run_dir = _run_dir(tmp_path)
    ledger_path = run_dir / "decision_ledger.jsonl"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))

    for field, value, expected in (
        ("parser_status", "parser_failure", "PYTHON_CANDIDATE_PARSER_FAILURE"),
        ("runtime_status", "runtime_failure", "PYTHON_CANDIDATE_RUNTIME_FAILURE"),
        ("termination_status", "timeout", "PYTHON_CANDIDATE_TIMEOUT"),
        ("semantic_result", "unknown", "PYTHON_CANDIDATE_SEMANTIC_STATUS_INVALID"),
    ):
        ledger["python_candidate"].update(
            parser_status="parsed",
            runtime_status="not_executed",
            termination_status="within_limit",
            semantic_result="semantic_invalid",
        )
        ledger["python_candidate"][field] = value
        ledger_path.write_text(
            json.dumps(ledger, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
        )
        _rewrite_manifest(run_dir)

        report = module.inspect_run(run_dir)

        assert report["overall"] == "fail"
        assert report["reason_code"] == expected
