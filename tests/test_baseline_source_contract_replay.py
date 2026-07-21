from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

import memcontam.cli as cli
from memcontam.evaluation.aggregate import aggregate_run
from memcontam.logging.schema import CallEvent, FailureEvent, FilterEvent, MemoryEvent, TrialLog


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "baseline_fidelity_v2_source_contract_replay.yaml"
FIXTURE_PATH = ROOT / "data" / "replay" / "baseline_fidelity_v2_source_contract.yaml"
INSPECTOR = ROOT / "scripts" / "inspect_baseline_fidelity_v2.py"
MANIFEST = ROOT / "scripts" / "build_bfv2_evidence_manifest.py"
SEMANTIC_CALL_FIXTURES = ROOT / "tests" / "fixtures" / "baseline_fidelity_v2_semantic_call_hashes.json"


def test_f1b_config_loads_the_committed_stage_native_fixture() -> None:
    config = cli.load_config(CONFIG_PATH)
    fixture = yaml.safe_load(FIXTURE_PATH.read_text(encoding="utf-8"))

    assert config["run"]["fidelity_gate_layer"] == "source_contract"
    assert config["replay"]["responses_by_sample"] == fixture["responses_by_sample"]


def _rows(run_dir: Path, filename: str, model):
    return [
        model.model_validate(json.loads(line))
        for line in (run_dir / filename).read_text(encoding="utf-8").splitlines()
        if line
    ]


def _inspect(run_dir: Path, output: Path | None = None) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(INSPECTOR), str(run_dir)]
    if output is not None:
        command.extend(["--output", str(output)])
    return subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)


def _semantic_call_hashes(trials: list[TrialLog]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for trial in trials:
        stage_counts: dict[str, int] = {}
        for call in trial.method_calls:
            stage_counts[call.stage] = stage_counts.get(call.stage, 0) + 1
            key = ":".join(
                [
                    trial.task_name,
                    trial.sample_id,
                    trial.baseline,
                    call.stage,
                    str(stage_counts[call.stage]),
                ]
            )
            rendered = json.dumps(call.messages, sort_keys=True, separators=(",", ":")).replace(
                trial.run_id, "{{run_id}}"
            )
            rendered = re.sub(r"\b[0-9a-f]{32}\b", "{{entry_id}}", rendered)
            hashes[key] = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
    return hashes


def test_f1b_replay_parses_artifacts_locks_prompt_bytes_and_rejects_mutations(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(ROOT)
    config = copy.deepcopy(cli.load_config(CONFIG_PATH))
    config["logging"]["output_dir"] = str(tmp_path / "runs")
    run_dir = cli.run_config(config, run_id="bfv2-source-contract-replay-test")
    trials = _rows(run_dir, "trials.jsonl", TrialLog)
    calls = _rows(run_dir, "calls.jsonl", CallEvent)
    failures = _rows(run_dir, "failures.jsonl", FailureEvent)
    filters = _rows(run_dir, "filter_events.jsonl", FilterEvent)
    memory_events = _rows(run_dir, "memory_events.jsonl", MemoryEvent)

    assert len(trials) == 18
    assert len(calls) == 32
    assert len(failures) == 5
    assert len(filters) == 0
    assert len(memory_events) == 10
    assert aggregate_run(run_dir, stage="replay", contract="phase11")["n_trials"] == len(trials)
    assert any(
        trial.baseline == "no_memory"
        and trial.status == "succeeded"
        and trial.verifier_result is not None
        and not trial.verifier_result.is_correct
        for trial in trials
    )
    assert any(
        trial.baseline == "full_history" and trial.status == "failed" and len(trial.memory_after) > len(trial.memory_before)
        for trial in trials
    )
    assert all(
        trial.memory_before == trial.memory_after and trial.memory_write_event is None
        for trial in trials
        if trial.baseline == "retrieval_rag"
    )
    assert any(call.retrieved_records for trial in trials if trial.baseline == "retrieval_rag" for call in trial.method_calls)
    assert any(trial.baseline == "dynamic_cheatsheet_rs_optional" for trial in trials)
    assert _semantic_call_hashes(trials) == json.loads(
        SEMANTIC_CALL_FIXTURES.read_text(encoding="utf-8")
    )

    inspector_output = tmp_path / "inspector.json"
    result = _inspect(run_dir, inspector_output)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["overall"] == "pass"
    assert json.loads(inspector_output.read_text(encoding="utf-8"))["overall"] == "pass"

    prompt_mutation = tmp_path / "prompt-mutation"
    shutil.copytree(run_dir, prompt_mutation)
    rows = [json.loads(line) for line in (prompt_mutation / "trials.jsonl").read_text(encoding="utf-8").splitlines()]
    rows[0]["method_calls"][0]["messages"][0]["content"] += "!"
    (prompt_mutation / "trials.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert _inspect(prompt_mutation).returncode == 1

    unlocked_prompt_mutation = tmp_path / "unlocked-prompt-mutation"
    shutil.copytree(run_dir, unlocked_prompt_mutation)
    rows = [
        json.loads(line)
        for line in (unlocked_prompt_mutation / "trials.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    row = next(
        row
        for row in rows
        if row["sample_id"] == "game24_pilot_003" and row["baseline"] == "no_memory"
    )
    row["method_calls"][0]["messages"][0]["content"] += "!"
    call_id = row["method_calls"][0]["call_id"]
    (unlocked_prompt_mutation / "trials.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    call_rows = [
        json.loads(line)
        for line in (unlocked_prompt_mutation / "calls.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    next(call for call in call_rows if call["call_id"] == call_id)["messages"][0]["content"] += "!"
    (unlocked_prompt_mutation / "calls.jsonl").write_text(
        "".join(json.dumps(call) + "\n" for call in call_rows), encoding="utf-8"
    )
    assert _inspect(unlocked_prompt_mutation).returncode == 1

    filter_mutation = tmp_path / "filter-mutation"
    shutil.copytree(run_dir, filter_mutation)
    (filter_mutation / "filter_events.jsonl").write_text('{"not":"a filter event"}\n', encoding="utf-8")
    assert _inspect(filter_mutation).returncode == 1

    span_mutation = tmp_path / "span-mutation"
    shutil.copytree(run_dir, span_mutation)
    rows = [json.loads(line) for line in (span_mutation / "trials.jsonl").read_text(encoding="utf-8").splitlines()]
    span_trial = next(row for row in rows if row["baseline"] == "retrieval_rag")
    span_trial["method_calls"][0]["source_spans"][0]["entry_id"] = "mutated-source-id"
    (span_mutation / "trials.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert _inspect(span_mutation).returncode == 1

    evidence_manifest = tmp_path / "evidence-manifest.json"
    manifest_result = subprocess.run(
        [sys.executable, str(MANIFEST), "--config", str(CONFIG_PATH), "--run-dir", str(run_dir), "--inspector-output", str(inspector_output), "--output", str(evidence_manifest)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert manifest_result.returncode == 0, manifest_result.stdout + manifest_result.stderr
    artifact_hashes = json.loads(evidence_manifest.read_text(encoding="utf-8"))["artifacts"]
    assert str(run_dir / "trials.jsonl") in artifact_hashes
    assert str(run_dir / "filter_events.jsonl") in artifact_hashes
    assert str(FIXTURE_PATH) in artifact_hashes
    assert any("baseline_fidelity_v2" in path for path in artifact_hashes)
    manifest = json.loads(evidence_manifest.read_text(encoding="utf-8"))
    assert manifest["commit"]
    assert manifest["plan"]["sha256"]
    assert manifest["versions"]["prompt_version"] == "baseline_fidelity_v2"
    assert manifest["embedding_identity"]
    assert manifest["corpus_identity"]
    assert manifest["commands"] == [{"command": "inspect_baseline_fidelity_v2", "exit_code": 0}]
    assert "filter_events.jsonl" in manifest["strict_streams"]
