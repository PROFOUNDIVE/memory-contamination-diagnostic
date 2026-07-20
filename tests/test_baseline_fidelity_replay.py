from __future__ import annotations

import copy
import hashlib
import json
import socket
from pathlib import Path
from typing import Any

import memcontam.cli as cli
from memcontam.evaluation.aggregate import aggregate_run
from memcontam.logging.schema import RunMetadata, TrialLog


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "pilot_multitask_replay.yaml"
EXPECTED_STAGES = {
    "no_memory": ["no_memory_generate"],
    "full_history": ["full_history_generate"],
    "retrieval_rag": ["rag_generate"],
    "reflexion_style": ["reflexion_generate"],
    "bot_style": ["bot_problem_distill", "bot_instantiate_solve", "bot_thought_distill"],
}


def _replay_config(tmp_path: Path) -> dict[str, Any]:
    config = copy.deepcopy(cli.load_config(CONFIG_PATH))
    config["logging"]["output_dir"] = str(tmp_path / "runs")
    return config


def _trials(run_dir: Path) -> list[TrialLog]:
    return [
        TrialLog.model_validate(json.loads(line))
        for line in (run_dir / "trials.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def _deny_network(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("offline replay attempted network access")


def test_clean_pilot_replay_is_offline_non_scientific_and_uses_faithful_adapters(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(socket, "create_connection", _deny_network)
    monkeypatch.setattr(socket.socket, "connect", _deny_network)

    run_dir = cli.run_config(_replay_config(tmp_path), run_id="baseline_fidelity_clean_replay")
    trials = _trials(run_dir)
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    resolved = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    profile = json.loads((run_dir / "provider_profile.json").read_text(encoding="utf-8"))
    metadata = RunMetadata.model_validate(manifest["run_metadata"])

    assert manifest["status"] == "completed"
    assert manifest["counts"] == {
        "trials": 90,
        "calls": 126,
        "failures": 0,
        "filter_events": 0,
        "memory_events": 36,
    }
    assert metadata.stage == "replay"
    assert (
        metadata.config_hash
        == hashlib.sha256(
            json.dumps(resolved, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    )
    assert resolved["run"].get("execution_class") == "offline_contract_replay"
    assert resolved["run"].get("provider") == "replay"
    assert resolved["run"].get("scientific_result") is False
    assert resolved["run"].get("scientific_gate_id") is None
    assert profile == resolved["provider_config"]

    assert len(trials) == 90
    assert {trial.task_name for trial in trials} == {
        "game24",
        "math_equation_balancer",
        "word_sorting",
    }
    assert {trial.backbone for trial in trials} == {"gpt4o", "frontier_reasoning"}
    assert {trial.baseline for trial in trials} == set(EXPECTED_STAGES)
    assert all(trial.arm == "clean" and trial.status == "succeeded" for trial in trials)
    assert all(trial.verifier_result and trial.verifier_result.is_correct for trial in trials)
    assert all(trial.error_type is None and trial.failure_id is None for trial in trials)
    assert all(trial.contamination_exposure.exposure_mode == "clean" for trial in trials)
    assert all(trial.filter_decision is None for trial in trials)

    for trial in trials:
        assert [call.stage for call in trial.method_calls] == EXPECTED_STAGES[trial.baseline]
        assert trial.answer_call_id in {call.call_id for call in trial.method_calls}

    bot_trials = [trial for trial in trials if trial.baseline == "bot_style"]
    assert all(trial.answer_call_id == trial.method_calls[1].call_id for trial in bot_trials)
    assert all(
        trial.memory_write_event and trial.memory_write_event["source_outcome"] is True
        for trial in bot_trials
    )
    assert aggregate_run(run_dir, stage="replay")["n_trials"] == 90


def test_replay_keeps_valid_incorrect_and_closed_failure_outcomes(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(ROOT)

    incorrect_config = _replay_config(tmp_path)
    incorrect_config["run"]["mode"] = "faithful"
    incorrect_config["models"] = ["gpt4o"]
    incorrect_config["tasks"] = [incorrect_config["tasks"][1]]
    incorrect_config["baselines"] = ["no_memory"]
    incorrect_config["replay"]["responses_by_sample"]["meb_pilot_003"] = "final: 3 * 6 = 17"
    incorrect_run = cli.run_config(incorrect_config, run_id="baseline_fidelity_valid_incorrect")
    incorrect_trial = next(
        trial for trial in _trials(incorrect_run) if trial.sample_id == "meb_pilot_003"
    )

    assert incorrect_trial.status == "succeeded"
    assert (
        incorrect_trial.verifier_result is not None
        and not incorrect_trial.verifier_result.is_correct
    )
    assert incorrect_trial.error_type is None
    assert incorrect_trial.failure_id is None
    assert "scientific_ineligibility_reason" not in incorrect_trial.metadata

    failure_config = _replay_config(tmp_path)
    failure_config["run"]["mode"] = "faithful"
    failure_config["models"] = ["gpt4o"]
    failure_config["tasks"] = [failure_config["tasks"][1]]
    failure_config["baselines"] = ["no_memory"]
    failure_config["replay"]["responses_by_sample"]["meb_pilot_003"] = "   "
    failure_run = cli.run_config(failure_config, run_id="baseline_fidelity_closed_failure")
    failure_trial = next(
        trial for trial in _trials(failure_run) if trial.sample_id == "meb_pilot_003"
    )

    assert failure_trial.status == "failed"
    assert failure_trial.error_type == "BaselineOutputError"
    assert failure_trial.failure_id is not None
    assert failure_trial.metadata["failure_disposition"] == "no_memory_invalid_final_answer"
    assert failure_trial.metadata["scientific_ineligibility_reason"] == "invalid_final_answer"
    failure_event = json.loads((failure_run / "failures.jsonl").read_text(encoding="utf-8"))
    assert failure_event["disposition"] == failure_trial.metadata["failure_disposition"]
