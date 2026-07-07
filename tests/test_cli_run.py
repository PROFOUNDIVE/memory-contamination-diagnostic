from __future__ import annotations

import json
import pytest

from memcontam.cli import run_config


def test_run_config_writes_replay_trial_log_jsonl(tmp_path) -> None:
    sample_path = tmp_path / "game24_one.jsonl"
    sample_path.write_text(
        '{"sample_id":"sample_1","numbers":[1,3,4,6],"target":24}\n',
        encoding="utf-8",
    )
    output_dir = tmp_path / "runs"
    config = {
        "run": {"name": "smoke"},
        "models": ["replay"],
        "tasks": [{"name": "game24", "sample_path": str(sample_path), "limit": 1}],
        "baselines": ["no_memory"],
        "arms": ["clean"],
        "logging": {"output_dir": str(output_dir)},
        "replay": {"responses": ["final: 6 / (1 - 3 / 4)"]},
    }

    run_dir = run_config(config, run_id="smoke_run")

    trials_path = run_dir / "trials.jsonl"
    rows = [json.loads(line) for line in trials_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["trial_id"] == "smoke_run:game24:sample_1:no_memory:clean:replay"
    assert row["prompt_messages"]
    assert row["raw_response"] == "final: 6 / (1 - 3 / 4)"
    assert row["parsed_answer"] == "6 / (1 - 3 / 4)"
    assert row["verifier_result"]["is_correct"] is True
    assert row["memory_before"] == []
    assert row["retrieved_memory"] == []
    assert row["memory_after"] == []


def test_run_config_rejects_missing_replay_response_for_sample(tmp_path) -> None:
    sample_path = tmp_path / "game24_one.jsonl"
    sample_path.write_text(
        '{"sample_id":"sample_1","numbers":[1,3,4,6],"target":24}\n',
        encoding="utf-8",
    )
    config = {
        "run": {"name": "smoke"},
        "models": ["replay"],
        "tasks": [{"name": "game24", "sample_path": str(sample_path), "limit": 1}],
        "baselines": ["no_memory"],
        "arms": ["clean"],
        "logging": {"output_dir": str(tmp_path / "runs")},
        "replay": {"responses_by_sample": {}},
    }

    with pytest.raises(SystemExit, match="missing replay response for sample: sample_1"):
        run_config(config, run_id="smoke_run")


def test_run_config_rejects_missing_contamination_catalog_for_contaminated_arm(
    tmp_path, monkeypatch
) -> None:
    sample_path = tmp_path / "game24_one.jsonl"
    sample_path.write_text(
        '{"sample_id":"sample_1","numbers":[1,3,4,6],"target":24}\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    config = {
        "run": {"name": "smoke"},
        "models": ["replay"],
        "tasks": [{"name": "game24", "sample_path": str(sample_path), "limit": 1}],
        "baselines": ["retrieval_rag"],
        "arms": ["contaminated"],
        "logging": {"output_dir": str(tmp_path / "runs")},
        "replay": {"responses": ["final: 6 / (1 - 3 / 4)"]},
    }

    with pytest.raises(SystemExit, match="contamination catalog not found"):
        run_config(config, run_id="smoke_run")


def test_run_config_rejects_run_id_path_traversal(tmp_path) -> None:
    config = {
        "run": {"name": "smoke"},
        "models": ["replay"],
        "tasks": [],
        "baselines": ["no_memory"],
        "arms": ["clean"],
        "logging": {"output_dir": str(tmp_path / "runs")},
        "replay": {"responses": ["final: 6 / (1 - 3 / 4)"]},
    }

    with pytest.raises(SystemExit, match="invalid run id"):
        run_config(config, run_id="../outside")
