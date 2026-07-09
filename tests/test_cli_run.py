from __future__ import annotations

import json
from pathlib import Path

import pytest

from memcontam.cli import run_config
from memcontam.cli import load_config
from memcontam.clients.base import LLMResponse
from memcontam.logging.schema import TrialLog


def _write_game24_sample(tmp_path, numbers=None) -> str:
    sample_path = tmp_path / "game24_one.jsonl"
    row = {"sample_id": "sample_1", "numbers": numbers or [1, 3, 4, 6], "target": 24}
    sample_path.write_text(json.dumps(row) + chr(10), encoding="utf-8")
    return str(sample_path)


def _write_contamination_catalog(tmp_path, baseline: str, content: str) -> None:
    catalog_dir = tmp_path / "data" / "contamination"
    catalog_dir.mkdir(parents=True)
    catalog_row = {
        "entry_id": f"{baseline}_memory_1",
        "task": "game24",
        "type": "proxy_memory",
        "content": content,
        "target_baselines": [baseline],
    }
    (catalog_dir / "catalog_v0.jsonl").write_text(json.dumps(catalog_row) + chr(10), encoding="utf-8")


def _run_single_baseline(
    tmp_path, monkeypatch, baseline: str, memory_content: str, response: str = "final: 6 / (1 - 3 / 4)"
) -> dict:
    sample_path = _write_game24_sample(tmp_path)
    _write_contamination_catalog(tmp_path, baseline, memory_content)
    monkeypatch.chdir(tmp_path)
    config = {
        "run": {"name": "smoke"},
        "models": ["replay"],
        "tasks": [{"name": "game24", "sample_path": sample_path, "limit": 1}],
        "baselines": [baseline],
        "arms": ["contaminated"],
        "logging": {"output_dir": str(tmp_path / "runs")},
        "replay": {"responses": [response]},
    }

    run_dir = run_config(config, run_id=f"{baseline}_run")
    trials_path = run_dir / "trials.jsonl"
    rows = [json.loads(line) for line in trials_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    TrialLog.model_validate(rows[0])
    return rows[0]


def test_run_config_writes_replay_trial_log_jsonl(tmp_path) -> None:
    sample_path = tmp_path / "game24_one.jsonl"
    sample_path.write_text(
        '{"sample_id":"sample_1","numbers":[1,3,4,6],"target":24}' + chr(10),
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
    TrialLog.model_validate(row)
    assert row["trial_id"] == "smoke_run:game24:sample_1:no_memory:clean:replay"
    assert row["prompt_messages"]
    assert row["raw_response"] == "final: 6 / (1 - 3 / 4)"
    assert row["parsed_answer"] == "6 / (1 - 3 / 4)"
    assert row["verifier_result"]["is_correct"] is True
    assert row["memory_before"] == []
    assert row["retrieved_memory"] == []
    assert row["memory_after"] == []
    assert row["filter_decision"] is None
    assert row["memory_write_event"] is None
    assert row["contamination_exposure"] == {
        "condition": "clean",
        "is_exposed": False,
        "source_entry_ids": [],
        "contamination_types": [],
        "memory_before_entry_ids": [],
        "retrieved_entry_ids": [],
        "exposure_mode": "none",
        "reason": "clean arm has no contaminated memory sources",
    }
    assert row["bad_memory_uptake_label"] == "not_applicable"
    assert row["repeated_failure_label"] == "not_applicable"
    assert row["recovery_after_filter_label"] == "not_applicable"


def test_run_config_clean_multitask_replay_emits_all_three_tasks(tmp_path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(repo_root)
    output_dir = tmp_path / "runs"
    config = {
        "run": {"name": "smoke"},
        "models": ["replay"],
        "tasks": [
            {"name": "game24", "sample_path": str((repo_root / "data/tasks/game24_pilot.jsonl").resolve()), "limit": 1},
            {
                "name": "math_equation_balancer",
                "sample_path": str((repo_root / "data/tasks/math_equation_balancer_pilot.jsonl").resolve()),
                "limit": 1,
            },
            {
                "name": "word_sorting",
                "sample_path": str((repo_root / "data/tasks/word_sorting_pilot.jsonl").resolve()),
                "limit": 1,
            },
        ],
        "baselines": ["no_memory"],
        "arms": ["clean"],
        "logging": {"output_dir": str(output_dir)},
        "replay": {
            "responses_by_sample": {
                "game24_pilot_001": "final: 6 / (1 - 3 / 4)",
                "meb_pilot_001": "2 + 5 = 7",
                "word_sorting_pilot_001": "apple banana pear",
            }
        },
    }

    run_dir = run_config(config, run_id="multitask_clean_run")
    rows = [json.loads(line) for line in (run_dir / "trials.jsonl").read_text(encoding="utf-8").splitlines()]

    assert len(rows) == 3
    assert {row["task_name"] for row in rows} == {"game24", "math_equation_balancer", "word_sorting"}
    for row in rows:
        TrialLog.model_validate(row)
        assert row["verifier_result"]["is_correct"] is True
        assert row["contamination_exposure"]["condition"] == "clean"


def test_clean_multitask_replay_ignores_catalog(tmp_path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(repo_root)
    config = load_config(repo_root / "configs/pilot_multitask_replay.yaml")
    config["logging"]["output_dir"] = str(tmp_path / "runs")

    run_dir = run_config(config, run_id="task_T9_clean_after_c")
    rows = [json.loads(line) for line in (run_dir / "trials.jsonl").read_text(encoding="utf-8").splitlines()]

    assert rows
    for row in rows:
        assert row["contamination_exposure"]["condition"] == "clean"
        assert row["contamination_exposure"]["source_entry_ids"] == []


def test_run_config_replay_mode_ignores_missing_provider_env_vars(tmp_path, monkeypatch) -> None:
    sample_path = tmp_path / "game24_one.jsonl"
    sample_path.write_text(
        '{"sample_id":"sample_1","numbers":[1,3,4,6],"target":24}' + chr(10),
        encoding="utf-8",
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_COMPATIBLE_API_KEY", raising=False)
    config = {
        "run": {"name": "smoke"},
        "models": ["replay"],
        "tasks": [{"name": "game24", "sample_path": str(sample_path), "limit": 1}],
        "baselines": ["no_memory"],
        "arms": ["clean"],
        "logging": {"output_dir": str(tmp_path / "runs")},
        "replay": {"responses": ["final: 6 / (1 - 3 / 4)"]},
    }

    run_dir = run_config(config, run_id="smoke_run")
    assert (run_dir / "trials.jsonl").exists()


def test_run_config_live_smoke_flag_defaults_to_disabled(tmp_path, monkeypatch) -> None:
    sample_path = tmp_path / "game24_one.jsonl"
    sample_path.write_text(
        '{"sample_id":"sample_1","numbers":[1,3,4,6],"target":24}' + chr(10),
        encoding="utf-8",
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = {
        "run": {"name": "smoke"},
        "models": ["replay"],
        "tasks": [{"name": "game24", "sample_path": str(sample_path), "limit": 1}],
        "baselines": ["no_memory"],
        "arms": ["clean"],
        "logging": {"output_dir": str(tmp_path / "runs")},
        "replay": {"responses": ["final: 6 / (1 - 3 / 4)"]},
    }

    run_dir = run_config(config, run_id="smoke_run")
    assert (run_dir / "trials.jsonl").exists()


def test_run_config_live_smoke_enabled_without_api_key_fails(tmp_path, monkeypatch) -> None:
    sample_path = tmp_path / "game24_one.jsonl"
    sample_path.write_text(
        '{"sample_id":"sample_1","numbers":[1,3,4,6],"target":24}' + chr(10),
        encoding="utf-8",
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = {
        "run": {"name": "smoke"},
        "models": ["replay"],
        "tasks": [{"name": "game24", "sample_path": str(sample_path), "limit": 1}],
        "baselines": ["no_memory"],
        "arms": ["clean"],
        "logging": {"output_dir": str(tmp_path / "runs")},
        "replay": {"responses": ["final: 6 / (1 - 3 / 4)"]},
        "live_smoke": {"enabled": True},
    }

    with pytest.raises(SystemExit, match="missing API key env var"):
        run_config(config, run_id="smoke_run")


def test_run_config_live_smoke_enabled_with_mocked_client_emits_trial_log(
    tmp_path, monkeypatch
) -> None:
    sample_path = tmp_path / "game24_one.jsonl"
    sample_path.write_text(
        '{"sample_id":"sample_1","numbers":[1,3,4,6],"target":24}' + chr(10),
        encoding="utf-8",
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    class FakeClient:
        def chat(self, messages, model, config):
            return LLMResponse(
                content="final: 6 / (1 - 3 / 4)",
                raw={"mock": True},
                token_usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
                latency_ms=42,
            )

    config = {
        "run": {"name": "smoke"},
        "models": ["openai_compatible"],
        "tasks": [{"name": "game24", "sample_path": str(sample_path), "limit": 1}],
        "baselines": ["no_memory"],
        "arms": ["clean"],
        "logging": {"output_dir": str(tmp_path / "runs")},
        "live_smoke": {"enabled": True},
    }

    run_dir = run_config(config, run_id="smoke_run", _client_override=FakeClient())
    trials_path = run_dir / "trials.jsonl"
    rows = [json.loads(line) for line in trials_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    row = rows[0]
    TrialLog.model_validate(row)
    assert row["raw_response"] == "final: 6 / (1 - 3 / 4)"
    assert row["latency_ms"] == 42
    assert row["token_usage"] == {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
    assert row["verifier_result"]["is_correct"] is True
    assert row["parsed_answer"] == "6 / (1 - 3 / 4)"


def test_run_config_rejects_missing_replay_response_for_sample(tmp_path) -> None:
    sample_path = tmp_path / "game24_one.jsonl"
    sample_path.write_text(
        '{"sample_id":"sample_1","numbers":[1,3,4,6],"target":24}' + chr(10),
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


def test_run_config_rejects_empty_replay_input(tmp_path) -> None:
    sample_path = tmp_path / "game24_empty.jsonl"
    sample_path.write_text(chr(10), encoding="utf-8")
    config = {
        "run": {"name": "smoke"},
        "models": ["replay"],
        "tasks": [{"name": "game24", "sample_path": str(sample_path), "limit": 1}],
        "baselines": ["no_memory"],
        "arms": ["clean"],
        "logging": {"output_dir": str(tmp_path / "runs")},
        "replay": {"responses": ["final: 6 / (1 - 3 / 4)"]},
    }

    with pytest.raises(SystemExit, match="empty replay input"):
        run_config(config, run_id="smoke_run")


def test_run_config_rejects_malformed_replay_input(tmp_path) -> None:
    sample_path = tmp_path / "game24_bad.jsonl"
    sample_path.write_text('{"sample_id":"sample_1",' + chr(10), encoding="utf-8")
    config = {
        "run": {"name": "smoke"},
        "models": ["replay"],
        "tasks": [{"name": "game24", "sample_path": str(sample_path), "limit": 1}],
        "baselines": ["no_memory"],
        "arms": ["clean"],
        "logging": {"output_dir": str(tmp_path / "runs")},
        "replay": {"responses": ["final: 6 / (1 - 3 / 4)"]},
    }

    with pytest.raises(SystemExit, match="malformed replay input"):
        run_config(config, run_id="smoke_run")


def test_run_config_rejects_missing_contamination_catalog_for_contaminated_arm(
    tmp_path, monkeypatch
) -> None:
    sample_path = tmp_path / "game24_one.jsonl"
    sample_path.write_text(
        '{"sample_id":"sample_1","numbers":[1,3,4,6],"target":24}' + chr(10),
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


def test_retrieval_rag_logs_lexical_retrieved_memory_and_scores(tmp_path, monkeypatch) -> None:
    row = _run_single_baseline(
        tmp_path,
        monkeypatch,
        "retrieval_rag",
        "For numbers 1 3 4 6 use the expression 6 / (1 - 3 / 4).",
    )

    assert row["retrieved_memory"]
    assert row["retrieved_memory"][0]["entry_id"] == "retrieval_rag_memory_1"
    assert row["retrieved_memory"][0]["content"].startswith("For numbers 1 3 4 6")
    assert row["retrieved_memory"][0]["memory_type"] == "proxy_memory"
    assert row["retrieved_scores"]
    assert row["retrieved_scores"][0] > 0
    prompt_text = "\n".join(message["content"] for message in row["prompt_messages"])
    assert "entry_id=retrieval_rag_memory_1" in prompt_text
    assert "score=" in prompt_text
    assert row["memory_write_event"] is None
    assert row["memory_before"] == row["memory_after"]


def test_bot_style_logs_at_most_one_retrieved_template(tmp_path, monkeypatch) -> None:
    row = _run_single_baseline(
        tmp_path,
        monkeypatch,
        "bot_style",
        "Template for 1 3 4 6: first create 1 - 3 / 4, then divide 6.",
    )

    assert len(row["retrieved_memory"]) <= 1
    assert len(row["retrieved_scores"]) <= 1
    assert row["retrieved_memory"][0]["content"].startswith("Template for 1 3 4 6")


def test_reflexion_style_includes_recent_reflection_in_prompt_messages(tmp_path, monkeypatch) -> None:
    row = _run_single_baseline(
        tmp_path,
        monkeypatch,
        "reflexion_style",
        "Reflection for 1 3 4 6: avoid early multiplication; try division last.",
    )

    prompt_text = "\n".join(message["content"] for message in row["prompt_messages"])
    assert "Reflection for 1 3 4 6" in prompt_text
    assert row["retrieved_memory"] == []
    assert row["retrieved_scores"] == []


def test_contaminated_filter_logs_filter_decision_and_filtered_exposure(tmp_path, monkeypatch) -> None:
    sample_path = _write_game24_sample(tmp_path)
    _write_contamination_catalog(
        tmp_path,
        "retrieval_rag",
        "For numbers 1 3 4 6 use the wrong expression 1 + 3 + 4 + 6.",
    )
    monkeypatch.chdir(tmp_path)
    config = {
        "run": {"name": "smoke"},
        "models": ["replay"],
        "tasks": [{"name": "game24", "sample_path": sample_path, "limit": 1}],
        "baselines": ["retrieval_rag"],
        "arms": ["contaminated_filter"],
        "logging": {"output_dir": str(tmp_path / "runs")},
        "replay": {"responses": ["final: 6 / (1 - 3 / 4)"]},
    }

    run_dir = run_config(config, run_id="filter_run")
    row = json.loads((run_dir / "trials.jsonl").read_text(encoding="utf-8"))
    TrialLog.model_validate(row)

    assert row["filter_decision"] == {"filter": "drop_known_contaminated", "dropped": 1}
    assert row["memory_before"] == []
    assert row["retrieved_memory"] == []
    assert row["contamination_exposure"]["condition"] == "contaminated_filter"
    assert row["contamination_exposure"]["is_exposed"] is False
    assert row["contamination_exposure"]["source_entry_ids"] == []
    assert row["contamination_exposure"]["exposure_mode"] == "none"
    assert row["recovery_after_filter_label"] == "not_applicable"


def test_contaminated_row_with_no_retrieval_logs_memory_before_exposure(tmp_path, monkeypatch) -> None:
    row = _run_single_baseline(
        tmp_path,
        monkeypatch,
        "full_history",
        "Wrong reflection for 1 3 4 6: use 1 + 3 + 4 + 6.",
    )

    assert row["retrieved_memory"] == []
    assert row["contamination_exposure"] == {
        "condition": "contaminated",
        "is_exposed": True,
        "source_entry_ids": ["full_history_memory_1"],
        "contamination_types": ["proxy_memory"],
        "memory_before_entry_ids": ["full_history_memory_1"],
        "retrieved_entry_ids": [],
        "exposure_mode": "memory_before",
        "reason": "contaminated memory sources were available before prompting",
    }


def test_controlled_exposure_does_not_imply_bad_memory_uptake(tmp_path, monkeypatch) -> None:
    row = _run_single_baseline(
        tmp_path,
        monkeypatch,
        "retrieval_rag",
        "For numbers 1 3 4 6 use the wrong expression 1 + 3 + 4 + 6.",
    )

    assert row["contamination_exposure"]["is_exposed"] is True
    assert row["retrieved_memory"]
    assert row["bad_memory_uptake_label"] == "not_evaluable"
    assert row["bad_memory_uptake_label"] != "uptake_detected"
    assert row["memory_write_event"] is None


def test_failure_row_still_validates_with_provenance_labels(tmp_path, monkeypatch) -> None:
    row = _run_single_baseline(
        tmp_path,
        monkeypatch,
        "no_memory",
        "Unused contaminated memory.",
        "final: 1 + 3 + 4 + 6",
    )

    assert row["verifier_result"]["is_correct"] is False
    assert row["verifier_result"]["reason"]
    assert row["bad_memory_uptake_label"] == "not_evaluable"
    assert row["repeated_failure_label"] == "first_failure"
    TrialLog.model_validate(row)


def test_repeated_failure_label_sequence(tmp_path, monkeypatch) -> None:
    sample_path = tmp_path / "game24_repeat.jsonl"
    sample_path.write_text(
        json.dumps({"sample_id": "sample_1", "numbers": [1, 3, 4, 6], "target": 24})
        + "\n"
        + json.dumps({"sample_id": "sample_1", "numbers": [1, 3, 4, 6], "target": 24})
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    config = {
        "run": {"name": "smoke"},
        "models": ["replay"],
        "tasks": [{"name": "game24", "sample_path": str(sample_path), "limit": 2}],
        "baselines": ["no_memory"],
        "arms": ["clean"],
        "logging": {"output_dir": str(tmp_path / "runs")},
        "replay": {"responses_by_sample": {"sample_1": "final: 1 + 3 + 4 + 6"}},
    }

    run_dir = run_config(config, run_id="repeat_run")
    rows = [
        json.loads(line)
        for line in (run_dir / "trials.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert len(rows) == 2
    assert all(row["verifier_result"]["is_correct"] is False for row in rows)
    assert rows[0]["repeated_failure_label"] == "first_failure"
    assert rows[1]["repeated_failure_label"] == "repeated_failure"


def test_repeated_failure_no_cross_task_repeat(tmp_path, monkeypatch) -> None:
    game24_path = tmp_path / "game24_sample.jsonl"
    game24_path.write_text(
        json.dumps({"sample_id": "sample_1", "numbers": [1, 3, 4, 6], "target": 24}) + "\n",
        encoding="utf-8",
    )
    word_sorting_path = tmp_path / "word_sorting_sample.jsonl"
    word_sorting_path.write_text(
        json.dumps(
            {
                "sample_id": "sample_1",
                "words": ["apple", "banana"],
                "sorted_words": ["apple", "banana"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    config = {
        "run": {"name": "smoke"},
        "models": ["replay"],
        "tasks": [
            {"name": "game24", "sample_path": str(game24_path), "limit": 1},
            {"name": "word_sorting", "sample_path": str(word_sorting_path), "limit": 1},
        ],
        "baselines": ["no_memory"],
        "arms": ["clean"],
        "logging": {"output_dir": str(tmp_path / "runs")},
        "replay": {"responses_by_sample": {"sample_1": "final: 1 + 3 + 4 + 6"}},
    }

    run_dir = run_config(config, run_id="cross_task_run")
    rows = [
        json.loads(line)
        for line in (run_dir / "trials.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert len(rows) == 2
    assert {row["task_name"] for row in rows} == {"game24", "word_sorting"}
    assert all(row["verifier_result"]["is_correct"] is False for row in rows)
    assert rows[0]["repeated_failure_label"] == "first_failure"
    assert rows[1]["repeated_failure_label"] == "first_failure"
