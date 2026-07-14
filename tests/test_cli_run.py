from __future__ import annotations

import json
from pathlib import Path

import pytest

import memcontam.cli as cli
from memcontam.cli import run_config
from memcontam.cli import load_config
from memcontam.clients.base import LLMResponse
from memcontam.logging.schema import TrialLog, VerifierResult
from memcontam.memory.embeddings import FakeEmbeddingProvider


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


def test_embedding_provider_only_uses_fake_when_offline_fallback_enabled(monkeypatch) -> None:
    class MissingPinnedProvider:
        def __init__(self, **_kwargs):
            raise RuntimeError("missing pinned checkpoint")

    monkeypatch.setattr(cli, "SentenceTransformerProvider", MissingPinnedProvider)

    with pytest.raises(RuntimeError, match="missing pinned checkpoint"):
        cli._embedding_provider({"embedding": {}})

    provider = cli._embedding_provider({"embedding": {"offline_fallback": True}})
    assert isinstance(provider, FakeEmbeddingProvider)


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


def test_retrieval_rag_row_contains_provenance_and_stays_read_only(
    tmp_path, monkeypatch
) -> None:
    row = _run_single_baseline(
        tmp_path,
        monkeypatch,
        "retrieval_rag",
        "For numbers 1 3 4 6 use the expression 6 / (1 - 3 / 4).",
    )

    retrieved = row["retrieved_memory"][0]
    assert retrieved["entry_id"] == "retrieval_rag_memory_1"
    assert retrieved["content"].startswith("For numbers 1 3 4 6")
    assert retrieved["memory_type"] == "proxy_memory"
    assert retrieved["source_trial_id"] is None
    assert retrieved["metadata"] == {
        "task": "game24",
        "arm": "contaminated",
        "contamination_type": "proxy_memory",
    }
    assert 0 < row["retrieved_scores"][0] <= 1
    prompt_text = "\n".join(message["content"] for message in row["prompt_messages"])
    assert "entry_id=retrieval_rag_memory_1" in prompt_text
    assert "score=" in prompt_text
    assert "memory_type=proxy_memory" in prompt_text
    assert "source_trial_id=None" in prompt_text
    assert "metadata={'task': 'game24', 'arm': 'contaminated', 'contamination_type': 'proxy_memory'}" in prompt_text
    assert row["memory_write_event"] is None
    assert row["memory_before"] == row["memory_after"]


def test_bot_style_row_contains_prompt_sections_and_writeback_lineage(
    tmp_path, monkeypatch
) -> None:
    row = _run_single_baseline(
        tmp_path,
        monkeypatch,
        "bot_style",
        "Template for 1 3 4 6: first create 1 - 3 / 4, then divide 6.",
    )

    retrieved = row["retrieved_memory"][0]
    assert len(row["retrieved_memory"]) == 1
    assert len(row["retrieved_scores"]) == 1
    assert retrieved["entry_id"] == "bot_style_memory_1"
    assert retrieved["content"].startswith("Template for 1 3 4 6")
    assert retrieved["memory_type"] == "proxy_memory"
    assert retrieved["source_trial_id"] is None
    assert retrieved["metadata"] == {
        "task": "game24",
        "arm": "contaminated",
        "contamination_type": "proxy_memory",
    }
    prompt_text = "\n".join(message["content"] for message in row["prompt_messages"])
    assert "Distilled problem" in prompt_text
    assert "Retrieved thought template" in prompt_text
    assert "1. Key information:" in prompt_text
    assert "2. Restriction:" in prompt_text
    assert "3. Distilled task:" in prompt_text
    assert "4. Python transformation:" in prompt_text
    assert "5. Answer form:" in prompt_text
    assert "Apply the retrieved thought template" in prompt_text

    trial_id = row["trial_id"]
    assert len(row["memory_after"]) == len(row["memory_before"]) + 1
    new_entry = row["memory_after"][-1]
    assert new_entry["memory_type"] == "thought_template"
    assert new_entry["source_trial_id"] == trial_id
    assert "Problem Type: game24" in new_entry["content"]
    assert "Solution Strategy" in new_entry["content"]
    assert "final: 6 / (1 - 3 / 4)" not in new_entry["content"]

    assert row["memory_write_event"] == {
        "event_type": "bot_write",
        "baseline": "bot_style",
        "parent_trial_id": trial_id,
        "source_entry_ids": [retrieved["entry_id"]],
        "new_entry_id": new_entry["entry_id"],
        "update_reason": "distilled_thought_template_from_problem_solution_pair",
    }


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


def test_faithful_rag_bot_sequence_persists_and_logs(tmp_path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(repo_root)
    config = load_config(repo_root / "configs/g0_rag_bot_faithful_replay.yaml")
    config["logging"]["output_dir"] = str(tmp_path / "runs")
    config["models"] = ["gpt4o"]
    config["tasks"] = [{"name": "game24", "sample_path": "data/tasks/game24_pilot.jsonl", "limit": 2}]
    config["arms"] = ["clean"]

    run_dir = run_config(config, run_id="faithful_sequence")
    rows = [json.loads(line) for line in (run_dir / "trials.jsonl").read_text(encoding="utf-8").splitlines()]

    assert len(rows) == 4
    for row in rows:
        TrialLog.model_validate(row)
        assert row["method_calls"]

    rag_rows = [row for row in rows if row["baseline"] == "retrieval_rag"]
    assert rag_rows
    assert all(row["memory_write_event"] is None for row in rag_rows)
    assert all(row["memory_before"] == row["memory_after"] for row in rag_rows)
    assert rag_rows[0]["method_calls"][0]["stage"] == "rag_generate"
    assert rag_rows[0]["method_calls"][0]["retrieved_records"]

    bot_rows = [row for row in rows if row["baseline"] == "bot_style"]
    first_bot, second_bot = bot_rows
    accepted_id = first_bot["memory_write_event"]["new_entry_id"]
    assert first_bot["memory_write_event"]["status"] == "accepted"
    assert accepted_id in {entry["entry_id"] for entry in second_bot["memory_before"]}
    assert accepted_id in {entry["entry_id"] for entry in second_bot["retrieved_memory"]}
    assert [call["stage"] for call in second_bot["method_calls"]] == [
        "bot_problem_distill",
        "bot_instantiate_solve",
    ]


def test_faithful_bot_state_isolated_across_conditions(tmp_path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(repo_root)
    config = load_config(repo_root / "configs/g0_rag_bot_faithful_replay.yaml")
    config["logging"]["output_dir"] = str(tmp_path / "runs")
    config["models"] = ["gpt4o", "frontier_reasoning"]
    config["tasks"] = [{"name": "game24", "sample_path": "data/tasks/game24_pilot.jsonl", "limit": 2}]
    config["baselines"] = ["bot_style"]
    config["arms"] = ["clean", "contaminated"]

    run_dir = run_config(config, run_id="faithful_isolation")
    rows = [json.loads(line) for line in (run_dir / "trials.jsonl").read_text(encoding="utf-8").splitlines()]

    assert len(rows) == 8
    first_clean_gpt4o = next(
        row
        for row in rows
        if row["sample_id"] == "game24_pilot_001" and row["arm"] == "clean" and row["backbone"] == "gpt4o"
    )
    second_contaminated_gpt4o = next(
        row
        for row in rows
        if row["sample_id"] == "game24_pilot_002"
        and row["arm"] == "contaminated"
        and row["backbone"] == "gpt4o"
    )
    second_clean_frontier = next(
        row
        for row in rows
        if row["sample_id"] == "game24_pilot_002"
        and row["arm"] == "clean"
        and row["backbone"] == "frontier_reasoning"
    )

    clean_gpt4o_id = first_clean_gpt4o["memory_write_event"]["new_entry_id"]
    assert clean_gpt4o_id in {entry["entry_id"] for entry in rows[4]["memory_before"]}
    assert clean_gpt4o_id not in {entry["entry_id"] for entry in second_contaminated_gpt4o["memory_before"]}
    assert clean_gpt4o_id not in {entry["entry_id"] for entry in second_clean_frontier["memory_before"]}


def test_is_faithful_config_accepts_explicit_mode_and_rejects_unknown_mode() -> None:
    assert cli._is_faithful_config({"run": {"mode": "faithful"}})

    with pytest.raises(SystemExit, match="unsupported run.mode: unsupported"):
        cli._is_faithful_config({"run": {"mode": "unsupported"}})


def test_faithful_native_memory_config_dispatches_without_embedding_or_bot_resources(
    tmp_path, monkeypatch
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(repo_root)
    config = load_config(repo_root / "configs/g0_fh_reflexion_dc_faithful_replay.yaml")
    config["logging"]["output_dir"] = str(tmp_path / "runs")

    def unexpected_resource(*_args, **_kwargs):
        raise AssertionError("native-memory faithful run initialized a legacy resource")

    monkeypatch.setattr(cli, "SentenceTransformerProvider", unexpected_resource)
    monkeypatch.setattr(cli, "RunState", unexpected_resource)
    monkeypatch.setattr(cli, "BotRuntime", unexpected_resource)

    run_dir = run_config(config, run_id="native_memory_resource_free")
    rows = [json.loads(line) for line in (run_dir / "trials.jsonl").read_text(encoding="utf-8").splitlines()]

    assert len(rows) == 162
    assert {row["baseline"] for row in rows} == {
        "full_history",
        "reflexion_style",
        "dynamic_cheatsheet_optional",
    }
    assert all(row["retrieved_memory"] == [] for row in rows)
    assert all(row["retrieved_scores"] == [] for row in rows)


def test_faithful_native_memory_state_isolated_by_identity(tmp_path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(repo_root)
    config = load_config(repo_root / "configs/g0_fh_reflexion_dc_faithful_replay.yaml")
    config["logging"]["output_dir"] = str(tmp_path / "runs")
    config["models"] = ["gpt4o", "frontier_reasoning"]
    config["tasks"] = [{"name": "game24", "sample_path": "data/tasks/game24_pilot.jsonl", "limit": 2}]
    config["baselines"] = ["full_history"]
    config["arms"] = ["clean", "contaminated"]

    run_dir = run_config(config, run_id="native_memory_isolation")
    rows = [json.loads(line) for line in (run_dir / "trials.jsonl").read_text(encoding="utf-8").splitlines()]

    first_clean_gpt4o = next(
        row
        for row in rows
        if row["sample_id"] == "game24_pilot_001" and row["arm"] == "clean" and row["backbone"] == "gpt4o"
    )
    second_clean_gpt4o = next(
        row
        for row in rows
        if row["sample_id"] == "game24_pilot_002" and row["arm"] == "clean" and row["backbone"] == "gpt4o"
    )
    second_contaminated_gpt4o = next(
        row
        for row in rows
        if row["sample_id"] == "game24_pilot_002" and row["arm"] == "contaminated" and row["backbone"] == "gpt4o"
    )
    second_clean_frontier = next(
        row
        for row in rows
        if row["sample_id"] == "game24_pilot_002" and row["arm"] == "clean" and row["backbone"] == "frontier_reasoning"
    )

    first_entry_id = first_clean_gpt4o["memory_after"][-1]["entry_id"]
    assert first_entry_id in {entry["entry_id"] for entry in second_clean_gpt4o["memory_before"]}
    assert first_entry_id not in {entry["entry_id"] for entry in second_contaminated_gpt4o["memory_before"]}
    assert first_entry_id not in {entry["entry_id"] for entry in second_clean_frontier["memory_before"]}


def test_faithful_config_rejects_unknown_baseline(tmp_path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sample_path = _write_game24_sample(tmp_path)
    monkeypatch.chdir(repo_root)
    config = {
        "run": {"name": "smoke", "mode": "faithful"},
        "models": ["replay"],
        "tasks": [{"name": "game24", "sample_path": sample_path, "limit": 1}],
        "baselines": ["expel_optional"],
        "arms": ["clean"],
        "memory": {"corpus_path": str(repo_root / "data/memory/catalog_v2.jsonl")},
        "logging": {"output_dir": str(tmp_path / "runs")},
        "replay": {"responses": ["final: 6 / (1 - 3 / 4)"]},
    }

    with pytest.raises(SystemExit, match="unsupported faithful baseline: expel_optional"):
        run_config(config, run_id="unknown_faithful_baseline")


_NATIVE_MEMORY_BASELINES = [
    "full_history",
    "reflexion_style",
    "dynamic_cheatsheet_optional",
]

_NATIVE_MEMORY_GAME24_PAIRS = {
    "full_history": (
        "memory_clean_game24_full_history_001",
        "memory_corrupted_game24_full_history_001",
    ),
    "reflexion_style": (
        "memory_clean_game24_reflexion_style_001",
        "memory_corrupted_game24_reflexion_style_001",
    ),
    "dynamic_cheatsheet_optional": (
        "memory_clean_game24_dynamic_cheatsheet_optional_001",
        "memory_corrupted_game24_dynamic_cheatsheet_optional_001",
    ),
}


def _run_native_memory_config(tmp_path, monkeypatch, **overrides) -> list[dict]:
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(repo_root)
    config = load_config(repo_root / "configs/g0_fh_reflexion_dc_faithful_replay.yaml")
    config["logging"]["output_dir"] = str(tmp_path / "runs")
    config["models"] = ["gpt4o"]
    config["tasks"] = [{"name": "game24", "sample_path": "data/tasks/game24_pilot.jsonl", "limit": 2}]
    for key, value in overrides.items():
        config[key] = value
    run_dir = run_config(config, run_id="native_memory_test")
    return [json.loads(line) for line in (run_dir / "trials.jsonl").read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize("baseline", _NATIVE_MEMORY_BASELINES)
def test_native_memory_arm_semantics(baseline: str, tmp_path, monkeypatch) -> None:
    rows = _run_native_memory_config(
        tmp_path,
        monkeypatch,
        baselines=[baseline],
        arms=["clean", "contaminated", "contaminated_filter"],
    )

    clean_entry_id, corrupted_entry_id = _NATIVE_MEMORY_GAME24_PAIRS[baseline]
    arm_rows = {row["arm"]: row for row in rows if row["baseline"] == baseline}
    assert set(arm_rows) == {"clean", "contaminated", "contaminated_filter"}

    clean_row = arm_rows["clean"]
    assert clean_row["contamination_exposure"]["is_exposed"] is False
    assert clean_row["contamination_exposure"]["exposure_mode"] == "none"
    assert all(
        entry.get("clean_or_contaminated") != "contaminated"
        for entry in clean_row["memory_before"]
    )

    contaminated_row = arm_rows["contaminated"]
    assert contaminated_row["contamination_exposure"]["is_exposed"] is True
    assert contaminated_row["contamination_exposure"]["exposure_mode"] == "memory_before"
    assert corrupted_entry_id in {
        entry.get("entry_id") for entry in contaminated_row["memory_before"]
    }

    filter_row = arm_rows["contaminated_filter"]
    assert filter_row["filter_decision"] is not None
    assert filter_row["filter_decision"]["dropped"] > 0
    assert corrupted_entry_id not in {
        entry.get("entry_id") for entry in filter_row["memory_before"]
    }
    assert clean_entry_id in {
        entry.get("entry_id") for entry in filter_row["memory_before"]
    }
    assert filter_row["contamination_exposure"]["is_exposed"] is False


@pytest.mark.parametrize("baseline", _NATIVE_MEMORY_BASELINES)
def test_native_memory_retrieval_fields_empty(baseline: str, tmp_path, monkeypatch) -> None:
    rows = _run_native_memory_config(tmp_path, monkeypatch, baselines=[baseline])
    for row in rows:
        assert row["retrieved_memory"] == []
        assert row["retrieved_scores"] == []


def test_native_memory_gold_leakage(tmp_path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(repo_root)
    sample_path = tmp_path / "meb_canary.jsonl"
    sample_path.write_text(
        json.dumps(
            {
                "sample_id": "meb_canary_001",
                "input": "1 + 1 = ?",
                "verifier_spec": {
                    "target": "CANARY_EXPECTED_abc123",
                    "target_value": "CANARY_VALUE_xyz789",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    canary_reason = "CANARY_REASON_leakCheck"
    canary_expected = "CANARY_EXPECTED_abc123"
    canary_value = "CANARY_VALUE_xyz789"

    def _canary_verifier(parsed_answer: str, task):
        return VerifierResult(
            is_correct=True,
            parsed_answer=parsed_answer,
            reason=f"ok {canary_reason}",
            metadata={
                "expected": canary_expected,
                "target_value": canary_value,
            },
        )

    monkeypatch.setitem(cli.TASK_DISPATCH["math_equation_balancer"], "verify", _canary_verifier)

    config = load_config(repo_root / "configs/g0_fh_reflexion_dc_faithful_replay.yaml")
    config["logging"]["output_dir"] = str(tmp_path / "runs")
    config["models"] = ["gpt4o"]
    config["tasks"] = [{"name": "math_equation_balancer", "sample_path": str(sample_path), "limit": 1}]
    config["baselines"] = _NATIVE_MEMORY_BASELINES
    config["arms"] = ["clean", "contaminated", "contaminated_filter"]
    config["replay"] = {
        "responses_by_sample": {
            "meb_canary_001": {
                "full_history_generate": "final: 2 + 5 = 7",
                "reflexion_generate": "final: 2 + 5 = 7",
                "reflexion_reflect": "I should evaluate left to right.",
                "dynamic_cheatsheet_generate": "final: 2 + 5 = 7",
                "dynamic_cheatsheet_curate": "<cheatsheet>Evaluate addition left-to-right.</cheatsheet>",
            }
        }
    }

    run_dir = run_config(config, run_id="native_memory_gold_leakage")
    rows = [json.loads(line) for line in (run_dir / "trials.jsonl").read_text(encoding="utf-8").splitlines()]

    assert len(rows) == 9
    canaries = {canary_expected, canary_value, canary_reason}
    for row in rows:
        TrialLog.model_validate(row)
        assert canary_expected in row["gold_or_verifier_spec"].get("target", "")
        assert canary_value in row["gold_or_verifier_spec"].get("target_value", "")
        assert canary_reason in row["verifier_result"].get("reason", "")

        prompt_text = "\n".join(message["content"] for message in row["prompt_messages"])
        for canary in canaries:
            assert canary not in prompt_text
            for entry in row["memory_before"] + row["memory_after"]:
                assert canary not in str(entry.get("content", ""))


def test_native_memory_lineage_on_writes(tmp_path, monkeypatch) -> None:
    rows = _run_native_memory_config(
        tmp_path,
        monkeypatch,
        arms=["clean", "contaminated"],
    )

    for baseline in _NATIVE_MEMORY_BASELINES:
        baseline_rows = [row for row in rows if row["baseline"] == baseline]
        assert baseline_rows
        for row in baseline_rows:
            event = row["memory_write_event"]
            if event is None or event.get("status") != "accepted":
                continue
            assert "source_trial_id" in event
            assert "parent_entry_ids" in event
            assert "source_entry_ids" in event
            assert row["trial_id"] == event["source_trial_id"]

        for arm in ["clean", "contaminated"]:
            arm_rows = [row for row in baseline_rows if row["arm"] == arm]
            assert arm_rows
            for row in arm_rows:
                event = row["memory_write_event"]
                if event is None or event.get("status") != "accepted":
                    continue
                new_entry_id = event.get("new_entry_id")
                if not new_entry_id:
                    continue
                after_entries = {entry["entry_id"]: entry for entry in row["memory_after"]}
                if new_entry_id not in after_entries:
                    continue
                new_entry = after_entries[new_entry_id]
                if arm == "clean":
                    assert new_entry["clean_or_contaminated"] == "clean"
                else:
                    assert event["source_entry_ids"]
                    assert new_entry["clean_or_contaminated"] == "contaminated"


def test_v0_5_config_without_reflexion_block_validates(monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(repo_root)
    cli.validate_config(repo_root / "configs/g0_fh_reflexion_dc_faithful_replay.yaml")


def test_followup_config_validates(monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(repo_root)
    cli.validate_config(
        repo_root / "configs/g0_dc_rs_reflexion_fidelity_followup_replay.yaml"
    )


def test_reflexion_max_attempts_three_rejected(tmp_path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(repo_root)
    sample_path = _write_game24_sample(tmp_path)
    config = {
        "run": {"name": "smoke", "mode": "faithful"},
        "models": ["replay"],
        "tasks": [{"name": "game24", "sample_path": sample_path, "limit": 1}],
        "baselines": ["reflexion_style"],
        "arms": ["clean"],
        "memory": {"corpus_path": str(repo_root / "data/memory/catalog_v2.jsonl")},
        "reflexion": {"max_attempts": 3},
        "logging": {"output_dir": str(tmp_path / "runs")},
        "replay": {"responses": ["final: 6 / (1 - 3 / 4)"]},
    }

    with pytest.raises(SystemExit, match="reflexion.max_attempts must be 1 or 2"):
        run_config(config, run_id="bad_reflexion_attempts")


def test_dc_rs_reflexion_followup_gate_emits_108_rows_with_isolated_identities(
    tmp_path, monkeypatch
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(repo_root)
    config = load_config(
        repo_root / "configs/g0_dc_rs_reflexion_fidelity_followup_replay.yaml"
    )
    config["logging"]["output_dir"] = str(tmp_path / "runs")

    run_dir = run_config(config, run_id="followup_gate_test")
    rows = [
        json.loads(line)
        for line in (run_dir / "trials.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert len(rows) == 108
    assert {row["baseline"] for row in rows} == {
        "dynamic_cheatsheet_rs_optional",
        "reflexion_style",
    }
    dc_rs_rows = [row for row in rows if row["baseline"] == "dynamic_cheatsheet_rs_optional"]
    reflexion_rows = [row for row in rows if row["baseline"] == "reflexion_style"]
    assert len(dc_rs_rows) == 54
    assert len(reflexion_rows) == 54

    for row in dc_rs_rows:
        assert [call["stage"] for call in row["method_calls"]] == [
            "dc_rs_synthesize",
            "dc_rs_generate",
        ]
        assert row["memory_write_event"]["type"] == "dynamic_cheatsheet_rs_update"

    retry_rows = [row for row in reflexion_rows if len(row["method_calls"]) == 3]
    assert len(retry_rows) == 6
    for row in retry_rows:
        assert [call["stage"] for call in row["method_calls"]] == [
            "reflexion_generate",
            "reflexion_reflect",
            "reflexion_generate",
        ]
        assert row["sample_id"] == "game24_pilot_001"

    success_rows = [row for row in reflexion_rows if len(row["method_calls"]) == 1]
    assert len(success_rows) == 48

    first_clean_gpt4o = next(
        row
        for row in dc_rs_rows
        if row["sample_id"] == "game24_pilot_001"
        and row["arm"] == "clean"
        and row["backbone"] == "gpt4o"
    )
    second_clean_gpt4o = next(
        row
        for row in dc_rs_rows
        if row["sample_id"] == "game24_pilot_002"
        and row["arm"] == "clean"
        and row["backbone"] == "gpt4o"
    )
    second_contaminated_gpt4o = next(
        row
        for row in dc_rs_rows
        if row["sample_id"] == "game24_pilot_002"
        and row["arm"] == "contaminated"
        and row["backbone"] == "gpt4o"
    )

    first_pair_id = first_clean_gpt4o["memory_after"][-1]["entry_id"]
    assert first_pair_id in {
        entry["entry_id"] for entry in second_clean_gpt4o["memory_before"]
    }
    assert first_pair_id not in {
        entry["entry_id"] for entry in second_contaminated_gpt4o["memory_before"]
    }


def test_dc_rs_reflexion_followup_uses_offline_embeddings(tmp_path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(repo_root)
    config = load_config(
        repo_root / "configs/g0_dc_rs_reflexion_fidelity_followup_replay.yaml"
    )
    config["logging"]["output_dir"] = str(tmp_path / "runs")

    def unexpected_sentence_transformer(**_kwargs):
        raise AssertionError("follow-up replay gate must not load sentence-transformers")

    monkeypatch.setattr(cli, "SentenceTransformerProvider", unexpected_sentence_transformer)

    run_dir = run_config(config, run_id="followup_offline_embeddings")
    rows = [
        json.loads(line)
        for line in (run_dir / "trials.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert len(rows) == 108
    dc_rs_rows = [row for row in rows if row["baseline"] == "dynamic_cheatsheet_rs_optional"]
    assert dc_rs_rows
    assert all(row["method_calls"] for row in dc_rs_rows)
