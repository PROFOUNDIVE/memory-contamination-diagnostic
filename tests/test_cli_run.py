from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

import memcontam.cli as cli
from memcontam.cli import run_config
from memcontam.cli import load_config
from memcontam.clients.base import LLMResponse
from memcontam.logging.provenance import compute_exposure_from_spans, normalize_memory_event
from memcontam.logging.schema import PromptSourceSpan, TrialLog, VerifierResult
from memcontam.memory.embeddings import FakeEmbeddingProvider
from memcontam.memory.stores import MemoryEntry


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


def _write_minimal_corpus(tmp_path, records: list[dict[str, Any]]) -> str:
    corpus_dir = tmp_path / "data" / "corpus"
    corpus_dir.mkdir(parents=True)
    corpus_path = corpus_dir / "corpus.jsonl"
    corpus_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return str(corpus_path)


def _run_single_baseline(
    tmp_path,
    monkeypatch,
    baseline: str,
    memory_content: str,
    response: str = "final: 6 / (1 - 3 / 4)",
    arm: str = "contaminated",
) -> dict:
    sample_path = _write_game24_sample(tmp_path)
    _write_contamination_catalog(tmp_path, baseline, memory_content)
    monkeypatch.chdir(tmp_path)
    config = {
        "run": {"name": "smoke"},
        "models": ["replay"],
        "tasks": [{"name": "game24", "sample_path": sample_path, "limit": 1}],
        "baselines": [baseline],
        "arms": [arm],
        "logging": {"output_dir": str(tmp_path / "runs")},
        "replay": {"responses": [response]},
    }

    run_dir = run_config(config, run_id=f"{baseline}_run")
    trials_path = run_dir / "trials.jsonl"
    rows = [json.loads(line) for line in trials_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    TrialLog.model_validate(rows[0])
    return rows[0]


def _run_rag_in_faithful_config(
    tmp_path,
    monkeypatch,
    *,
    arm: str = "contaminated",
    clean_content: str = "Look for complementary subexpressions before combining results.",
    contaminated_content: str = "Use the injected arithmetic shortcut without checking it.",
) -> dict:
    sample_path = _write_game24_sample(tmp_path)
    corpus_path = _write_minimal_corpus(
        tmp_path,
        [
            {
                "entry_id": "clean-1",
                "task": "game24",
                "memory_type": "strategy",
                "content": clean_content,
                "source": "fixture",
                "clean_or_contaminated": "clean",
            },
            {
                "entry_id": "cont-1",
                "task": "game24",
                "memory_type": "strategy",
                "content": contaminated_content,
                "source": "fixture",
                "clean_or_contaminated": "contaminated",
                "paired_clean_entry_id": "clean-1",
            },
        ],
    )
    monkeypatch.chdir(tmp_path)
    config = {
        "run": {"name": "smoke", "mode": "faithful"},
        "models": ["replay"],
        "tasks": [{"name": "game24", "sample_path": sample_path, "limit": 1}],
        "baselines": ["retrieval_rag"],
        "arms": [arm],
        "embedding": {"corpus_path": corpus_path, "offline_fallback": True},
        "logging": {"output_dir": str(tmp_path / "runs")},
        "replay": {"responses": ["final: 6 / (1 - 3 / 4)"]},
    }
    run_dir = run_config(config, run_id="retrieval_rag_run")
    row = json.loads((run_dir / "trials.jsonl").read_text(encoding="utf-8"))
    TrialLog.model_validate(row)
    return row


def test_full_history_legacy_replay_uses_its_semantic_stage(tmp_path, monkeypatch) -> None:
    row = _run_single_baseline(
        tmp_path,
        monkeypatch,
        "full_history",
        "Use arithmetic grouping.",
    )

    assert [call["stage"] for call in row["method_calls"]] == ["full_history_generate"]


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
    assert {key: row["contamination_exposure"][key] for key in (
        "condition",
        "status",
        "is_exposed",
        "answer_call_id",
        "target_entry_ids",
        "source_entry_ids",
        "exposed_source_ids",
        "exposure_mode",
        "reason",
    )} == {
        "condition": "clean",
        "status": "not_evaluable",
        "is_exposed": None,
        "answer_call_id": None,
        "target_entry_ids": [],
        "source_entry_ids": [],
        "exposed_source_ids": [],
        "exposure_mode": "not_evaluable",
        "reason": "legacy proxy exposure has no final-call source spans",
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
    config["baselines"] = [
        baseline
        for baseline in config["baselines"]
        if baseline not in {"retrieval_rag", "bot_style"}
    ]

    run_dir = run_config(config, run_id="task_T9_clean_after_c")
    rows = [json.loads(line) for line in (run_dir / "trials.jsonl").read_text(encoding="utf-8").splitlines()]

    assert rows
    for row in rows:
        assert row["contamination_exposure"]["condition"] == "clean"
        assert row["contamination_exposure"]["source_entry_ids"] == []


def test_pilot_replay_dispatches_rag_without_legacy_prompt(tmp_path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(repo_root)
    config = load_config(repo_root / "configs/pilot_multitask_replay.yaml")
    config["logging"]["output_dir"] = str(tmp_path / "runs")
    config["models"] = ["gpt4o"]
    config["tasks"] = [{**config["tasks"][0], "limit": 1}]
    config["baselines"] = ["retrieval_rag"]

    assert cli._is_faithful_config(config)
    run_dir = run_config(config, run_id="pilot_rag_adapter")
    row = json.loads((run_dir / "trials.jsonl").read_text(encoding="utf-8"))

    assert (run_dir / "run.json").exists()
    assert row["baseline"] == "retrieval_rag"
    assert [call["stage"] for call in row["method_calls"]] == ["rag_generate"]


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
    row = _run_rag_in_faithful_config(tmp_path, monkeypatch)

    assert row["answer_call_id"] is not None
    call = next(call for call in row["method_calls"] if call["call_id"] == row["answer_call_id"])
    assert call["stage"] == "rag_generate"
    assert {span["entry_id"] for span in call["source_spans"]} == {"clean-1", "cont-1"}
    prompt_text = "\n".join(message["content"] for message in row["prompt_messages"])
    assert "Look for complementary subexpressions" in prompt_text
    assert "Use the injected arithmetic shortcut" in prompt_text
    assert "Current task:\n" in prompt_text
    assert "entry_id=clean-1" not in prompt_text
    assert "score=" not in prompt_text
    assert "memory_type=strategy" not in prompt_text
    assert "source_trial_id=" not in prompt_text
    assert "paired_clean_entry_id" not in prompt_text
    assert row["memory_write_event"] is None
    assert row["memory_before"] == row["memory_after"]


@pytest.mark.skip(reason="Task 14 owns legacy runner integration after Task 8 writes")
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
    row = _run_rag_in_faithful_config(tmp_path, monkeypatch, arm="contaminated_filter")

    assert row["filter_decision"]["filter_name"] == "drop_known_contaminated"
    assert row["filter_decision"]["input_count"] == 2
    assert row["filter_decision"]["removed_count"] == 1
    assert row["filter_decision"]["dropped"] == 1
    assert {
        (decision["ground_truth"], decision["action"])
        for decision in row["filter_decision"]["decisions"]
    } == {("clean", "kept"), ("contaminated", "removed")}
    assert [entry["entry_id"] for entry in row["memory_before"]] == ["clean-1"]
    assert row["contamination_exposure"]["condition"] == "contaminated_filter"
    assert row["contamination_exposure"]["status"] == "not_evaluable"
    assert row["contamination_exposure"]["is_exposed"] is None
    assert row["contamination_exposure"]["source_entry_ids"] == []
    assert row["contamination_exposure"]["exposure_mode"] == "not_evaluable"
    assert row["recovery_after_filter_label"] == "not_applicable"


def test_contaminated_row_with_no_retrieval_logs_memory_before_exposure(tmp_path, monkeypatch) -> None:
    row = _run_single_baseline(
        tmp_path,
        monkeypatch,
        "full_history",
        "Wrong reflection for 1 3 4 6: use 1 + 3 + 4 + 6.",
    )

    assert row["retrieved_memory"] == []
    assert {key: row["contamination_exposure"][key] for key in (
        "condition",
        "status",
        "is_exposed",
        "answer_call_id",
        "target_entry_ids",
        "source_entry_ids",
        "exposed_source_ids",
        "exposure_mode",
        "reason",
    )} == {
        "condition": "contaminated",
        "status": "not_evaluable",
        "is_exposed": None,
        "answer_call_id": None,
        "target_entry_ids": [],
        "source_entry_ids": ["full_history_memory_1"],
        "exposed_source_ids": [],
        "exposure_mode": "not_evaluable",
        "reason": "legacy proxy exposure has no final-call source spans",
    }


def test_controlled_exposure_does_not_imply_bad_memory_uptake(tmp_path, monkeypatch) -> None:
    row = _run_rag_in_faithful_config(tmp_path, monkeypatch)

    assert row["contamination_exposure"]["status"] == "not_evaluable"
    assert row["contamination_exposure"]["is_exposed"] is None
    assert row["contamination_exposure"]["source_entry_ids"]
    assert row["retrieved_memory"]
    assert row["bad_memory_uptake_label"] == "not_evaluable"
    assert row["bad_memory_uptake_label"] != "uptake_detected"
    assert row["memory_write_event"] is None


def test_no_memory_and_rag_rows_normalize_to_no_memory_event(
    tmp_path, monkeypatch
) -> None:
    for baseline in ("no_memory", "retrieval_rag"):
        baseline_tmp = tmp_path / baseline
        baseline_tmp.mkdir()
        if baseline == "retrieval_rag":
            row = _run_rag_in_faithful_config(baseline_tmp, monkeypatch)
        else:
            row = _run_single_baseline(
                baseline_tmp,
                monkeypatch,
                baseline,
                "Unused contaminated memory.",
                arm="clean",
            )

        before = [MemoryEntry.model_validate(entry) for entry in row["memory_before"]]
        after = [MemoryEntry.model_validate(entry) for entry in row["memory_after"]]
        event = normalize_memory_event(
            baseline,
            row["trial_id"],
            before,
            after,
            row["memory_write_event"],
        )
        assert event is None, f"{baseline} should not emit a memory event"


def test_failure_row_still_validates_with_provenance_labels(tmp_path, monkeypatch) -> None:
    row = _run_single_baseline(
        tmp_path,
        monkeypatch,
        "no_memory",
        "Unused contaminated memory.",
        "final: 1 + 3 + 4 + 6",
        arm="clean",
    )

    assert row["verifier_result"]["is_correct"] is False
    assert row["verifier_result"]["reason"]
    assert row["bad_memory_uptake_label"] == "not_applicable"
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


@pytest.mark.skip(reason="Task 8 owns faithful BoT write admission")
def test_faithful_rag_bot_sequence_persists_and_logs(tmp_path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(repo_root)
    config = load_config(repo_root / "configs/g0_rag_bot_faithful_replay.yaml")
    config["logging"]["output_dir"] = str(tmp_path / "runs")
    config["embedding"]["cache_path"] = str(tmp_path / "embedding_cache")
    assert config["embedding"]["cache_path"] == str(tmp_path / "embedding_cache")
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


@pytest.mark.skip(reason="Task 8 owns faithful BoT buffer mutation")
def test_faithful_bot_state_isolated_across_conditions(tmp_path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(repo_root)
    config = load_config(repo_root / "configs/g0_rag_bot_faithful_replay.yaml")
    config["logging"]["output_dir"] = str(tmp_path / "runs")
    config["embedding"]["cache_path"] = str(tmp_path / "embedding_cache")
    assert config["embedding"]["cache_path"] == str(tmp_path / "embedding_cache")
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
    assert cli._is_faithful_config({
        "run": {"mode": "faithful"},
        "baselines": ["no_memory"],
        "arms": ["clean"],
    })

    with pytest.raises(SystemExit, match="unsupported run.mode: unsupported"):
        cli._is_faithful_config({
            "run": {"mode": "unsupported"},
            "baselines": ["no_memory"],
            "arms": ["clean"],
        })


def test_baseline_outcome_conversion_preserves_read_only_bot_result() -> None:
    from memcontam.baselines.contracts import BaselineExecutionOutcome

    outcome = BaselineExecutionOutcome(
        status="succeeded",
        final_response="final: 24",
        parsed_answer="24",
        method_calls=("call-1",),
        memory_before=({"entry_id": "template-1"},),
        memory_after=({"entry_id": "template-1"},),
        metadata={"solution_trace": "Combine factors."},
    )

    result = cli._outcome_result_dict(outcome)

    assert result["final_response"] == "final: 24"
    assert result["method_calls"] == ["call-1"]
    assert result["memory_after"] == result["memory_before"]
    assert result["memory_write_event"] is None


_MAIN_BASELINES = [
    "no_memory",
    "full_history",
    "retrieval_rag",
    "reflexion_style",
    "bot_style",
]


def test_valid_arms_for_baseline_no_memory_is_clean_only() -> None:
    assert cli._valid_arms_for_baseline(
        "no_memory", ["clean", "contaminated", "contaminated_filter"]
    ) == ["clean"]
    assert cli._valid_arms_for_baseline("no_memory", ["contaminated", "contaminated_filter"]) == []
    assert cli._valid_arms_for_baseline("no_memory", ["clean"]) == ["clean"]


def test_valid_arms_for_baseline_memory_baselines_keep_all_requested_arms() -> None:
    arms = ["clean", "contaminated", "contaminated_filter"]
    for baseline in ["full_history", "retrieval_rag", "reflexion_style", "bot_style"]:
        assert cli._valid_arms_for_baseline(baseline, arms) == arms


def test_valid_arms_main_matrix_yields_thirteen_pairs_and_thirty_nine_task_combinations() -> None:
    arms = ["clean", "contaminated", "contaminated_filter"]
    valid_pairs = [
        (baseline, arm)
        for baseline in _MAIN_BASELINES
        for arm in cli._valid_arms_for_baseline(baseline, arms)
    ]
    assert len(valid_pairs) == 13
    assert len(valid_pairs) * 3 == 39


def test_validate_config_rejects_no_memory_only_without_clean_arm(tmp_path) -> None:
    sample_path = tmp_path / "game24_one.jsonl"
    sample_path.write_text(
        '{"sample_id":"sample_1","numbers":[1,3,4,6],"target":24}' + chr(10),
        encoding="utf-8",
    )
    config_path = tmp_path / "no_memory_only.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run": {"name": "smoke"},
                "models": ["replay"],
                "tasks": [
                    {"name": "game24", "sample_path": str(sample_path), "limit": 1}
                ],
                "baselines": ["no_memory"],
                "arms": ["contaminated", "contaminated_filter"],
                "logging": {"output_dir": str(tmp_path / "runs")},
                "replay": {"responses": ["final: 6 / (1 - 3 / 4)"]},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="zero valid"):
        cli.validate_config(config_path)


def test_run_config_no_memory_skips_contaminated_arms(tmp_path, monkeypatch) -> None:
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
        "baselines": ["no_memory"],
        "arms": ["clean", "contaminated", "contaminated_filter"],
        "logging": {"output_dir": str(tmp_path / "runs")},
        "replay": {"responses": ["final: 6 / (1 - 3 / 4)"]},
    }

    run_dir = run_config(config, run_id="no_memory_arm_filter")
    rows = [
        json.loads(line)
        for line in (run_dir / "trials.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert len(rows) == 1
    assert rows[0]["baseline"] == "no_memory"
    assert rows[0]["arm"] == "clean"
    assert rows[0]["filter_decision"] is None
    assert rows[0]["memory_write_event"] is None
    assert rows[0]["memory_before"] == []
    assert rows[0]["memory_after"] == []


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
    assert clean_row["contamination_exposure"]["status"] == "not_applicable"
    assert clean_row["contamination_exposure"]["is_exposed"] is None
    assert clean_row["contamination_exposure"]["exposure_mode"] == "clean"
    assert all(
        entry.get("clean_or_contaminated") != "contaminated"
        for entry in clean_row["memory_before"]
    )

    contaminated_row = arm_rows["contaminated"]
    assert contaminated_row["contamination_exposure"]["status"] == "supported"
    assert contaminated_row["contamination_exposure"]["is_exposed"] is True
    assert contaminated_row["contamination_exposure"]["source_entry_ids"]
    assert corrupted_entry_id in {
        entry.get("entry_id") for entry in contaminated_row["memory_before"]
    }

    filter_row = arm_rows["contaminated_filter"]
    assert filter_row["filter_decision"] is not None
    assert filter_row["filter_decision"]["removed_count"] > 0
    assert filter_row["filter_decision"]["dropped"] > 0
    assert corrupted_entry_id not in {
        entry.get("entry_id") for entry in filter_row["memory_before"]
    }
    assert clean_entry_id in {
        entry.get("entry_id") for entry in filter_row["memory_before"]
    }
    assert filter_row["contamination_exposure"]["status"] == "supported"
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
                elif new_entry["clean_or_contaminated"] == "contaminated":
                    assert event["source_entry_ids"]


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


def test_full_matrix_config_rejects_todo_limits(monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(repo_root)

    with pytest.raises(SystemExit, match="unresolved task limits"):
        cli.validate_config(repo_root / "configs/full_matrix.yaml")


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
    config["embedding"]["cache_path"] = str(tmp_path / "embedding_cache")
    assert config["embedding"]["cache_path"] == str(tmp_path / "embedding_cache")

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
    config["embedding"]["cache_path"] = str(tmp_path / "embedding_cache")
    assert config["embedding"]["cache_path"] == str(tmp_path / "embedding_cache")

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


def test_faithful_no_memory_run_has_answer_call_and_no_source_spans(tmp_path, monkeypatch) -> None:
    sample_path = _write_game24_sample(tmp_path)
    corpus_path = _write_minimal_corpus(tmp_path, [])
    monkeypatch.chdir(tmp_path)
    config = {
        "run": {"name": "smoke", "mode": "faithful"},
        "models": ["replay"],
        "tasks": [{"name": "game24", "sample_path": sample_path, "limit": 1}],
        "baselines": ["no_memory"],
        "arms": ["clean"],
        "embedding": {"corpus_path": corpus_path},
        "logging": {"output_dir": str(tmp_path / "runs")},
        "replay": {"responses": ["final: 6 / (1 - 3 / 4)"]},
    }

    run_dir = run_config(config, run_id="no_memory_faithful")
    row = json.loads((run_dir / "trials.jsonl").read_text(encoding="utf-8"))
    TrialLog.model_validate(row)

    assert row["baseline"] == "no_memory"
    assert row["arm"] == "clean"
    assert row["answer_call_id"] is not None
    assert len(row["method_calls"]) == 1
    call = row["method_calls"][0]
    assert call["call_id"] == row["answer_call_id"]
    assert call["stage"] == "no_memory_generate"
    assert call["source_spans"] == []
    assert row["memory_before"] == []
    assert row["memory_after"] == []
    assert row["retrieved_memory"] == []
    assert row["memory_write_event"] is None
    assert row["filter_decision"] is None


def test_faithful_rag_contaminated_filter_false_positive_not_in_final_prompt(
    tmp_path, monkeypatch
) -> None:
    sample_path = _write_game24_sample(tmp_path)
    corpus_records = [
        {
            "entry_id": "game24_clean_001",
            "task": "game24",
            "memory_type": "memory_seed",
            "content": "A general strategy for Game24 is to look for ways to make 1 from subtraction or division.",
            "source": "pilot_corpus",
            "clean_or_contaminated": "clean",
        },
        {
            "entry_id": "game24_cont_001",
            "task": "game24",
            "memory_type": "memory_seed",
            "content": "A contaminated hint that should be filtered.",
            "source": "pilot_corpus",
            "clean_or_contaminated": "contaminated",
            "paired_clean_entry_id": "game24_clean_001",
        },
    ]
    corpus_path = _write_minimal_corpus(tmp_path, corpus_records)
    monkeypatch.chdir(tmp_path)
    config = {
        "run": {"name": "smoke", "mode": "faithful"},
        "models": ["replay"],
        "tasks": [{"name": "game24", "sample_path": sample_path, "limit": 1}],
        "baselines": ["retrieval_rag"],
        "arms": ["contaminated_filter"],
        "embedding": {
            "corpus_path": corpus_path,
            "top_k": 1,
            "offline_fallback": True,
        },
        "logging": {"output_dir": str(tmp_path / "runs")},
        "replay": {"responses": ["final: 6 / (1 - 3 / 4)"]},
    }

    run_dir = run_config(config, run_id="rag_filter_false_positive")
    row = json.loads((run_dir / "trials.jsonl").read_text(encoding="utf-8"))
    TrialLog.model_validate(row)

    assert row["baseline"] == "retrieval_rag"
    assert row["arm"] == "contaminated_filter"
    assert row["answer_call_id"] is not None
    call = next(call for call in row["method_calls"] if call["call_id"] == row["answer_call_id"])
    spans = [PromptSourceSpan.model_validate(span) for span in call["source_spans"]]
    assert all(span.clean_or_contaminated == "clean" for span in spans)
    assert all(span.entry_id != "game24_cont_001" for span in spans)
    exposure = compute_exposure_from_spans(row["answer_call_id"], spans, "contaminated_filter")
    assert exposure.status == "supported"
    assert exposure.is_exposed is False
    assert exposure.exposure_mode == "not_in_final_prompt"


def test_strict_faithful_replay_writes_manifest_calls_and_canonical_trial(
    tmp_path, monkeypatch
) -> None:
    sample_path = _write_game24_sample(tmp_path)
    corpus_path = _write_minimal_corpus(tmp_path, [])
    monkeypatch.chdir(tmp_path)
    config = {
        "run": {
            "name": "strict-smoke",
            "mode": "faithful",
            "stage": "replay",
            "provider": "replay",
            "model_snapshots": {"replay": "fixture-v1"},
            "task_order_seed": 7,
            "sample_order_seed": 11,
            "retry_policy_version": "retry-v1",
        },
        "models": ["replay"],
        "tasks": [{"name": "game24", "sample_path": sample_path, "limit": 1}],
        "baselines": ["no_memory"],
        "arms": ["clean"],
        "memory": {"corpus_path": corpus_path},
        "logging": {
            "output_dir": str(tmp_path / "runs"),
            "schema_version": "logging_v1",
            "prompt_version": "prompt-v1",
            "memory_policy_version": "memory-v1",
            "contamination_catalog_version": "catalog-v1",
        },
        "replay": {"fixture_version": "fixture-v1", "responses": ["final: 6 / (1 - 3 / 4)"]},
        "live_smoke": {"enabled": False},
    }

    run_dir = run_config(config, run_id="strict_replay")

    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    calls = [
        json.loads(line)
        for line in (run_dir / "calls.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    trials = [
        json.loads(line)
        for line in (run_dir / "trials.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert manifest["status"] == "completed"
    assert manifest["counts"] == {
        "calls": 1,
        "failures": 0,
        "filter_events": 0,
        "memory_events": 0,
        "trials": 1,
    }
    assert len(calls) == len(trials) == 1
    trial = trials[0]
    assert trial["schema_version"] == "logging_v1"
    assert trial["status"] == "succeeded"
    assert trial["run_metadata_id"] == manifest["run_metadata"]["run_metadata_id"]
    assert trial["answer_call_id"] == calls[0]["call_id"]
    assert trial["prompt_messages"] == calls[0]["messages"]
    assert trial["latency_ms"] == calls[0]["latency_ms"]
    assert trial["token_usage"] == calls[0]["token_usage"]
    assert trial["retry_count"] == calls[0]["retry_count"]
    assert trial["event_seq"] > calls[0]["event_seq"]


def test_strict_faithful_runner_continues_after_provider_and_verifier_failures(
    tmp_path, monkeypatch
) -> None:
    sample_path = tmp_path / "game24_three.jsonl"
    sample_path.write_text(
        "".join(
            json.dumps({"sample_id": f"sample_{index}", "numbers": [1, 3, 4, 6], "target": 24})
            + "\n"
            for index in range(1, 4)
        ),
        encoding="utf-8",
    )
    corpus_path = _write_minimal_corpus(tmp_path, [])
    monkeypatch.chdir(tmp_path)

    class FlakyClient:
        def __init__(self) -> None:
            self.calls = 0

        def chat(self, messages, model, config):
            self.calls += 1
            if self.calls == 1:
                raise ConnectionError("provider failure")
            return LLMResponse(
                content="final: 6 / (1 - 3 / 4)",
                raw={},
                token_usage={"total_tokens": 3},
                latency_ms=5,
            )

    original_verify = cli.TASK_DISPATCH["game24"]["verify"]
    verify_calls = 0

    def flaky_verify(parsed_answer, task):
        nonlocal verify_calls
        verify_calls += 1
        if verify_calls == 1:
            raise ValueError("verifier failure")
        return original_verify(parsed_answer, task)

    monkeypatch.setitem(cli.TASK_DISPATCH["game24"], "verify", flaky_verify)
    config = {
        "run": {
            "name": "strict-failures",
            "mode": "faithful",
            "stage": "replay",
            "provider": "replay",
            "model_snapshots": {"replay": "fixture-v1"},
            "task_order_seed": 7,
            "sample_order_seed": 11,
            "retry_policy_version": "retry-v1",
        },
        "models": ["replay"],
        "tasks": [{"name": "game24", "sample_path": str(sample_path), "limit": 3}],
        "baselines": ["no_memory"],
        "arms": ["clean"],
        "memory": {"corpus_path": corpus_path},
        "logging": {
            "output_dir": str(tmp_path / "runs"),
            "schema_version": "logging_v1",
            "prompt_version": "prompt-v1",
            "memory_policy_version": "memory-v1",
            "contamination_catalog_version": "catalog-v1",
        },
        "replay": {"fixture_version": "fixture-v1"},
        "live_smoke": {"enabled": False},
    }

    run_dir = run_config(config, run_id="strict_failures", _client_override=FlakyClient())

    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    calls = [
        json.loads(line)
        for line in (run_dir / "calls.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    failures = [
        json.loads(line)
        for line in (run_dir / "failures.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    trials = [
        json.loads(line)
        for line in (run_dir / "trials.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert manifest["status"] == "completed"
    assert manifest["counts"] == {
        "calls": 3,
        "failures": 2,
        "filter_events": 0,
        "memory_events": 0,
        "trials": 3,
    }
    assert [failure["origin"] for failure in failures] == ["provider_call", "verifier"]
    assert [trial["status"] for trial in trials] == ["failed", "failed", "succeeded"]
    for trial in trials[:2]:
        assert trial["raw_response"] is None
        assert trial["parsed_answer"] is None
        assert trial["verifier_result"] is None
        assert trial["failure_id"] in {failure["failure_id"] for failure in failures}
        assert trial["answer_call_id"] in {call["call_id"] for call in calls}
    assert trials[-1]["verifier_result"]["is_correct"] is True


def _strict_no_memory_config(tmp_path, sample_path: str, corpus_path: str) -> dict[str, Any]:
    return {
        "run": {
            "name": "strict-negative",
            "mode": "faithful",
            "stage": "replay",
            "provider": "replay",
            "model_snapshots": {"replay": "fixture-v1"},
            "task_order_seed": 7,
            "sample_order_seed": 11,
            "retry_policy_version": "retry-v1",
        },
        "models": ["replay"],
        "tasks": [{"name": "game24", "sample_path": sample_path, "limit": 1}],
        "baselines": ["no_memory"],
        "arms": ["clean"],
        "memory": {"corpus_path": corpus_path},
        "logging": {
            "output_dir": str(tmp_path / "runs"),
            "schema_version": "logging_v1",
            "prompt_version": "prompt-v1",
            "memory_policy_version": "memory-v1",
            "contamination_catalog_version": "catalog-v1",
        },
        "replay": {"fixture_version": "fixture-v1", "responses": ["final: 6 / (1 - 3 / 4)"]},
        "live_smoke": {"enabled": False},
    }


def _phase11_config(tmp_path, sample_path: str, corpus_path: str) -> dict[str, Any]:
    config = _strict_no_memory_config(tmp_path, sample_path, corpus_path)
    config["run"]["contract_level"] = "phase11"
    config["logging"]["schema_version"] = "logging_v2"
    config["evaluation"] = {
        "evaluation_law_id": "law-online-v1",
        "regime": "online",
        "task_law_id": "task-law-v1",
        "inference_law_id": "inference-law-v1",
        "checkpoint_policy_id": None,
    }
    config["target_contamination_set"] = {
        "target_set_id": "controlled_injected_derived_v1",
        "definition_version": "phase11-v1",
        "included_classes": ["injected", "derived"],
        "require_exact_lineage": True,
    }
    config["embedding"] = {
        "corpus_path": corpus_path,
        "top_k": 1,
        "offline_fallback": True,
        "cache_path": str(tmp_path / "embedding_cache"),
    }
    return config


def test_phase11_runner_writes_law_target_pairing_and_update_context(
    tmp_path, monkeypatch
) -> None:
    sample_path = _write_game24_sample(tmp_path)
    corpus_path = _write_minimal_corpus(
        tmp_path,
        [
            {
                "entry_id": "game24_clean_001",
                "task": "game24",
                "memory_type": "memory_seed",
                "content": "Look for fractional complements when solving Game24.",
                "source": "test",
                "clean_or_contaminated": "clean",
                "contamination_class": "clean",
                "lineage_status": "exact",
                "lineage_basis": "seed",
            },
            {
                "entry_id": "game24_injected_001",
                "task": "game24",
                "memory_type": "memory_seed",
                "content": "Injected hint: prefer a misleading addition route.",
                "source": "test",
                "clean_or_contaminated": "contaminated",
                "paired_clean_entry_id": "game24_clean_001",
                "contamination_class": "injected",
                "lineage_status": "exact",
                "lineage_basis": "seed",
                "injected_root_ids": ["game24_injected_001"],
            },
        ],
    )
    monkeypatch.chdir(tmp_path)
    config = _phase11_config(tmp_path, sample_path, corpus_path)
    config["models"] = ["replay", "replay_alt"]
    config["run"]["model_snapshots"] = {"replay": "fixture-v1", "replay_alt": "fixture-v2"}
    config["baselines"] = ["no_memory", "retrieval_rag"]
    config["arms"] = ["clean", "contaminated", "contaminated_filter"]
    config["replay"] = {"responses_by_sample": {"sample_1": "final: 6 / (1 - 3 / 4)"}}

    run_dir = run_config(config, run_id="phase11_context")

    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    trials = [
        json.loads(line)
        for line in (run_dir / "trials.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert manifest["run_metadata"]["evaluation_law"]["evaluation_law_id"] == "law-online-v1"
    assert manifest["run_metadata"]["target_contamination_set"]["target_set_id"] == "controlled_injected_derived_v1"
    assert len(trials) == 8
    for trial in trials:
        TrialLog.model_validate(trial)
        assert trial["schema_version"] == "logging_v2"
        assert trial["evaluation_law_id"] == "law-online-v1"
        assert trial["target_set_id"] == "controlled_injected_derived_v1"
        assert trial["checkpoint_ref"] is None
        assert trial["checkpoint_index"] == 0
        assert trial["pair_id"] == ":".join(
            [trial["trajectory_pair_id"], str(trial["checkpoint_index"]), trial["sample_id"]]
        )

    no_memory_rows = [trial for trial in trials if trial["baseline"] == "no_memory"]
    assert {trial["memory_update_mode"] for trial in no_memory_rows} == {"not_applicable"}

    rag_rows = [trial for trial in trials if trial["baseline"] == "retrieval_rag"]
    assert {trial["memory_update_mode"] for trial in rag_rows} == {"enabled"}
    assert len({trial["pair_id"] for trial in rag_rows if trial["backbone"] == "replay"}) == 1
    assert len({trial["pair_id"] for trial in rag_rows}) == 2


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda config: config.pop("evaluation"), "logging_v2 requires evaluation"),
        (lambda config: config["evaluation"].update(regime="stage"), "invalid evaluation.regime"),
        (lambda config: config["memory"].update(update_mode="disabled"), "memory.update_mode"),
        (
            lambda config: config.update(checkpoint_ref={"checkpoint_id": "ckpt-1"}),
            "checkpoint_ref.checkpoint_trial_index",
        ),
        (lambda config: config.update(baselines=["full_history"]), "frozen.*memory-writing"),
    ],
)
def test_phase11_frozen_and_law_validation_fail_closed_before_run_dir(
    tmp_path, monkeypatch, mutate, message
) -> None:
    sample_path = _write_game24_sample(tmp_path)
    corpus_path = _write_minimal_corpus(tmp_path, [])
    monkeypatch.chdir(tmp_path)
    config = _phase11_config(tmp_path, sample_path, corpus_path)
    config["evaluation"]["regime"] = "frozen"
    config["evaluation"]["evaluation_law_id"] = "law-frozen-v1"
    config["evaluation"]["checkpoint_policy_id"] = "checkpoint-policy-v1"
    config["checkpoint_ref"] = {
        "checkpoint_id": "ckpt-1",
        "checkpoint_trial_index": 0,
        "checkpoint_memory_hash": "abc123",
        "checkpoint_source_run_id": "source-run",
    }
    mutate(config)

    class UnexpectedClient:
        def chat(self, messages, model, config):
            raise AssertionError("provider should not be constructed or called")

    with pytest.raises(SystemExit, match=message):
        run_config(config, run_id="phase11_invalid_frozen", _client_override=UnexpectedClient())

    assert not (tmp_path / "runs" / "phase11_invalid_frozen").exists()


def _phase11_frozen_config(tmp_path, sample_path: str, corpus_path: str) -> dict[str, Any]:
    config = _phase11_config(tmp_path, sample_path, corpus_path)
    config["evaluation"]["regime"] = "frozen"
    config["evaluation"]["evaluation_law_id"] = "law-frozen-v1"
    config["evaluation"]["checkpoint_policy_id"] = "checkpoint-policy-v1"
    config["baselines"] = ["no_memory", "retrieval_rag"]
    config["arms"] = ["clean", "contaminated"]
    config["checkpoint_ref"] = {
        "checkpoint_id": "ckpt-1",
        "checkpoint_trial_index": 7,
        "checkpoint_memory_hash": "checkpoint-hash-abc123",
        "checkpoint_source_run_id": "source-run-1",
        "artifact_path": None,
    }
    config["replay"] = {"responses_by_sample": {"sample_1": "final: 6 / (1 - 3 / 4)"}}
    return config


def test_phase11_frozen_read_only_run_logs_checkpoint_context(
    tmp_path, monkeypatch
) -> None:
    sample_path = _write_game24_sample(tmp_path)
    corpus_path = _write_minimal_corpus(
        tmp_path,
        [
            {
                "entry_id": "game24_clean_001",
                "task": "game24",
                "memory_type": "memory_seed",
                "content": "Use careful arithmetic grouping for Game24.",
                "source": "test",
                "clean_or_contaminated": "clean",
                "contamination_class": "clean",
                "lineage_status": "exact",
                "lineage_basis": "seed",
            },
            {
                "entry_id": "game24_injected_001",
                "task": "game24",
                "memory_type": "memory_seed",
                "content": "Injected distractor: try simple addition first.",
                "source": "test",
                "clean_or_contaminated": "contaminated",
                "paired_clean_entry_id": "game24_clean_001",
                "contamination_class": "injected",
                "lineage_status": "exact",
                "lineage_basis": "seed",
                "injected_root_ids": ["game24_injected_001"],
            },
        ],
    )
    monkeypatch.chdir(tmp_path)
    config = _phase11_frozen_config(tmp_path, sample_path, corpus_path)

    run_dir = run_config(config, run_id="phase11_frozen_read_only")

    trials = [
        json.loads(line)
        for line in (run_dir / "trials.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    memory_events = (run_dir / "memory_events.jsonl").read_text(encoding="utf-8")
    assert memory_events == ""

    no_memory = [trial for trial in trials if trial["baseline"] == "no_memory"]
    assert len(no_memory) == 1
    assert no_memory[0]["memory_update_mode"] == "not_applicable"
    assert no_memory[0]["checkpoint_ref"] is None
    assert no_memory[0]["memory_write_event"] is None
    assert no_memory[0]["memory_before"] == no_memory[0]["memory_after"] == []

    rag_rows = [trial for trial in trials if trial["baseline"] == "retrieval_rag"]
    assert {trial["arm"] for trial in rag_rows} == {"clean", "contaminated"}
    for trial in rag_rows:
        TrialLog.model_validate(trial)
        assert trial["memory_update_mode"] == "disabled"
        assert trial["checkpoint_ref"] == config["checkpoint_ref"]
        assert trial["memory_write_event"] is None
        assert trial["memory_before"] == trial["memory_after"]
        assert trial["checkpoint_index"] == 0
        assert trial["pair_id"] == ":".join(
            [trial["trajectory_pair_id"], str(trial["checkpoint_index"]), trial["sample_id"]]
        )


def test_phase11_frozen_disabled_memory_drift_fails_closed(
    tmp_path, monkeypatch
) -> None:
    sample_path = _write_game24_sample(tmp_path)
    corpus_path = _write_minimal_corpus(
        tmp_path,
        [
            {
                "entry_id": "game24_clean_001",
                "task": "game24",
                "memory_type": "memory_seed",
                "content": "Use careful arithmetic grouping for Game24.",
                "source": "test",
                "clean_or_contaminated": "clean",
                "contamination_class": "clean",
                "lineage_status": "exact",
                "lineage_basis": "seed",
            }
        ],
    )
    monkeypatch.chdir(tmp_path)
    config = _phase11_frozen_config(tmp_path, sample_path, corpus_path)
    config["baselines"] = ["retrieval_rag"]
    config["arms"] = ["clean"]
    original_policy = cli.RetrievalRagPolicy

    class MutatingRetrievalRagPolicy:
        def run(self, *args, **kwargs):
            result = original_policy().run(*args, **kwargs)
            result["memory_after"] = [
                *result["memory_after"],
                {
                    "entry_id": "illegal-frozen-write",
                    "content": "mutated",
                    "memory_type": "memory_seed",
                    "clean_or_contaminated": "clean",
                    "source_trial_id": None,
                    "metadata": {},
                },
            ]
            result["memory_write_event"] = {"type": "illegal_frozen_write"}
            return result

    monkeypatch.setattr(cli, "RetrievalRagPolicy", MutatingRetrievalRagPolicy)

    with pytest.raises(SystemExit, match="frozen logging_v2 trial changed memory"):
        run_config(config, run_id="phase11_frozen_drift")

    assert not (tmp_path / "runs" / "phase11_frozen_drift").exists()


def test_strict_failed_multicall_trial_keeps_answer_call_not_failed_curation(
    tmp_path, monkeypatch
) -> None:
    sample_path = _write_game24_sample(tmp_path)
    corpus_path = _write_minimal_corpus(tmp_path, [])
    monkeypatch.chdir(tmp_path)

    class CurationFailureClient:
        def __init__(self) -> None:
            self.calls = 0

        def chat(self, messages, model, config):
            self.calls += 1
            if self.calls == 2:
                raise ConnectionError("curation failure")
            return LLMResponse(
                content="final: 6 / (1 - 3 / 4)",
                raw={},
                token_usage={"total_tokens": 3},
                latency_ms=5,
            )

    config = _strict_no_memory_config(tmp_path, sample_path, corpus_path)
    config["baselines"] = ["dynamic_cheatsheet_optional"]
    run_dir = run_config(config, run_id="strict_curation_failure", _client_override=CurationFailureClient())

    calls = [
        json.loads(line)
        for line in (run_dir / "calls.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    trial = json.loads((run_dir / "trials.jsonl").read_text(encoding="utf-8"))
    assert [call["method_stage"] for call in calls] == [
        "dynamic_cheatsheet_generate",
        "dynamic_cheatsheet_curate",
    ]
    assert trial["status"] == "failed"
    assert trial["answer_call_id"] == calls[0]["call_id"]
    assert trial["prompt_messages"] == calls[0]["messages"]
    assert trial["method_calls"][0]["stage"] == "dynamic_cheatsheet_generate"
    assert trial["method_calls"][1]["stage"] == "dynamic_cheatsheet_curate"


def test_strict_run_rejects_collision_without_changing_existing_directory(tmp_path, monkeypatch) -> None:
    sample_path = _write_game24_sample(tmp_path)
    corpus_path = _write_minimal_corpus(tmp_path, [])
    monkeypatch.chdir(tmp_path)
    config = _strict_no_memory_config(tmp_path, sample_path, corpus_path)
    run_dir = tmp_path / "runs" / "strict_collision"
    run_dir.mkdir(parents=True)
    sentinel = run_dir / "keep.txt"
    sentinel.write_text("unchanged", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        run_config(config, run_id="strict_collision")

    assert sentinel.read_text(encoding="utf-8") == "unchanged"
    assert not list(run_dir.parent.glob("strict_collision.tmp-*"))


def test_strict_fatal_writer_error_never_finalizes_completed_run(tmp_path, monkeypatch) -> None:
    sample_path = _write_game24_sample(tmp_path)
    corpus_path = _write_minimal_corpus(tmp_path, [])
    monkeypatch.chdir(tmp_path)
    config = _strict_no_memory_config(tmp_path, sample_path, corpus_path)

    def fail_trial_write(self, trial):
        raise OSError("disk full")

    monkeypatch.setattr(cli.RunLogWriter, "write_trial", fail_trial_write)
    with pytest.raises(OSError, match="disk full"):
        run_config(config, run_id="strict_writer_failure")

    final_dir = tmp_path / "runs" / "strict_writer_failure"
    temp_dirs = list(final_dir.parent.glob("strict_writer_failure.tmp-*"))
    assert not final_dir.exists()
    assert len(temp_dirs) == 1
    manifest = json.loads((temp_dirs[0] / "run.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"


def test_strict_offline_config_rejects_live_smoke_before_client_creation(tmp_path, monkeypatch) -> None:
    sample_path = _write_game24_sample(tmp_path)
    corpus_path = _write_minimal_corpus(tmp_path, [])
    config = _strict_no_memory_config(tmp_path, sample_path, corpus_path)
    config["live_smoke"]["enabled"] = True

    with pytest.raises(SystemExit, match="live_smoke.enabled=false"):
        run_config(config, run_id="strict_live_smoke")

    assert not (tmp_path / "runs" / "strict_live_smoke").exists()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda config: config["run"].update(stage="unsupported"), "unsupported run.stage"),
        (
            lambda config: config["run"].update(model_snapshots={"replay": "TODO"}),
            "resolved snapshot",
        ),
        (lambda config: config["tasks"][0].update(limit=0), "positive task limit"),
        (
            lambda config: config["run"].update(mode="legacy", stage="partial"),
            "legacy run.mode",
        ),
    ],
)
def test_run_config_rejects_invalid_strict_or_legacy_stage_boundaries(
    tmp_path, mutate, message
) -> None:
    sample_path = _write_game24_sample(tmp_path)
    corpus_path = _write_minimal_corpus(tmp_path, [])
    config = _strict_no_memory_config(tmp_path, sample_path, corpus_path)
    mutate(config)

    with pytest.raises(SystemExit, match=message):
        run_config(config, run_id="invalid_strict_boundary")

    assert not (tmp_path / "runs" / "invalid_strict_boundary").exists()
