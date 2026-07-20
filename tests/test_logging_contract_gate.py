from __future__ import annotations

import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import memcontam.cli as cli
from memcontam.clients.base import LLMResponse
from memcontam.logging.provenance import memory_snapshot_hash
from memcontam.logging.schema import (
    LOGGING_V1,
    CallEvent,
    FailureEvent,
    FilterEvent,
    MemoryEvent,
    RunMetadata,
    TrialLog,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "logging_contract_replay.yaml"
ARTIFACT_FILENAMES = (
    "run.json",
    "trials.jsonl",
    "calls.jsonl",
    "failures.jsonl",
    "filter_events.jsonl",
    "memory_events.jsonl",
)
SENTINELS = (
    "SENTINEL_API_KEY_logging_contract",
    "SENTINEL_AUTHORIZATION_HEADER_logging_contract",
    "SENTINEL_RAW_OBJECT_logging_contract",
)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class _SentinelReplayClient:
    def __init__(self, responses_by_sample: dict[str, dict[str, Any]]) -> None:
        self._responses_by_sample = responses_by_sample
        self._response_indices: dict[tuple[str, str, str, str], int] = defaultdict(int)

    def chat(
        self, messages: list[dict[str, str]], model: str, config: dict[str, Any]
    ) -> LLMResponse:
        del messages, model
        sample_id = config["sample_id"]
        stage = config["method_stage"]
        response = self._responses_by_sample[sample_id][stage]
        if isinstance(response, list):
            key = (sample_id, config["baseline"], config["arm"], stage)
            response = response[self._response_indices[key]]
            self._response_indices[key] += 1
        return LLMResponse(
            content=response,
            raw={
                "api_key": SENTINELS[0],
                "headers": {"Authorization": SENTINELS[1]},
                "raw_object": SENTINELS[2],
            },
            token_usage={"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
            latency_ms=7,
        )


def _strict_contract_config(tmp_path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    config = copy.deepcopy(cli.load_config(CONFIG_PATH))
    responses_by_sample = config["replay"].pop("responses_by_sample")
    config["logging"]["output_dir"] = str(tmp_path / "runs")
    config["embedding"]["cache_path"] = str(tmp_path / "embedding_cache")
    return config, responses_by_sample


def _answer_call(calls_by_id: dict[str, CallEvent], trial: TrialLog) -> CallEvent:
    assert trial.answer_call_id is not None
    return calls_by_id[trial.answer_call_id]


def _source_ids_from_spans(call: CallEvent) -> list[str]:
    return [source_id for span in call.source_spans for source_id in (span.source_ids or [span.entry_id])]


def _memory_snapshot(entries: list[dict[str, Any]]) -> str:
    return memory_snapshot_hash(cli._snapshot_memory_entries(entries))


def test_logging_contract_replay_emits_exact_strict_39_row_artifacts(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(ROOT)
    config, responses_by_sample = _strict_contract_config(tmp_path)
    assert config["embedding"]["cache_path"] == str(tmp_path / "embedding_cache")
    run_dir = cli.run_config(
        config,
        run_id="logging_contract_gate",
        _client_override=_SentinelReplayClient(responses_by_sample),
    )

    manifest_data = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    trial_data = _jsonl(run_dir / "trials.jsonl")
    call_data = _jsonl(run_dir / "calls.jsonl")
    failure_data = _jsonl(run_dir / "failures.jsonl")
    filter_data = _jsonl(run_dir / "filter_events.jsonl")
    memory_data = _jsonl(run_dir / "memory_events.jsonl")

    metadata = RunMetadata.model_validate(manifest_data["run_metadata"])
    trials = [TrialLog.model_validate(row) for row in trial_data]
    calls = [CallEvent.model_validate(row) for row in call_data]
    failures = [FailureEvent.model_validate(row) for row in failure_data]
    filters = [FilterEvent.model_validate(row) for row in filter_data]
    memory_events = [MemoryEvent.model_validate(row) for row in memory_data]

    assert manifest_data["status"] == "completed"
    assert metadata.stage == "replay"
    assert metadata.schema_version == LOGGING_V1
    assert metadata.provider.startswith("replay:")
    assert config["embedding"]["offline_fallback"] is True
    assert config["live_smoke"]["enabled"] is False
    assert manifest_data["counts"] == {
        "trials": len(trials),
        "calls": len(calls),
        "failures": len(failures),
        "filter_events": len(filters),
        "memory_events": len(memory_events),
    }

    combinations = {(trial.task_name, trial.baseline, trial.arm) for trial in trials}
    assert len(trials) == len(combinations) == 39
    assert all(trial.status == "succeeded" and trial.verifier_result for trial in trials)
    assert {trial.task_name for trial in trials} == {
        "game24",
        "math_equation_balancer",
        "word_sorting",
    }
    assert {trial.backbone for trial in trials} == {"replay_logging_contract"}
    assert len([trial for trial in trials if trial.baseline == "no_memory"]) == 3
    assert all(trial.arm == "clean" for trial in trials if trial.baseline == "no_memory")
    assert len([trial for trial in trials if trial.baseline != "no_memory"]) == 36
    assert not {
        (trial.task_name, trial.baseline, trial.arm)
        for trial in trials
        if trial.baseline == "no_memory" and trial.arm != "clean"
    }

    calls_by_id = {call.call_id: call for call in calls}
    calls_by_trial: dict[str, list[CallEvent]] = defaultdict(list)
    filters_by_trial: dict[str, list[FilterEvent]] = defaultdict(list)
    memory_by_trial: dict[str, list[MemoryEvent]] = defaultdict(list)
    for call in calls:
        calls_by_trial[call.trial_id].append(call)
    for event in filters:
        filters_by_trial[event.trial_id].append(event)
    for event in memory_events:
        memory_by_trial[event.trial_id].append(event)

    assert not failures
    assert {call.trial_id for call in calls} == {trial.trial_id for trial in trials}
    assert {event.trial_id for event in filters}.issubset({trial.trial_id for trial in trials})
    assert {event.trial_id for event in memory_events}.issubset({trial.trial_id for trial in trials})

    all_event_sequences = [
        *(call.event_seq for call in calls),
        *(event.event_seq for event in filters),
        *(event.event_seq for event in memory_events),
        *(trial.event_seq for trial in trials),
    ]
    assert len(all_event_sequences) == len(set(all_event_sequences))
    assert sorted(all_event_sequences) == list(range(1, len(all_event_sequences) + 1))

    for trial in trials:
        assert trial.verifier_result is not None
        assert trial.event_seq is not None
        trial_event_seq = trial.event_seq
        assert trial.run_id == metadata.run_id
        assert trial.run_metadata_id == metadata.run_metadata_id
        assert trial.schema_version == LOGGING_V1
        assert trial.stage == metadata.stage
        answer_call = _answer_call(calls_by_id, trial)
        trial_calls = calls_by_trial[trial.trial_id]
        assert [call.call_id for call in trial_calls] == [call.call_id for call in trial.method_calls]
        assert trial.prompt_messages == answer_call.messages
        assert trial.raw_response == answer_call.response_text
        assert trial.parsed_answer == trial.verifier_result.parsed_answer
        assert trial.latency_ms == sum(call.latency_ms or 0 for call in trial_calls)
        assert trial.token_usage == {
            key: sum(call.token_usage.get(key, 0) for call in trial_calls)
            for key in {key for call in trial_calls for key in call.token_usage}
        }
        assert trial.retry_count == max(call.retry_count for call in trial_calls)
        assert all(call.event_seq < trial_event_seq for call in trial_calls)

        exposure = trial.contamination_exposure
        if trial.arm == "clean":
            assert exposure.status == "not_applicable"
            assert exposure.is_exposed is None
            assert exposure.exposure_mode == "clean"
        else:
            assert exposure.status == "supported"
            assert exposure.answer_call_id == trial.answer_call_id
            if exposure.is_exposed:
                assert answer_call.source_spans
                assert set(exposure.exposed_source_ids).issubset(
                    {source_id for span in answer_call.source_spans for source_id in [span.entry_id, *span.source_ids]}
                )
                assert exposure.exposure_mode == "final_prompt"
            else:
                assert exposure.exposure_mode == "not_in_final_prompt"

        trial_filters = filters_by_trial[trial.trial_id]
        if trial.arm == "contaminated_filter":
            assert trial.filter_decision is not None
            assert [event.action for event in trial_filters] == ["apply", "outcome"]
            apply, outcome = trial_filters
            assert apply.event_seq < min(call.event_seq for call in trial_calls)
            assert max(call.event_seq for call in trial_calls) < outcome.event_seq < trial_event_seq
            assert apply.decisions == outcome.decisions == trial.filter_decision["decisions"]
            contaminated_ids = [
                decision["entry_id"]
                for decision in trial.filter_decision["decisions"]
                if decision["ground_truth"] == "contaminated"
            ]
            assert apply.ground_truth_contaminated_ids == contaminated_ids
            assert apply.pre_source_ids == trial.filter_decision["input_source_ids"]
            assert apply.post_source_ids == trial.filter_decision["kept_source_ids"]
            assert apply.removed_source_ids == trial.filter_decision["removed_source_ids"]
            assert outcome.final_answer_source_ids == _source_ids_from_spans(answer_call)
            assert outcome.verdict == str(trial.verifier_result.is_correct).lower()
        else:
            assert trial.filter_decision is None
            assert not trial_filters

        trial_memory_events = memory_by_trial[trial.trial_id]
        if trial.memory_write_event is None:
            assert not trial_memory_events
        else:
            assert len(trial_memory_events) == 1
            event = trial_memory_events[0]
            assert event.event_seq < trial_event_seq
            assert event.baseline == trial.baseline
            assert event.source_trial_id == trial.trial_id
            assert event.before_entry_ids == [entry["entry_id"] for entry in trial.memory_before]
            assert event.after_entry_ids == [entry["entry_id"] for entry in trial.memory_after]
            assert event.before_snapshot_hash == _memory_snapshot(trial.memory_before)
            assert event.after_snapshot_hash == _memory_snapshot(trial.memory_after)
            assert set(event.new_entry_ids).issubset(set(event.after_entry_ids))

    for filename in ARTIFACT_FILENAMES:
        artifact = (run_dir / filename).read_text(encoding="utf-8")
        assert not any(sentinel in artifact for sentinel in SENTINELS)


def test_logging_contract_failure_continues_to_later_strict_trial(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(ROOT)
    sample_path = tmp_path / "game24_two.jsonl"
    sample_path.write_text(
        "\n".join(
            json.dumps({"sample_id": f"contract_failure_{index}", "numbers": [1, 3, 4, 6], "target": 24})
            for index in (1, 2)
        )
        + "\n",
        encoding="utf-8",
    )
    config, _ = _strict_contract_config(tmp_path)
    config["tasks"] = [{"name": "game24", "sample_path": str(sample_path), "limit": 2}]
    config["baselines"] = ["no_memory"]
    config["arms"] = ["clean"]

    class _FlakyClient:
        def __init__(self) -> None:
            self.calls = 0

        def chat(
            self, messages: list[dict[str, str]], model: str, config: dict[str, Any]
        ) -> LLMResponse:
            del messages, model, config
            self.calls += 1
            if self.calls == 1:
                raise ConnectionError("intentional contract continuation failure")
            return LLMResponse(
                content="final: 6 / (1 - (3 / 4))",
                raw={},
                token_usage={"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
                latency_ms=7,
            )

    run_dir = cli.run_config(
        config,
        run_id="logging_contract_failure_continuation",
        _client_override=_FlakyClient(),
    )
    trials = [TrialLog.model_validate(row) for row in _jsonl(run_dir / "trials.jsonl")]
    calls = [CallEvent.model_validate(row) for row in _jsonl(run_dir / "calls.jsonl")]
    failures = [FailureEvent.model_validate(row) for row in _jsonl(run_dir / "failures.jsonl")]

    assert [trial.status for trial in trials] == ["failed", "succeeded"]
    assert len(calls) == len(trials) == 2
    assert len(failures) == 1
    assert failures[0].origin == "provider_call"
    assert trials[0].failure_id == failures[0].failure_id
    assert trials[0].raw_response is None
    assert trials[0].parsed_answer is None
    assert trials[0].verifier_result is None
    assert trials[1].verifier_result is not None and trials[1].verifier_result.is_correct
