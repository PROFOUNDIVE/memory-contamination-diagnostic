from __future__ import annotations

import copy
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import pytest

import memcontam.cli as cli
from memcontam.clients.base import LLMResponse
from memcontam.evaluation.aggregate import aggregate_run
from memcontam.logging.provenance import compute_exposure_from_spans_v2
from memcontam.logging.schema import (
    LOGGING_V2,
    CallEvent,
    FailureEvent,
    FilterEvent,
    MemoryEvent,
    MemoryItemLog,
    PromptSourceSpan,
    RunMetadata,
    TargetContaminationSetSpec,
    TrialLog,
)
from memcontam.memory.stores import MemoryEntry


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "logging_contract_phase11_replay.yaml"
ARTIFACT_FILENAMES = (
    "run.json",
    "trials.jsonl",
    "calls.jsonl",
    "failures.jsonl",
    "filter_events.jsonl",
    "memory_events.jsonl",
)
SENTINELS = (
    "SENTINEL_API_KEY_phase11_contract",
    "SENTINEL_AUTHORIZATION_HEADER_phase11_contract",
    "SENTINEL_RAW_OBJECT_phase11_contract",
)
EVIDENCE_MARKER = "EVIDENCE_ONLY_PHASE11_CONTRACT"


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


def _phase11_config(tmp_path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    config = copy.deepcopy(cli.load_config(CONFIG_PATH))
    responses_by_sample = config["replay"].pop("responses_by_sample")
    config["logging"]["output_dir"] = str(tmp_path / "runs")
    config["embedding"]["cache_path"] = str(tmp_path / "embedding_cache")
    return config, responses_by_sample


def _read_artifacts(run_dir: Path) -> dict[str, Any]:
    return {
        "manifest": json.loads((run_dir / "run.json").read_text(encoding="utf-8")),
        "trials": _jsonl(run_dir / "trials.jsonl"),
        "calls": _jsonl(run_dir / "calls.jsonl"),
        "failures": _jsonl(run_dir / "failures.jsonl"),
        "filters": _jsonl(run_dir / "filter_events.jsonl"),
        "memory_events": _jsonl(run_dir / "memory_events.jsonl"),
    }


def _contains_ancestor_closure(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            ("ancestor" in key and ("id" in key or "path" in key))
            or _contains_ancestor_closure(nested)
            for key, nested in value.items()
        )
    if isinstance(value, list):
        return any(_contains_ancestor_closure(item) for item in value)
    return False


def _assert_no_evidence_exposure(artifacts: dict[str, Any]) -> None:
    runtime_data = {
        "trials": artifacts["trials"],
        "calls": artifacts["calls"],
        "memory_events": artifacts["memory_events"],
    }
    assert EVIDENCE_MARKER not in json.dumps(runtime_data, sort_keys=True)


def _assert_direct_edge_contract(trials: list[TrialLog], memory_events: list[MemoryEvent]) -> None:
    events_by_trial: dict[str, list[MemoryEvent]] = defaultdict(list)
    for event in memory_events:
        events_by_trial[event.trial_id].append(event)

    expected_edges: set[tuple[str, str, str]] = set()
    actual_edges: list[tuple[str, str, str]] = []
    for trial in trials:
        after_by_id = {entry["entry_id"]: entry for entry in trial.memory_after}
        for event in events_by_trial[trial.trial_id]:
            for child_id in [*event.new_entry_ids, *event.updated_entry_ids]:
                metadata = after_by_id[child_id].get("metadata", {})
                if metadata.get("lineage_status") != "exact":
                    continue
                for parent_id in metadata.get("direct_parent_ids", []):
                    expected_edges.add((trial.trial_id, child_id, parent_id))
            actual_edges.extend(
                (trial.trial_id, edge.child_entry_id, edge.parent_entry_id)
                for edge in event.lineage_edges
                if edge.lineage_status == "exact"
                and not (edge.relation == "version_edge" and edge.lineage_basis == "version_edge")
            )

    assert len(actual_edges) == len(set(actual_edges)), "duplicate exact direct edge"
    assert set(actual_edges) == expected_edges, "missing or non-direct exact lineage edge"


def _assert_online_phase11_contract(run_dir: Path) -> None:
    assert {path.name for path in run_dir.iterdir()} == set(ARTIFACT_FILENAMES) | {
        "provider_profile.json",
        "resolved_config.json",
    }
    artifacts = _read_artifacts(run_dir)
    manifest = artifacts["manifest"]
    metadata = RunMetadata.model_validate(manifest["run_metadata"])
    trials = [TrialLog.model_validate(row) for row in artifacts["trials"]]
    calls = [CallEvent.model_validate(row) for row in artifacts["calls"]]
    failures = [FailureEvent.model_validate(row) for row in artifacts["failures"]]
    filters = [FilterEvent.model_validate(row) for row in artifacts["filters"]]
    memory_events = [MemoryEvent.model_validate(row) for row in artifacts["memory_events"]]

    assert manifest["status"] == "completed"
    assert metadata.schema_version == LOGGING_V2
    assert metadata.contract_level == "phase11"
    assert metadata.stage == "replay"
    assert metadata.evaluation_law is not None
    assert metadata.evaluation_law.regime == "online"
    assert metadata.target_contamination_set is not None
    assert manifest["counts"] == {
        "trials": len(trials),
        "calls": len(calls),
        "failures": len(failures),
        "filter_events": len(filters),
        "memory_events": len(memory_events),
    }
    assert len(trials) == 39
    assert all(trial.status == "succeeded" and trial.verifier_result for trial in trials)
    assert not failures
    identities = {(trial.task_name, trial.baseline, trial.arm, trial.backbone) for trial in trials}
    assert len(identities) == len(trials)
    assert {trial.task_name for trial in trials} == {
        "game24",
        "math_equation_balancer",
        "word_sorting",
    }
    assert {trial.backbone for trial in trials} == {"replay_logging_contract_phase11"}
    assert len([trial for trial in trials if trial.baseline == "no_memory"]) == 3
    assert all(trial.arm == "clean" for trial in trials if trial.baseline == "no_memory")
    assert {trial.evaluation_law_id for trial in trials} == {
        metadata.evaluation_law.evaluation_law_id
    }
    assert {trial.target_set_id for trial in trials} == {
        metadata.target_contamination_set.target_set_id
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

    trial_ids = {trial.trial_id for trial in trials}
    assert {call.trial_id for call in calls} == trial_ids
    assert {event.trial_id for event in filters}.issubset(trial_ids)
    assert {event.trial_id for event in memory_events}.issubset(trial_ids)
    for record in [*calls, *failures, *filters, *memory_events]:
        assert record.run_id == metadata.run_id
        assert record.run_metadata_id == metadata.run_metadata_id
        assert record.stage == metadata.stage

    event_sequences = [
        *(call.event_seq for call in calls),
        *(event.event_seq for event in filters),
        *(event.event_seq for event in memory_events),
        *(trial.event_seq for trial in trials),
    ]
    assert sorted(event_sequences) == list(range(1, len(event_sequences) + 1))

    target_set = metadata.target_contamination_set
    for trial in trials:
        assert trial.answer_call_id is not None
        assert trial.run_id == metadata.run_id
        assert trial.run_metadata_id == metadata.run_metadata_id
        assert trial.schema_version == LOGGING_V2
        assert trial.stage == metadata.stage
        assert trial.pair_id == ":".join(
            [trial.trajectory_pair_id or "", str(trial.checkpoint_index), trial.sample_id]
        )
        assert trial.memory_update_mode in {"enabled", "not_applicable"}
        assert trial.checkpoint_ref is None
        answer_call = calls_by_id[trial.answer_call_id]
        trial_calls = calls_by_trial[trial.trial_id]
        assert [call.call_id for call in trial_calls] == [
            call.call_id for call in trial.method_calls
        ]
        assert trial.prompt_messages == answer_call.messages
        assert trial.raw_response == answer_call.response_text
        assert trial.verifier_result is not None
        assert trial.parsed_answer == trial.verifier_result.parsed_answer
        assert trial.latency_ms == sum(call.latency_ms or 0 for call in trial_calls)
        assert trial.token_usage == {
            key: sum(call.token_usage.get(key, 0) for call in trial_calls)
            for key in {key for call in trial_calls for key in call.token_usage}
        }
        assert trial.retry_count == max(call.retry_count for call in trial_calls)
        assert trial.event_seq is not None
        trial_event_seq = cast(int, trial.event_seq)
        assert all(call.event_seq is not None for call in trial_calls)
        call_event_seqs = [cast(int, call.event_seq) for call in trial_calls]
        assert all(event_seq < trial_event_seq for event_seq in call_event_seqs)

        span_ids = [span.entry_id for span in answer_call.source_spans]
        assert trial.contamination_exposure.source_entry_ids == span_ids
        expected_targets = [
            entry["entry_id"]
            for entry in trial.memory_before
            if entry.get("metadata", {}).get("contamination_class") in target_set.included_classes
            and entry.get("metadata", {}).get("lineage_status") == "exact"
        ]
        assert trial.contamination_exposure.target_entry_ids == expected_targets
        for span in answer_call.source_spans:
            assert span.target_set_id == trial.target_set_id
            assert span.is_target_contamination == (
                span.contamination_class in target_set.included_classes
                and span.lineage_status == "exact"
            )

        trial_filters = filters_by_trial[trial.trial_id]
        if trial.arm == "contaminated_filter":
            assert [event.action for event in trial_filters] == ["apply", "outcome"]
            assert all(event.event_seq is not None for event in trial_filters)
            filter_event_seqs = [cast(int, event.event_seq) for event in trial_filters]
            assert filter_event_seqs[0] < min(call_event_seqs)
            assert max(call_event_seqs) < filter_event_seqs[1] < trial_event_seq
            assert trial_filters[1].final_answer_source_ids == span_ids
        else:
            assert not trial_filters

        trial_memory_events = memory_by_trial[trial.trial_id]
        if trial.memory_write_event is None:
            assert not trial_memory_events
        else:
            assert len(trial_memory_events) == 1
            event = trial_memory_events[0]
            assert event.event_seq is not None
            assert cast(int, event.event_seq) < trial_event_seq
            assert event.baseline == trial.baseline
            assert event.source_trial_id == trial.trial_id
            assert event.before_entry_ids == [entry["entry_id"] for entry in trial.memory_before]
            assert event.after_entry_ids == [entry["entry_id"] for entry in trial.memory_after]

    pairs: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for trial in trials:
        pairs[(trial.task_name, trial.baseline, trial.backbone)].add(trial.pair_id or "")
    assert all(len(pair_ids) == 1 for pair_ids in pairs.values())

    memory_items = [
        MemoryItemLog.from_memory_entry(MemoryEntry.model_validate(entry))
        for trial in trials
        for entry in [*trial.memory_before, *trial.memory_after]
        if "memory_type" in entry
    ]
    classes = {item.contamination_class for item in memory_items}
    assert {"clean", "injected"}.issubset(classes)
    assert classes.issubset({"clean", "injected", "derived", "natural"})
    injected_ids = {
        item.entry_id for item in memory_items if item.contamination_class == "injected"
    }
    for item in memory_items:
        if item.contamination_class == "injected":
            assert item.injected_root_ids == [item.entry_id]
        if item.contamination_class == "derived" and item.lineage_status == "exact":
            assert item.direct_parent_ids
            assert set(item.injected_root_ids).issubset(injected_ids)
        if item.contamination_class == "natural":
            assert not item.injected_root_ids

    _assert_direct_edge_contract(trials, memory_events)
    assert not any(
        _contains_ancestor_closure(value)
        for trial in trials
        for value in (trial.memory_before, trial.memory_after, trial.memory_write_event)
    ), "materialized ancestor closure"
    _assert_no_evidence_exposure(artifacts)
    for filename in ARTIFACT_FILENAMES:
        artifact = (run_dir / filename).read_text(encoding="utf-8")
        assert not any(sentinel in artifact for sentinel in SENTINELS)
    assert aggregate_run(run_dir, stage="replay", contract="phase11")["n_trials"] == 39


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_phase11_online_replay_emits_exact_contract(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(ROOT)
    config, responses_by_sample = _phase11_config(tmp_path)
    run_dir = cli.run_config(
        config,
        run_id="phase11_online_contract_gate",
        _client_override=_SentinelReplayClient(responses_by_sample),
    )

    _assert_online_phase11_contract(run_dir)


def test_phase11_frozen_replay_is_checkpointed_read_only_and_rejects_writers(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(ROOT)
    config, responses_by_sample = _phase11_config(tmp_path)
    checkpoint = {
        "checkpoint_id": "phase11-frozen-checkpoint",
        "checkpoint_trial_index": 7,
        "checkpoint_memory_hash": "sha256:phase11-frozen-memory",
        "checkpoint_source_run_id": "phase11-source-run",
        "artifact_path": None,
    }
    config["evaluation"].update(
        regime="frozen",
        evaluation_law_id="phase11_logging_contract_frozen_replay_v1",
        checkpoint_policy_id="phase11-frozen-policy-v1",
    )
    config["baselines"] = ["no_memory", "retrieval_rag"]
    config["arms"] = ["clean", "contaminated"]
    config["checkpoint_ref"] = checkpoint
    run_dir = cli.run_config(
        config,
        run_id="phase11_frozen_contract_gate",
        _client_override=_SentinelReplayClient(responses_by_sample),
    )

    artifacts = _read_artifacts(run_dir)
    trials = [TrialLog.model_validate(row) for row in artifacts["trials"]]
    assert len(trials) == 9
    assert artifacts["memory_events"] == []
    rag_trials = [trial for trial in trials if trial.baseline == "retrieval_rag"]
    assert len(rag_trials) == 6
    assert {trial.memory_update_mode for trial in rag_trials} == {"disabled"}
    checkpoints = []
    for trial in rag_trials:
        assert trial.checkpoint_ref is not None
        checkpoints.append(json.dumps(trial.checkpoint_ref.model_dump(), sort_keys=True))
    assert set(checkpoints) == {json.dumps(checkpoint, sort_keys=True)}
    assert all(trial.memory_before == trial.memory_after for trial in rag_trials)
    assert all(trial.memory_write_event is None for trial in rag_trials)
    no_memory_trials = [trial for trial in trials if trial.baseline == "no_memory"]
    assert len(no_memory_trials) == 3
    assert {trial.memory_update_mode for trial in no_memory_trials} == {"not_applicable"}
    assert all(trial.checkpoint_ref is None for trial in no_memory_trials)

    writer_config = copy.deepcopy(config)
    writer_config["baselines"] = ["full_history"]
    with pytest.raises(SystemExit, match="frozen logging_v2 rejects memory-writing baselines"):
        cli.run_config(
            writer_config,
            run_id="phase11_frozen_writer_rejected",
            _client_override=_SentinelReplayClient(responses_by_sample),
        )


def test_phase11_negative_contract_cases_fail_at_the_named_invariant(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(ROOT)
    config, responses_by_sample = _phase11_config(tmp_path)
    run_dir = cli.run_config(
        config,
        run_id="phase11_negative_contract_source",
        _client_override=_SentinelReplayClient(responses_by_sample),
    )
    artifacts = _read_artifacts(run_dir)
    trials = [TrialLog.model_validate(row) for row in artifacts["trials"]]
    memory_events = [MemoryEvent.model_validate(row) for row in artifacts["memory_events"]]
    edge_event = next(event for event in memory_events if event.lineage_edges)

    missing_edge = copy.deepcopy(memory_events)
    next(
        event for event in missing_edge if event.memory_id == edge_event.memory_id
    ).lineage_edges.pop()
    with pytest.raises(AssertionError, match="missing or non-direct exact lineage edge"):
        _assert_direct_edge_contract(trials, missing_edge)

    duplicate_edge = copy.deepcopy(memory_events)
    duplicate = next(event for event in duplicate_edge if event.memory_id == edge_event.memory_id)
    duplicate.lineage_edges.append(copy.deepcopy(duplicate.lineage_edges[0]))
    with pytest.raises(AssertionError, match="duplicate exact direct edge"):
        _assert_direct_edge_contract(trials, duplicate_edge)

    closure_artifacts = copy.deepcopy(artifacts)
    memory_trial = next(row for row in closure_artifacts["trials"] if row["memory_before"])
    memory_trial["memory_before"][0]["metadata"]["ancestor_ids"] = ["synthetic-ancestor"]
    closure_run = tmp_path / "phase11_ancestor_closure"
    shutil.copytree(run_dir, closure_run)
    _write_jsonl(closure_run / "trials.jsonl", closure_artifacts["trials"])
    with pytest.raises(AssertionError, match="materialized ancestor closure"):
        _assert_online_phase11_contract(closure_run)

    evidence_artifacts = copy.deepcopy(artifacts)
    evidence_trial = next(row for row in evidence_artifacts["trials"] if row["memory_before"])
    evidence_trial["memory_before"][0]["metadata"]["provenance_record"] = EVIDENCE_MARKER
    evidence_run = tmp_path / "phase11_evidence_exposure"
    shutil.copytree(run_dir, evidence_run)
    _write_jsonl(evidence_run / "trials.jsonl", evidence_artifacts["trials"])
    with pytest.raises(AssertionError, match=EVIDENCE_MARKER):
        _assert_online_phase11_contract(evidence_run)

    target_set = TargetContaminationSetSpec(
        target_set_id="controlled_injected_derived_v1",
        definition_version="phase11-v1",
        included_classes=["injected", "derived"],
        require_exact_lineage=True,
    )
    span = PromptSourceSpan.model_validate(
        {
            "message_index": 0,
            "start": 0,
            "end": 1,
            "rendered_hash": "sha256:phase11",
            "entry_id": "approximate-derived",
            "source_ids": ["approximate-derived"],
            "parent_ids": [],
            "lineage_id": "approximate-derived",
            "version": "v2",
            "origin": "gate",
            "clean_or_contaminated": "contaminated",
            "contamination_class": "derived",
            "injected_root_ids": [],
            "lineage_status": "approximate",
            "lineage_basis": "signature",
            "direct_parent_ids": [],
            "target_set_id": target_set.target_set_id,
            "is_target_contamination": False,
        }
    )
    approximate = compute_exposure_from_spans_v2(
        "answer-call",
        [span],
        "contaminated",
        [
            {
                "entry_id": span.entry_id,
                "content": "approximate candidate",
                "memory_type": "memory_seed",
                "clean_or_contaminated": "contaminated",
                "metadata": {
                    "contamination_class": "derived",
                    "lineage_status": "approximate",
                    "lineage_basis": "signature",
                    "direct_parent_ids": [],
                    "injected_root_ids": [],
                },
            }
        ],
        target_set,
    )
    assert approximate.status == "not_evaluable"
    assert approximate.evidence_lineage_status == "approximate"

    natural_span = span.model_copy(
        update={
            "entry_id": "natural-only",
            "source_ids": ["natural-only"],
            "lineage_id": "natural-only",
            "clean_or_contaminated": "clean",
            "contamination_class": "natural",
            "lineage_status": "exact",
            "lineage_basis": "recorded_parent",
        }
    )
    natural = compute_exposure_from_spans_v2(
        "answer-call",
        [natural_span],
        "contaminated",
        [
            {
                "entry_id": natural_span.entry_id,
                "content": "natural candidate",
                "memory_type": "memory_seed",
                "clean_or_contaminated": "clean",
                "metadata": {
                    "contamination_class": "natural",
                    "lineage_status": "exact",
                    "lineage_basis": "recorded_parent",
                    "direct_parent_ids": [],
                    "injected_root_ids": [],
                },
            }
        ],
        target_set,
    )
    assert natural.status == "supported"
    assert natural.is_exposed is False
    assert natural.target_entry_ids == natural.exposed_entry_ids == []

    for field, value, message in (
        ("evaluation_law", "other-law", "mixed evaluation_law_id"),
        ("target_contamination_set", "other-target", "mixed target_set_id"),
    ):
        tampered = tmp_path / f"phase11_{value}"
        shutil.copytree(run_dir, tampered)
        manifest = json.loads((tampered / "run.json").read_text(encoding="utf-8"))
        context_id = {
            "evaluation_law": "evaluation_law_id",
            "target_contamination_set": "target_set_id",
        }[field]
        manifest["run_metadata"][field][context_id] = value
        _write_json(tampered / "run.json", manifest)
        with pytest.raises(SystemExit, match=message):
            aggregate_run(tampered, stage="replay", contract="phase11")

    v1_as_phase11 = tmp_path / "phase10_requested_as_phase11"
    shutil.copytree(run_dir, v1_as_phase11)
    manifest = json.loads((v1_as_phase11 / "run.json").read_text(encoding="utf-8"))
    manifest["run_metadata"]["schema_version"] = "logging_v1"
    manifest["run_metadata"]["contract_level"] = "phase10"
    _write_json(v1_as_phase11 / "run.json", manifest)
    with pytest.raises(SystemExit, match="contract mismatch: requested phase11, found phase10"):
        aggregate_run(v1_as_phase11, stage="replay", contract="phase11")

    frozen_config, frozen_responses = _phase11_config(tmp_path)
    frozen_config["evaluation"].update(
        regime="frozen",
        evaluation_law_id="phase11_mutation_frozen_v1",
        checkpoint_policy_id="phase11-frozen-policy-v1",
    )
    frozen_config["baselines"] = ["retrieval_rag"]
    frozen_config["arms"] = ["clean"]
    frozen_config["checkpoint_ref"] = {
        "checkpoint_id": "phase11-mutation-checkpoint",
        "checkpoint_trial_index": 0,
        "checkpoint_memory_hash": "sha256:phase11-mutation",
        "checkpoint_source_run_id": "phase11-source-run",
    }
    original_policy = cli.RetrievalRagPolicy

    class _MutatingRetrievalRagPolicy:
        def run(self, *args, **kwargs):
            result = original_policy().run(*args, **kwargs)
            result["memory_after"] = [*result["memory_after"], {"entry_id": "illegal"}]
            result["memory_write_event"] = {"type": "illegal_frozen_write"}
            return result

    monkeypatch.setattr(cli, "RetrievalRagPolicy", _MutatingRetrievalRagPolicy)
    with pytest.raises(SystemExit, match="frozen logging_v2 trial changed memory"):
        cli.run_config(
            frozen_config,
            run_id="phase11_frozen_mutation_rejected",
            _client_override=_SentinelReplayClient(frozen_responses),
        )
