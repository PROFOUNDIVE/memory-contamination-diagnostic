from typing import Any

import pytest
from pydantic import ValidationError

from memcontam.clients.replay import ReplayClient
from memcontam.logging.schema import (
    CallEvent,
    CheckpointRef,
    ContaminationExposure,
    EventContext,
    FailureEvent,
    FilterEvent,
    LineageEdge,
    LOGGING_V1,
    LOGGING_V2,
    MemoryEvent,
    MemoryItemLog,
    MethodCall,
    PromptSourceSpan,
    RetrievalRecord,
    RunMetadata,
    TrialLog,
    TargetContaminationSetSpec,
    VerifierResult,
)
from memcontam.logging import provenance
from memcontam.memory.retrieval import retrieve_records
from memcontam.memory.stores import MemoryEntry


EXPOSURE_KEYS = {
    "condition",
    "status",
    "is_exposed",
    "answer_call_id",
    "target_entry_ids",
    "source_entry_ids",
    "exposed_source_ids",
    "exposure_mode",
    "reason",
    "target_set_id",
    "exposed_entry_ids",
    "exposed_injected_root_ids",
    "evidence_lineage_status",
}


def test_trial_log_minimal_shape() -> None:
    log = TrialLog(
        trial_id="t1",
        run_id="r1",
        task_name="game24",
        sample_id="s1",
        baseline="no_memory",
        arm="clean",
        backbone="gpt4o",
        input={"numbers": [1, 3, 4, 6]},
        gold_or_verifier_spec={"target": 24},
        prompt_messages=[{"role": "user", "content": "solve"}],
        raw_response="final: (6/(1-3/4))",
        verifier_result=VerifierResult(is_correct=True),
    )
    assert log.verifier_result is not None
    assert log.verifier_result.is_correct is True


def test_logging_v1_success_requires_answer_call_id() -> None:
    with pytest.raises(ValidationError, match="answer_call_id"):
        TrialLog(
            trial_id="t_strict_missing_answer",
            run_id="r1",
            task_name="game24",
            sample_id="s1",
            baseline="no_memory",
            arm="clean",
            backbone="gpt4o",
            input={"numbers": [1, 3, 4, 6]},
            gold_or_verifier_spec={"target": 24},
            prompt_messages=[{"role": "user", "content": "solve"}],
            raw_response="final: 24",
            parsed_answer="24",
            verifier_result=VerifierResult(is_correct=True),
            schema_version=LOGGING_V1,
            stage="replay",
            status="succeeded",
            run_metadata_id="run-meta-1",
            trial_seq=0,
            event_seq=1,
        )


def test_trial_log_metadata_and_metric_fields_are_explicit() -> None:
    log = TrialLog(
        trial_id="t2",
        run_id="r1",
        task_name="game24",
        sample_id="s2",
        baseline="retrieval_rag",
        arm="contaminated",
        backbone="gpt4o",
        input={"numbers": [1, 3, 4, 6]},
        gold_or_verifier_spec={"target": 24},
        prompt_messages=[{"role": "user", "content": "solve"}],
        memory_before=[{"entry_id": "m1"}],
        retrieved_memory=[{"entry_id": "m1"}],
        retrieved_scores=[0.75],
        filter_decision={"filter": "drop_known_contaminated", "dropped": True},
        raw_response="final: wrong",
        parsed_answer="wrong",
        verifier_result=VerifierResult(is_correct=False, reason="incorrect"),
        memory_write_event={"event_type": "reflection_append"},
        contamination_exposure=ContaminationExposure(
            condition="contaminated",
            status="not_evaluable",
            is_exposed=None,
            source_entry_ids=["m1"],
            exposure_mode="not_evaluable",
            reason="legacy proxy exposure has no final-call source spans",
        ),
        bad_memory_uptake_label="not_evaluable",
        repeated_failure_label="repeated_failure",
        recovery_after_filter_label="not_applicable",
        latency_ms=0,
        token_usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        metadata={
            "git_commit": "abc123",
            "config_hash": "hash123",
            "model_provider": "replay",
            "model_id": "gpt4o",
            "model_snapshot_or_served_name": "replay",
            "query_date": "2026-07-07T00:00:00Z",
            "seed_or_order": 1,
            "temperature": None,
            "top_p": None,
            "max_tokens": None,
            "prompt_version": "prompt_v0",
            "memory_policy_version": "memory_policy_v0",
            "contamination_set_version": "contamination_v0",
            "retry_policy_version": "retry_v0",
        },
    )

    assert log.verifier_result is not None
    assert log.verifier_result.is_correct is False
    assert log.metadata["git_commit"] == "abc123"
    assert log.metadata["config_hash"] == "hash123"
    assert log.metadata["model_provider"] == "replay"
    assert log.metadata["model_id"] == "gpt4o"
    assert log.metadata["model_snapshot_or_served_name"] == "replay"
    assert log.metadata["seed_or_order"] == 1
    assert log.metadata["prompt_version"] == "prompt_v0"
    assert log.latency_ms == 0
    assert log.token_usage == {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
    assert set(log.contamination_exposure.model_dump()) == EXPOSURE_KEYS


def test_bot_template_entry_and_write_event_contract_are_supported() -> None:
    template_entry = MemoryEntry(
        entry_id="bot-template-1",
        content="Distilled problem: 1 3 4 6 -> 24.",
        memory_type="thought_template",
        clean_or_contaminated="clean",
        source_trial_id="trial-source-1",
        metadata={
            "distilled_problem": "Use 6 / (1 - 3 / 4)",
            "template_description": "A reusable BoT template for the sample",
            "instantiation_source": "trial-source-1",
        },
    )
    log = TrialLog(
        trial_id="t_bot",
        run_id="r1",
        task_name="game24",
        sample_id="s_bot",
        baseline="bot_style",
        arm="contaminated",
        backbone="gpt4o",
        input={"numbers": [1, 3, 4, 6]},
        gold_or_verifier_spec={"target": 24},
        prompt_messages=[{"role": "user", "content": "solve"}],
        raw_response="final: wrong",
        verifier_result=VerifierResult(is_correct=False),
        memory_before=[template_entry.model_dump()],
        memory_write_event={
            "event_type": "bot_write",
            "baseline": "bot_style",
            "parent_trial_id": "trial-parent-1",
            "source_entry_ids": [template_entry.entry_id],
            "new_entry_id": "bot-template-2",
            "update_reason": "distilled from the successful solve",
        },
    )

    assert template_entry.memory_type == "thought_template"
    assert template_entry.source_trial_id == "trial-source-1"
    assert template_entry.metadata == {
        "distilled_problem": "Use 6 / (1 - 3 / 4)",
        "template_description": "A reusable BoT template for the sample",
        "instantiation_source": "trial-source-1",
    }
    assert log.memory_write_event == {
        "event_type": "bot_write",
        "baseline": "bot_style",
        "parent_trial_id": "trial-parent-1",
        "source_entry_ids": ["bot-template-1"],
        "new_entry_id": "bot-template-2",
        "update_reason": "distilled from the successful solve",
    }


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("bad_memory_uptake_label", "exposed"),
        ("repeated_failure_label", "unknown"),
        ("recovery_after_filter_label", "filtered"),
    ],
)
def test_trial_log_rejects_unknown_label_values(field: str, bad_value: str) -> None:
    kwargs = {
        "trial_id": "t_label",
        "run_id": "r1",
        "task_name": "game24",
        "sample_id": "s1",
        "baseline": "no_memory",
        "arm": "clean",
        "backbone": "gpt4o",
        "input": {"numbers": [1, 3, 4, 6]},
        "gold_or_verifier_spec": {"target": 24},
        "prompt_messages": [{"role": "user", "content": "solve"}],
        "raw_response": "final: 24",
        "verifier_result": VerifierResult(is_correct=True),
        field: bad_value,
    }

    with pytest.raises(ValidationError):
        TrialLog(**kwargs)


def test_trial_log_rejects_incomplete_contamination_exposure() -> None:
    with pytest.raises(ValidationError):
        TrialLog.model_validate(
            {
                "trial_id": "t_exposure",
                "run_id": "r1",
                "task_name": "game24",
                "sample_id": "s1",
                "baseline": "retrieval_rag",
                "arm": "contaminated",
                "backbone": "gpt4o",
                "input": {"numbers": [1, 3, 4, 6]},
                "gold_or_verifier_spec": {"target": 24},
                "prompt_messages": [{"role": "user", "content": "solve"}],
                "raw_response": "final: 24",
                "verifier_result": {"is_correct": True},
                "contamination_exposure": {"is_exposed": True, "source_entry_ids": ["m1"]},
            }
        )


def test_trial_log_accepts_legacy_null_memory_write_event_rows() -> None:
    log = TrialLog.model_validate(
        {
            "trial_id": "t_legacy",
            "run_id": "r1",
            "task_name": "game24",
            "sample_id": "s1",
            "baseline": "no_memory",
            "arm": "clean",
            "backbone": "gpt4o",
            "input": {"numbers": [1, 3, 4, 6]},
            "gold_or_verifier_spec": {"target": 24},
            "prompt_messages": [{"role": "user", "content": "solve"}],
            "raw_response": "final: 24",
            "verifier_result": {"is_correct": True},
            "memory_write_event": None,
        }
    )

    assert log.memory_write_event is None


@pytest.mark.parametrize("latency_ms", [-1, "7", 1.5])
def test_trial_log_rejects_invalid_latency(latency_ms: Any) -> None:
    with pytest.raises(ValidationError):
        TrialLog(
            trial_id="t3",
            run_id="r1",
            task_name="game24",
            sample_id="s3",
            baseline="no_memory",
            arm="clean",
            backbone="gpt4o",
            input={"numbers": [1, 3, 4, 6]},
            gold_or_verifier_spec={"target": 24},
            prompt_messages=[{"role": "user", "content": "solve"}],
            raw_response="final: 24",
            verifier_result=VerifierResult(is_correct=True),
            latency_ms=latency_ms,
        )


def test_replay_client_normalizes_token_usage_and_latency() -> None:
    response = ReplayClient().chat([{"role": "user", "content": "solve"}], model="gpt4o", config={})

    assert response.token_usage == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    assert response.latency_ms == 0


def test_retrieve_records_returns_deterministic_ordered_provenance_records() -> None:
    entries = [
        MemoryEntry(
            entry_id="m1",
            content="alpha beta gamma",
            memory_type="template",
            clean_or_contaminated="clean",
            source_trial_id="trial-1",
            metadata={"topic": "alpha"},
        ),
        MemoryEntry(
            entry_id="m2",
            content="alpha beta",
            memory_type="template",
            clean_or_contaminated="contaminated",
            source_trial_id="trial-2",
            metadata={"topic": "beta"},
        ),
        MemoryEntry(
            entry_id="m3",
            content="completely unrelated",
            memory_type="note",
            clean_or_contaminated="clean",
            source_trial_id=None,
            metadata={"topic": "gamma"},
        ),
    ]

    records = retrieve_records("alpha beta gamma", entries, k=2)

    assert [record["rank"] for record in records] == [1, 2]
    assert records[0]["entry_id"] == "m1"
    assert records[0]["content"] == "alpha beta gamma"
    assert records[0]["memory_type"] == "template"
    assert records[0]["clean_or_contaminated"] == "clean"
    assert records[0]["source_trial_id"] == "trial-1"
    assert records[0]["metadata"] == {"topic": "alpha"}
    assert isinstance(records[0]["memory_entry"], MemoryEntry)
    assert records[0]["memory_entry"].entry_id == "m1"
    assert records[0]["score"] >= records[1]["score"]
    assert records == retrieve_records("alpha beta gamma", entries, k=2)


def test_retrieve_records_handles_empty_memory_safely() -> None:
    assert retrieve_records("anything", [], k=3) == []


def test_trial_log_accepts_faithful_method_calls() -> None:
    rag_retrieval = RetrievalRecord(
        document_id="rag-doc-1",
        rank=1,
        score=0.91,
        text="Useful strategy for 24 game.",
        title_or_type="game24_strategy",
        clean_or_contaminated="clean",
        source="memory_catalog_v1",
        corpus_hash="sha256:abc123",
        embedding_model_id="sentence-transformers/all-MiniLM-L6-v2",
        embedding_revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        embedding_library_version="sentence-transformers-3.0.0",
    )
    bot_calls = [
        MethodCall(
            stage="bot_problem_distill",
            messages=[{"role": "user", "content": "distill"}],
            raw_response="Key info...",
            model="gpt4o",
            temperature=0.0,
            top_p=1.0,
            max_tokens=1024,
            latency_ms=100,
            token_usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            retry_count=0,
            error_type=None,
        ),
        MethodCall(
            stage="bot_instantiate_solve",
            messages=[{"role": "user", "content": "solve"}],
            raw_response="final: 6 / (1 - 3/4)",
            model="gpt4o",
            temperature=0.0,
            top_p=1.0,
            max_tokens=1024,
            latency_ms=200,
            token_usage={"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            retry_count=0,
            error_type=None,
        ),
        MethodCall(
            stage="bot_thought_distill",
            messages=[{"role": "user", "content": "distill thought"}],
            raw_response="High-level template...",
            model="gpt4o",
            temperature=0.0,
            top_p=1.0,
            max_tokens=1024,
            latency_ms=150,
            token_usage={"prompt_tokens": 15, "completion_tokens": 8, "total_tokens": 23},
            retry_count=0,
            error_type=None,
        ),
        MethodCall(
            stage="bot_novelty_decide",
            messages=[{"role": "user", "content": "decide"}],
            raw_response="True",
            model="gpt4o",
            temperature=0.0,
            top_p=1.0,
            max_tokens=256,
            latency_ms=50,
            token_usage={"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            retry_count=0,
            error_type=None,
        ),
    ]

    log = TrialLog(
        trial_id="t_faithful",
        run_id="r1",
        task_name="game24",
        sample_id="s_faithful",
        baseline="bot_style",
        arm="contaminated_filter",
        backbone="gpt4o",
        input={"numbers": [1, 3, 4, 6]},
        gold_or_verifier_spec={"target": 24},
        prompt_messages=[{"role": "user", "content": "solve"}],
        raw_response="final: 6 / (1 - 3/4)",
        verifier_result=VerifierResult(is_correct=True),
        method_calls=[
            MethodCall(
                stage="rag_generate",
                messages=[{"role": "user", "content": "generate with retrieved memory"}],
                raw_response="final: 6 / (1 - 3/4)",
                model="gpt4o",
                temperature=0.0,
                top_p=1.0,
                max_tokens=1024,
                latency_ms=120,
                token_usage={"prompt_tokens": 25, "completion_tokens": 10, "total_tokens": 35},
                retry_count=0,
                error_type=None,
                retrieved_records=[rag_retrieval],
            ),
            *bot_calls,
        ],
    )

    assert len(log.method_calls) == 5
    assert [call.stage for call in log.method_calls] == [
        "rag_generate",
        "bot_problem_distill",
        "bot_instantiate_solve",
        "bot_thought_distill",
        "bot_novelty_decide",
    ]
    assert log.method_calls[0].retrieved_records == [rag_retrieval]
    assert log.method_calls[0].retrieved_records[0].embedding_revision == (
        "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    )
    serialized = log.model_dump()
    assert len(serialized["method_calls"]) == 5
    assert serialized["method_calls"][0]["stage"] == "rag_generate"


def test_trial_log_legacy_method_calls_default() -> None:
    log = TrialLog.model_validate(
        {
            "trial_id": "t_legacy_methods",
            "run_id": "r1",
            "task_name": "game24",
            "sample_id": "s_legacy",
            "baseline": "retrieval_rag",
            "arm": "clean",
            "backbone": "gpt4o",
            "input": {"numbers": [1, 3, 4, 6]},
            "gold_or_verifier_spec": {"target": 24},
            "prompt_messages": [{"role": "user", "content": "solve"}],
            "raw_response": "final: 24",
            "verifier_result": {"is_correct": True},
        }
    )

    assert log.method_calls == []
    serialized = log.model_dump()
    assert serialized["method_calls"] == []


def test_method_call_rejects_missing_stage() -> None:
    with pytest.raises(ValidationError):
        MethodCall.model_validate(
            {
                "messages": [{"role": "user", "content": "solve"}],
                "raw_response": "final: 24",
                "model": "gpt4o",
            }
        )


@pytest.mark.parametrize(
    "baseline,memory_write_event",
    [
        (
            "full_history",
            {
                "type": "full_history_append",
                "status": "accepted",
                "new_entry_id": "fh:game24:s1:abc",
                "source_trial_id": "r1:game24:s1:full_history:clean:gpt4o",
                "parent_entry_ids": ["memory_clean_game24_full_history_001"],
                "source_entry_ids": [],
            },
        ),
        (
            "reflexion_style",
            {
                "type": "reflexion_append",
                "status": "accepted",
                "new_entry_id": "ref:game24:s1:abc",
                "source_trial_id": "r1:game24:s1:reflexion_style:clean:gpt4o",
                "parent_entry_ids": ["memory_clean_game24_reflexion_style_001"],
                "source_entry_ids": [],
            },
        ),
        (
            "dynamic_cheatsheet_optional",
            {
                "type": "dynamic_cheatsheet_update",
                "status": "accepted",
                "previous_entry_ids": ["memory_clean_game24_dynamic_cheatsheet_optional_001"],
                "new_entry_id": "dc:game24:abc",
                "source_trial_id": "r1:game24:s1:dynamic_cheatsheet_optional:clean:gpt4o",
                "parent_entry_ids": ["memory_clean_game24_dynamic_cheatsheet_optional_001"],
                "source_entry_ids": [],
                "source_contaminated_entry_ids": [],
            },
        ),
    ],
)
def test_native_memory_baseline_trial_log_conforms_to_schema(
    baseline: str, memory_write_event: dict[str, Any]
) -> None:
    log = TrialLog(
        trial_id="t_native",
        run_id="r1",
        task_name="game24",
        sample_id="s1",
        baseline=baseline,
        arm="clean",
        backbone="gpt4o",
        input={"numbers": [1, 3, 4, 6]},
        gold_or_verifier_spec={"target": 24},
        prompt_messages=[{"role": "user", "content": "solve"}],
        raw_response="final: 24",
        verifier_result=VerifierResult(is_correct=True),
        memory_write_event=memory_write_event,
        contamination_exposure=ContaminationExposure(),
    )

    expected_top_keys = {
        "trial_id",
        "run_id",
        "task_name",
        "sample_id",
        "baseline",
        "arm",
        "backbone",
        "input",
        "gold_or_verifier_spec",
        "prompt_messages",
        "memory_before",
        "retrieved_memory",
        "retrieved_scores",
        "filter_decision",
        "raw_response",
        "parsed_answer",
        "verifier_result",
        "metadata",
        "memory_write_event",
        "memory_after",
        "method_calls",
        "contamination_exposure",
        "bad_memory_uptake_label",
        "repeated_failure_label",
        "recovery_after_filter_label",
        "latency_ms",
        "token_usage",
        "cost_estimate",
        "retry_count",
        "error_type",
        "schema_version",
        "stage",
        "status",
        "run_metadata_id",
        "trial_seq",
        "event_seq",
        "answer_call_id",
        "failure_id",
        "evaluation_law_id",
        "target_set_id",
        "memory_update_mode",
        "trajectory_pair_id",
        "checkpoint_index",
        "pair_id",
        "checkpoint_ref",
    }
    assert set(log.model_dump().keys()) == expected_top_keys
    assert set(log.contamination_exposure.model_dump().keys()) == EXPOSURE_KEYS
    assert log.memory_write_event == memory_write_event


def _strict_success_payload() -> dict[str, Any]:
    prompt_messages = [{"role": "user", "content": "solve"}]
    return {
        "trial_id": "strict-trial-1",
        "run_id": "strict-run-1",
        "task_name": "game24",
        "sample_id": "sample-1",
        "baseline": "retrieval_rag",
        "arm": "contaminated",
        "backbone": "replay-model",
        "input": {"numbers": [1, 3, 4, 6]},
        "gold_or_verifier_spec": {"target": 24},
        "prompt_messages": prompt_messages,
        "raw_response": "final: 24",
        "parsed_answer": "24",
        "verifier_result": {"is_correct": True, "parsed_answer": "24"},
        "schema_version": LOGGING_V1,
        "stage": "replay",
        "status": "succeeded",
        "run_metadata_id": "run-meta-1",
        "trial_seq": 0,
        "event_seq": 4,
        "answer_call_id": "call-answer-1",
        "method_calls": [
            {
                "call_id": "call-answer-1",
                "stage": "rag_generate",
                "messages": prompt_messages,
                "raw_response": "final: 24",
                "model": "replay-model",
                "source_spans": [
                    {
                        "message_index": 0,
                        "start": 0,
                        "end": 5,
                        "rendered_hash": "sha256:prompt",
                        "entry_id": "memory-1",
                        "source_ids": ["memory-1"],
                        "parent_ids": [],
                        "lineage_id": "lineage-1",
                        "version": "v1",
                        "origin": "memory_catalog",
                        "clean_or_contaminated": "contaminated",
                    }
                ],
            }
        ],
        "contamination_exposure": {
            "condition": "contaminated",
            "status": "supported",
            "is_exposed": True,
            "answer_call_id": "call-answer-1",
            "target_entry_ids": ["memory-1"],
            "source_entry_ids": ["memory-1"],
            "exposed_source_ids": ["memory-1"],
            "exposure_mode": "final_prompt",
            "reason": "contaminated source span appears in the answer request",
        },
    }


@pytest.mark.parametrize("field", ["answer_call_id", "prompt_messages", "stage", "run_metadata_id"])
def test_logging_v1_success_rejects_missing_strict_links(field: str) -> None:
    payload = _strict_success_payload()
    del payload[field]

    with pytest.raises(ValidationError):
        TrialLog.model_validate(payload)


def test_logging_v1_success_requires_prompt_from_answer_call() -> None:
    payload = _strict_success_payload()
    payload["method_calls"].insert(
        0,
        {
            "call_id": "call-auxiliary-1",
            "stage": "reflect",
            "messages": [{"role": "user", "content": "auxiliary"}],
            "raw_response": "reflection",
            "model": "replay-model",
        },
    )
    payload["prompt_messages"] = [{"role": "user", "content": "flattened auxiliary and answer"}]

    with pytest.raises(ValidationError, match="prompt_messages"):
        TrialLog.model_validate(payload)


def test_logging_v1_failed_trial_allows_semantic_nulls_with_failure_link() -> None:
    payload = _strict_success_payload()
    payload.update(
        {
            "status": "failed",
            "raw_response": None,
            "parsed_answer": None,
            "verifier_result": None,
            "failure_id": "failure-1",
            "contamination_exposure": {
                "condition": "contaminated",
                "status": "not_evaluable",
                "is_exposed": None,
                "answer_call_id": "call-answer-1",
                "target_entry_ids": ["memory-1"],
                "source_entry_ids": ["memory-1"],
                "exposed_source_ids": [],
                "exposure_mode": "not_evaluable",
                "reason": "provider call failed",
            },
        }
    )
    payload["method_calls"][0]["raw_response"] = None

    log = TrialLog.model_validate(payload)

    assert log.status == "failed"
    assert log.raw_response is None
    assert log.parsed_answer is None
    assert log.verifier_result is None
    assert log.failure_id == "failure-1"


@pytest.mark.parametrize("field", ["raw_response", "parsed_answer", "verifier_result"])
def test_logging_v1_success_rejects_null_semantic_values(field: str) -> None:
    payload = _strict_success_payload()
    payload[field] = None

    with pytest.raises(ValidationError, match=field):
        TrialLog.model_validate(payload)


def test_legacy_trial_is_downgraded_and_proxy_exposure_is_not_supported() -> None:
    legacy_payload = {
        "trial_id": "legacy-trial-1",
        "run_id": "legacy-run-1",
        "task_name": "game24",
        "sample_id": "sample-1",
        "baseline": "retrieval_rag",
        "arm": "contaminated",
        "backbone": "replay-model",
        "input": {"numbers": [1, 3, 4, 6]},
        "gold_or_verifier_spec": {"target": 24},
        "prompt_messages": [{"role": "user", "content": "solve"}],
        "raw_response": "final: 24",
        "verifier_result": {"is_correct": True},
        "contamination_exposure": {
            "condition": "contaminated",
            "is_exposed": True,
            "source_entry_ids": ["memory-1"],
            "contamination_types": ["wrong_solution"],
            "memory_before_entry_ids": ["memory-1"],
            "retrieved_entry_ids": ["memory-1"],
            "exposure_mode": "retrieved_memory",
            "reason": "retrieval proxy",
        },
    }

    log = TrialLog.model_validate(legacy_payload)

    assert log.schema_version == "legacy"
    assert log.stage == "legacy"
    assert log.contamination_exposure.status == "not_evaluable"
    assert log.contamination_exposure.is_exposed is None


def test_legacy_trial_rejects_supported_exposure_without_answer_call_spans() -> None:
    payload = _strict_success_payload()
    for field in (
        "schema_version",
        "stage",
        "status",
        "run_metadata_id",
        "trial_seq",
        "event_seq",
        "answer_call_id",
    ):
        del payload[field]

    with pytest.raises(ValidationError, match="legacy"):
        TrialLog.model_validate(payload)


def test_typed_event_models_share_join_context() -> None:
    context = {
        "run_metadata_id": "run-meta-1",
        "run_id": "run-1",
        "trial_id": "trial-1",
        "trial_seq": 3,
        "event_seq": 9,
        "stage": "replay",
    }
    span = PromptSourceSpan(
        message_index=0,
        start=0,
        end=5,
        rendered_hash="sha256:prompt",
        entry_id="memory-1",
        source_ids=["memory-1"],
        parent_ids=[],
        lineage_id="lineage-1",
        version="v1",
        origin="memory_catalog",
        clean_or_contaminated="contaminated",
    )
    call = CallEvent(
        call_id="call-1",
        **context,
        messages=[{"role": "user", "content": "solve"}],
        model="replay-model",
        decoding_params={"temperature": 0.0},
        response_text="final: 24",
        token_usage={"total_tokens": 3},
        latency_ms=7,
        retry_count=0,
        source_spans=[span],
        created_at="2026-07-16T00:00:00Z",
    )
    failure = FailureEvent(
        failure_id="failure-1",
        **context,
        origin="provider_call",
        error_type="ConnectionError",
        failure_function="chat",
        failure_module="memcontam.clients",
        failure_line=12,
        retry_count=0,
        disposition="trial_failed",
        created_at="2026-07-16T00:00:01Z",
    )
    filter_event = FilterEvent(
        filter_id="filter-1",
        **context,
        arm="contaminated_filter",
        baseline="retrieval_rag",
        decisions=[{"entry_id": "memory-1", "action": "remove"}],
        kept_source_ids=[],
        removed_source_ids=["memory-1"],
        pre_source_ids=["memory-1"],
        post_source_ids=[],
        ground_truth_contaminated_ids=["memory-1"],
        action="outcome",
        final_answer_source_ids=[],
        verdict="not_exposed",
        created_at="2026-07-16T00:00:02Z",
    )
    memory_event = MemoryEvent(
        memory_id="memory-event-1",
        **context,
        event_type="write",
        operation="append",
        baseline="full_history",
        source_trial_id="trial-0",
        parent_entry_ids=["memory-0"],
        source_entry_ids=["memory-0"],
        contaminated_source_ids=[],
        before_entry_ids=["memory-0"],
        after_entry_ids=["memory-0", "memory-1"],
        before_snapshot_hash="sha256:before",
        after_snapshot_hash="sha256:after",
        new_entry_ids=["memory-1"],
        updated_entry_ids=[],
        removed_entry_ids=[],
        creation_origin="full_history_append",
        memory_version="v1",
        status="accepted",
        created_at="2026-07-16T00:00:03Z",
    )

    assert EventContext(**context).run_metadata_id == "run-meta-1"
    assert call.source_spans == [span]
    assert failure.origin == "provider_call"
    assert filter_event.removed_source_ids == ["memory-1"]
    assert memory_event.new_entry_ids == ["memory-1"]
    assert memory_event.lineage_edges == []
    with pytest.raises(ValidationError, match="lineage_edges"):
        CallEvent.model_validate({**call.model_dump(), "lineage_edges": []})
    with pytest.raises(ValidationError):
        FailureEvent.model_validate({**failure.model_dump(), "exception_message": "secret"})


def test_run_metadata_is_self_contained_and_versioned() -> None:
    metadata = RunMetadata(
        run_metadata_id="run-meta-1",
        run_id="run-1",
        git_commit="abc123",
        config_hash="sha256:config",
        provider="replay",
        model_snapshots={"replay-model": "fixture-v1"},
        query_date="2026-07-16",
        start_date="2026-07-16",
        seed=7,
        order="task-sample-baseline",
        decoding_defaults={"temperature": 0.0},
        sample_set_hash="sha256:samples",
        sample_order_hash="sha256:order",
        stage="replay",
        schema_version=LOGGING_V1,
        prompt_version="prompt-v1",
        memory_policy_version="memory-v1",
        contamination_catalog_version="catalog-v1",
        retry_policy_version="retry-v1",
    )

    assert metadata.schema_version == LOGGING_V1
    assert metadata.model_snapshots == {"replay-model": "fixture-v1"}


def test_logging_v2_run_metadata_requires_phase11_law_and_target_set() -> None:
    metadata = _logging_v2_run_metadata()

    assert metadata.schema_version == LOGGING_V2
    assert metadata.contract_level == "phase11"
    assert metadata.evaluation_law is not None
    assert metadata.evaluation_law.evaluation_law_id == "law-online-v1"
    assert metadata.target_contamination_set is not None
    assert metadata.target_contamination_set.target_set_id == "target-set-v1"

    with pytest.raises(ValidationError, match="evaluation_law"):
        _logging_v2_run_metadata(evaluation_law=None)
    with pytest.raises(ValidationError, match="target_contamination_set"):
        _logging_v2_run_metadata(target_contamination_set=None)


def test_logging_v2_trial_round_trips_with_exact_lineage_and_target_exposure() -> None:
    trial = TrialLog.model_validate(_logging_v2_trial_payload())

    assert trial.schema_version == LOGGING_V2
    assert trial.evaluation_law_id == "law-online-v1"
    assert trial.target_set_id == "target-set-v1"
    assert trial.pair_id == "pair-sample-1"
    assert trial.method_calls[0].source_spans[0].contamination_class == "derived"
    assert trial.contamination_exposure.target_set_id == "target-set-v1"
    assert trial.contamination_exposure.exposed_injected_root_ids == ["root-1"]
    assert "ancestor_ids" not in trial.model_dump()


@pytest.mark.parametrize(
    "field",
    ["evaluation_law_id", "target_set_id", "memory_update_mode", "trajectory_pair_id", "pair_id"],
)
def test_logging_v2_trial_rejects_missing_phase11_fields(field: str) -> None:
    payload = _logging_v2_trial_payload()
    del payload[field]

    with pytest.raises(ValidationError, match=field):
        TrialLog.model_validate(payload)


def test_logging_v2_trial_rejects_missing_required_lineage_fields() -> None:
    payload = _logging_v2_trial_payload()
    del payload["method_calls"][0]["source_spans"][0]["lineage_status"]

    with pytest.raises(ValidationError, match="lineage_status"):
        TrialLog.model_validate(payload)


@pytest.mark.parametrize("field", ["direct_parent_ids", "injected_root_ids"])
def test_exact_derived_span_and_item_require_direct_parents_and_injected_roots(field: str) -> None:
    span_payload = _logging_v2_span_payload()
    span_payload[field] = []
    with pytest.raises(ValidationError, match=field):
        PromptSourceSpan.model_validate(span_payload)

    item_payload = _logging_v2_memory_item_payload()
    item_payload[field] = []
    with pytest.raises(ValidationError, match=field):
        MemoryItemLog.model_validate(item_payload)


def test_signature_basis_cannot_claim_exact_lineage() -> None:
    with pytest.raises(ValidationError, match="signature"):
        PromptSourceSpan.model_validate(
            {**_logging_v2_span_payload(), "lineage_status": "exact", "lineage_basis": "signature"}
        )
    with pytest.raises(ValidationError, match="signature"):
        MemoryItemLog.model_validate(
            {
                **_logging_v2_memory_item_payload(),
                "lineage_status": "exact",
                "lineage_basis": "signature",
            }
        )


def test_logging_v2_accepts_an_exact_natural_target_set_row() -> None:
    payload = _logging_v2_trial_payload()
    natural_span = payload["method_calls"][0]["source_spans"][0]
    natural_span.update(
        {
            "entry_id": "natural-1",
            "source_ids": ["natural-1"],
            "parent_ids": [],
            "lineage_id": "natural-1",
            "clean_or_contaminated": "contaminated",
            "contamination_class": "natural",
            "injected_root_ids": [],
            "lineage_basis": "recorded_parent",
            "direct_parent_ids": [],
            "target_set_id": "natural-target-v1",
            "is_target_contamination": True,
        }
    )
    payload["target_set_id"] = "natural-target-v1"
    payload["memory_before"] = [{"entry_id": "natural-1", "metadata": {}}]
    payload["contamination_exposure"].update(
        {
            "target_entry_ids": ["natural-1"],
            "source_entry_ids": ["natural-1"],
            "exposed_source_ids": ["natural-1"],
            "target_set_id": "natural-target-v1",
            "exposed_entry_ids": ["natural-1"],
            "exposed_injected_root_ids": [],
        }
    )

    trial = TrialLog.model_validate(payload)

    assert trial.method_calls[0].source_spans[0].is_target_contamination is True


def test_legacy_parent_metadata_does_not_become_exact_direct_lineage() -> None:
    root = MemoryEntry(
        entry_id="root-1",
        content="Injected root.",
        memory_type="seed",
        clean_or_contaminated="contaminated",
        metadata={
            "contamination_class": "injected",
            "lineage_status": "exact",
            "lineage_basis": "seed",
            "direct_parent_ids": [],
            "injected_root_ids": ["root-1"],
        },
    )
    legacy_child = MemoryEntry(
        entry_id="legacy-child-1",
        content="Legacy descendant.",
        memory_type="note",
        clean_or_contaminated="contaminated",
        source_trial_id="trial-1",
        metadata={
            "parent_entry_ids": ["root-1"],
            "injected_root_ids": ["root-1"],
        },
    )

    lineage = provenance.canonical_lineage_for_entry(legacy_child, [root, legacy_child])

    assert lineage.contamination_class == "clean"
    assert lineage.lineage_status == "unavailable"
    assert lineage.direct_parent_ids == []


def test_lineage_edge_uses_explicit_direct_entry_edge_fields() -> None:
    edge = LineageEdge.model_validate(
        {
            "child_entry_id": "memory-derived-1",
            "parent_entry_id": "memory-root-1",
            "relation": "derived_from",
            "lineage_status": "exact",
            "lineage_basis": "recorded_parent",
            "injected_root_ids": ["root-1"],
        }
    )

    assert edge.child_entry_id == "memory-derived-1"
    assert edge.parent_entry_id == "memory-root-1"
    with pytest.raises(ValidationError, match="child_id"):
        LineageEdge.model_validate(
            {
                "child_id": "memory-derived-1",
                "parent_id": "memory-root-1",
                "lineage_status": "exact",
                "lineage_basis": "recorded_parent",
            }
        )
    with pytest.raises(ValidationError, match="signature"):
        LineageEdge.model_validate(
            {
                "child_entry_id": "memory-derived-1",
                "parent_entry_id": "memory-root-1",
                "relation": "derived_from",
                "lineage_status": "exact",
                "lineage_basis": "signature",
            }
        )


def test_checkpoint_ref_uses_phase11_checkpoint_identity() -> None:
    checkpoint = CheckpointRef.model_validate(_checkpoint_ref_payload())

    assert checkpoint.checkpoint_trial_index == 3
    assert checkpoint.checkpoint_memory_hash == "sha256:checkpoint"
    assert checkpoint.checkpoint_source_run_id == "run-v2"
    with pytest.raises(ValidationError, match="checkpoint_index"):
        CheckpointRef.model_validate(
            {
                "checkpoint_id": "checkpoint-3",
                "checkpoint_index": 3,
                "snapshot_hash": "sha256:checkpoint",
            }
        )


def test_logging_v2_frozen_and_online_checkpoint_rules() -> None:
    frozen = _logging_v2_trial_payload(
        evaluation_law_id="law-frozen-v1",
        memory_update_mode="disabled",
        checkpoint_index=3,
        checkpoint_ref=_checkpoint_ref_payload(),
    )
    assert TrialLog.model_validate(frozen).checkpoint_ref is not None

    missing_checkpoint = dict(frozen)
    missing_checkpoint["checkpoint_ref"] = None
    with pytest.raises(ValidationError, match="checkpoint_ref"):
        TrialLog.model_validate(missing_checkpoint)

    writing_baseline = dict(frozen)
    writing_baseline["baseline"] = "full_history"
    with pytest.raises(ValidationError, match="frozen"):
        TrialLog.model_validate(writing_baseline)

    online_with_checkpoint = _logging_v2_trial_payload(
        checkpoint_ref=_checkpoint_ref_payload(
            checkpoint_id="checkpoint-1", checkpoint_trial_index=1
        )
    )
    with pytest.raises(ValidationError, match="online"):
        TrialLog.model_validate(online_with_checkpoint)


def test_logging_v1_remains_phase10_and_does_not_accept_phase11_contract_level() -> None:
    v1_payload = _strict_success_payload()

    assert TrialLog.model_validate(v1_payload).schema_version == LOGGING_V1
    assert TrialLog.model_validate(v1_payload).evaluation_law_id is None
    with pytest.raises(ValidationError, match="contract_level"):
        RunMetadata.model_validate(
            {**_logging_v2_run_metadata().model_dump(), "schema_version": LOGGING_V1}
        )


def test_memory_item_log_normalizes_lineage_without_mutating_entry_id() -> None:
    entry = MemoryEntry(
        entry_id="entry-1",
        content="A persisted reflection.",
        memory_type="verbal_reflection",
        clean_or_contaminated="contaminated",
        source_trial_id="trial-0",
        metadata={
            "parent_entry_ids": ["entry-parent"],
            "source_entry_ids": ["entry-source"],
            "lineage_id": "lineage-1",
            "memory_version": "v2",
            "creation_origin": "reflexion_reflect",
        },
    )

    item = MemoryItemLog.from_memory_entry(entry)

    assert item.entry_id == "entry-1"
    assert entry.entry_id == "entry-1"
    assert item.content_hash == "8307c2a606d574c4d841efd53d451f2a69d294c1fc961eeb778f1dd4d62e8dde"
    assert item.parent_entry_ids == ["entry-parent"]
    assert item.source_entry_ids == ["entry-source"]
    assert item.lineage_id == "lineage-1"
    assert item.version == "v2"
    assert item.creation_origin == "reflexion_reflect"


def test_memory_item_log_canonicalizes_seed_natural_and_approximate_lineage() -> None:
    injected_seed = MemoryEntry(
        entry_id="injected-seed",
        content="Injected misleading rule.",
        memory_type="strategy",
        clean_or_contaminated="contaminated",
        metadata={
            "contamination_class": "injected",
            "lineage_status": "exact",
            "lineage_basis": "seed",
            "direct_parent_ids": [],
            "injected_root_ids": ["injected-seed"],
            "ancestor_ids": ["must-not-be-canonical"],
        },
    )
    natural_transcript = MemoryEntry(
        entry_id="natural-transcript",
        content="Incorrect full-history response.",
        memory_type="full_history_transcript",
        clean_or_contaminated="clean",
        source_trial_id="trial-1",
        metadata={"memory_error_status": "satisfied", "parent_entry_ids": ["clean-seed"]},
    )
    clean_reflection = MemoryEntry(
        entry_id="clean-reflection",
        content="Reflection: verify the arithmetic.",
        memory_type="verbal_reflection",
        clean_or_contaminated="clean",
        source_trial_id="trial-1",
        metadata={"parent_entry_ids": ["clean-seed"]},
    )
    signature_candidate = MemoryEntry(
        entry_id="signature-candidate",
        content="Possibly related note.",
        memory_type="note",
        clean_or_contaminated="contaminated",
        source_trial_id="trial-1",
        metadata={
            "contamination_class": "derived",
            "lineage_status": "approximate",
            "lineage_basis": "signature",
            "direct_parent_ids": ["injected-seed"],
            "injected_root_ids": ["injected-seed"],
        },
    )

    seed_item = MemoryItemLog.from_memory_entry(injected_seed)
    natural_item = MemoryItemLog.from_memory_entry(natural_transcript)
    reflection_item = MemoryItemLog.from_memory_entry(clean_reflection)
    signature_item = MemoryItemLog.from_memory_entry(signature_candidate)

    assert seed_item.contamination_class == "injected"
    assert seed_item.injected_root_ids == ["injected-seed"]
    assert "ancestor_ids" not in seed_item.metadata
    assert natural_item.contamination_class == "natural"
    assert natural_item.injected_root_ids == []
    assert reflection_item.contamination_class == "clean"
    assert signature_item.contamination_class == "clean"
    assert signature_item.lineage_status == "approximate"
    assert signature_item.lineage_basis == "signature"
    assert signature_item.injected_root_ids == []


def test_v2_exposure_uses_exact_answer_spans_against_fixed_target_set() -> None:
    target_set = TargetContaminationSetSpec(
        target_set_id="controlled_injected_derived_v1",
        definition_version="phase11-v1",
        included_classes=["injected", "derived"],
        require_exact_lineage=True,
    )
    exact_span = PromptSourceSpan.model_validate(
        {
            **_logging_v2_span_payload(),
            "parent_ids": ["root-1"],
            "direct_parent_ids": ["root-1"],
        }
    )
    injected_span = PromptSourceSpan.model_validate(
        {
            **_logging_v2_span_payload(),
            "entry_id": "root-1",
            "source_ids": ["root-1"],
            "parent_ids": [],
            "lineage_id": "root-1",
            "contamination_class": "injected",
            "injected_root_ids": ["root-1"],
            "lineage_basis": "seed",
            "direct_parent_ids": [],
        }
    )
    natural_span = PromptSourceSpan.model_validate(
        {
            **_logging_v2_span_payload(),
            "start": 6,
            "end": 12,
            "entry_id": "natural-1",
            "source_ids": ["natural-1"],
            "parent_ids": [],
            "lineage_id": "natural-1",
            "contamination_class": "natural",
            "injected_root_ids": [],
            "lineage_status": "exact",
            "lineage_basis": "recorded_parent",
            "direct_parent_ids": [],
            "is_target_contamination": False,
        }
    )
    exposure = provenance.compute_exposure_from_spans_v2(
        "call-answer-1",
        [exact_span, natural_span],
        "contaminated",
        [
            _memory_entry_payload("root-1", injected_span.model_dump()),
            _memory_entry_payload("memory-derived-1", exact_span.model_dump()),
            _memory_entry_payload("natural-1", natural_span.model_dump()),
        ],
        target_set,
    )

    assert exposure.status == "supported"
    assert exposure.is_exposed is True
    assert exposure.target_entry_ids == ["root-1", "memory-derived-1"]
    assert exposure.source_entry_ids == ["memory-derived-1", "natural-1"]
    assert exposure.exposed_entry_ids == ["memory-derived-1"]
    assert exposure.exposed_source_ids == ["memory-derived-1"]
    assert exposure.exposed_injected_root_ids == ["root-1"]
    assert exposure.evidence_lineage_status == "exact"


def test_v2_exposure_marks_approximate_only_target_evidence_not_evaluable() -> None:
    target_set = TargetContaminationSetSpec(
        target_set_id="controlled_injected_derived_v1",
        definition_version="phase11-v1",
        included_classes=["injected", "derived"],
        require_exact_lineage=True,
    )
    approximate_span = PromptSourceSpan.model_validate(
        {
            **_logging_v2_span_payload(),
            "entry_id": "approximate-1",
            "source_ids": ["approximate-1"],
            "parent_ids": [],
            "lineage_id": "approximate-1",
            "lineage_status": "approximate",
            "lineage_basis": "signature",
            "direct_parent_ids": [],
            "injected_root_ids": [],
            "is_target_contamination": False,
        }
    )

    exposure = provenance.compute_exposure_from_spans_v2(
        "call-answer-1",
        [approximate_span],
        "contaminated",
        [_memory_entry_payload("approximate-1", approximate_span.model_dump())],
        target_set,
    )

    assert exposure.status == "not_evaluable"
    assert exposure.is_exposed is None
    assert exposure.exposed_entry_ids == []
    assert exposure.exposed_source_ids == []
    assert exposure.exposed_injected_root_ids == []


def test_v2_exposure_marks_filtered_target_absence_with_distinct_reason() -> None:
    target_set = TargetContaminationSetSpec(
        target_set_id="controlled_injected_derived_v1",
        definition_version="phase11-v1",
        included_classes=["injected", "derived"],
        require_exact_lineage=True,
    )

    exposure = provenance.compute_exposure_from_spans_v2(
        "call-answer-1", [], "contaminated_filter", [], target_set
    )

    assert exposure.status == "supported"
    assert exposure.is_exposed is False
    assert exposure.reason == "target memory was filtered before final answer rendering"


def test_source_spans_preserve_exact_natural_target_context() -> None:
    target_set = TargetContaminationSetSpec(
        target_set_id="natural-target-v1",
        definition_version="phase11-v1",
        included_classes=["natural"],
        require_exact_lineage=True,
    )
    clean_seed = MemoryEntry(
        entry_id="clean-seed",
        content="A clean hint.",
        memory_type="strategy",
        clean_or_contaminated="clean",
        metadata={
            "contamination_class": "clean",
            "lineage_status": "exact",
            "lineage_basis": "seed",
            "direct_parent_ids": [],
            "injected_root_ids": [],
        },
    )
    natural = MemoryEntry(
        entry_id="natural-1",
        content="A failed response transcript.",
        memory_type="full_history_transcript",
        clean_or_contaminated="contaminated",
        source_trial_id="trial-1",
        metadata={
            "parent_entry_ids": ["clean-seed"],
            "direct_parent_ids": ["clean-seed"],
            "memory_error_status": "satisfied",
        },
    )
    entries = [clean_seed, natural]
    natural.metadata.update(provenance.phase11_lineage_metadata(natural, entries, target_set))
    _, spans = provenance.build_prompt_with_sources(
        [
            provenance.PromptSourcePart(natural.content, natural),
        ],
        entries=entries,
    )

    exposure = provenance.compute_exposure_from_spans_v2(
        "answer-call", spans, "contaminated", entries, target_set
    )
    clean_exposure = provenance.compute_exposure_from_spans_v2(
        "answer-call", spans, "clean", entries, target_set
    )

    assert spans[-1].contamination_class == "natural"
    assert spans[-1].lineage_status == "exact"
    assert spans[-1].is_target_contamination is True
    assert exposure.is_exposed is True
    assert exposure.exposed_entry_ids == ["natural-1"]
    assert clean_exposure.status == "not_applicable"
    assert clean_exposure.target_entry_ids == ["natural-1"]
    assert clean_exposure.source_entry_ids == ["natural-1"]


def _memory_entry_payload(entry_id: str, span: dict[str, Any]) -> dict[str, Any]:
    return {
        "entry_id": entry_id,
        "content": "memory",
        "memory_type": "memory_seed",
        "clean_or_contaminated": span["clean_or_contaminated"],
        "metadata": {
            field: span[field]
            for field in (
                "contamination_class",
                "injected_root_ids",
                "lineage_status",
                "lineage_basis",
                "direct_parent_ids",
                "target_set_id",
                "is_target_contamination",
            )
        },
    }


def _logging_v2_run_metadata(**overrides: Any) -> RunMetadata:
    payload = {
        "run_metadata_id": "run-meta-v2",
        "run_id": "run-v2",
        "git_commit": "abc123",
        "config_hash": "sha256:config",
        "provider": "replay",
        "model_snapshots": {"replay-model": "fixture-v1"},
        "query_date": "2026-07-17",
        "start_date": "2026-07-17",
        "seed": 7,
        "order": "task-sample-baseline",
        "decoding_defaults": {"temperature": 0.0},
        "sample_set_hash": "sha256:samples",
        "sample_order_hash": "sha256:order",
        "stage": "replay",
        "schema_version": LOGGING_V2,
        "contract_level": "phase11",
        "evaluation_law": {
            "evaluation_law_id": "law-online-v1",
            "regime": "online",
            "task_law_id": "locked-tasks-v1",
            "inference_law_id": "replay-fixtures-v1",
        },
        "target_contamination_set": {
            "target_set_id": "target-set-v1",
            "definition_version": "phase11-v1",
            "included_classes": ["injected", "derived"],
            "require_exact_lineage": True,
        },
        "checkpoint_policy": {"enabled": False},
        "prompt_version": "prompt-v1",
        "memory_policy_version": "memory-v1",
        "contamination_catalog_version": "catalog-v1",
        "retry_policy_version": "retry-v1",
    }
    payload.update(overrides)
    return RunMetadata.model_validate(payload)


def _logging_v2_span_payload() -> dict[str, Any]:
    return {
        "message_index": 0,
        "start": 0,
        "end": 5,
        "rendered_hash": "sha256:prompt",
        "entry_id": "memory-derived-1",
        "source_ids": ["memory-derived-1"],
        "parent_ids": ["memory-root-1"],
        "lineage_id": "lineage-1",
        "version": "v2",
        "origin": "reflexion_reflect",
        "clean_or_contaminated": "contaminated",
        "contamination_class": "derived",
        "injected_root_ids": ["root-1"],
        "lineage_status": "exact",
        "lineage_basis": "recorded_parent",
        "direct_parent_ids": ["memory-root-1"],
        "target_set_id": "target-set-v1",
        "is_target_contamination": True,
    }


def _logging_v2_memory_item_payload() -> dict[str, Any]:
    return {
        "entry_id": "memory-derived-1",
        "content_hash": "sha256:content",
        "memory_type": "verbal_reflection",
        "clean_or_contaminated": "contaminated",
        "source_trial_id": "trial-0",
        "parent_entry_ids": ["memory-root-1"],
        "source_entry_ids": ["memory-derived-1"],
        "lineage_id": "lineage-1",
        "version": "v2",
        "creation_origin": "reflexion_reflect",
        "metadata": {},
        "contamination_class": "derived",
        "injected_root_ids": ["root-1"],
        "lineage_status": "exact",
        "lineage_basis": "recorded_parent",
        "direct_parent_ids": ["memory-root-1"],
        "target_set_id": "target-set-v1",
        "is_target_contamination": True,
    }


def _checkpoint_ref_payload(**overrides: Any) -> dict[str, Any]:
    payload = {
        "checkpoint_id": "checkpoint-3",
        "checkpoint_trial_index": 3,
        "checkpoint_memory_hash": "sha256:checkpoint",
        "checkpoint_source_run_id": "run-v2",
    }
    payload.update(overrides)
    return payload


def _logging_v2_trial_payload(**overrides: Any) -> dict[str, Any]:
    prompt_messages = [{"role": "user", "content": "solve"}]
    payload = {
        "trial_id": "trial-v2-1",
        "run_id": "run-v2",
        "task_name": "game24",
        "sample_id": "sample-1",
        "baseline": "retrieval_rag",
        "arm": "contaminated",
        "backbone": "replay-model",
        "input": {"numbers": [1, 3, 4, 6]},
        "gold_or_verifier_spec": {"target": 24},
        "prompt_messages": prompt_messages,
        "memory_before": [
            {
                "entry_id": "memory-derived-1",
                "metadata": {
                    "contamination_class": "derived",
                    "lineage_status": "exact",
                },
            }
        ],
        "raw_response": "final: 24",
        "parsed_answer": "24",
        "verifier_result": {"is_correct": True, "parsed_answer": "24"},
        "schema_version": LOGGING_V2,
        "stage": "replay",
        "status": "succeeded",
        "run_metadata_id": "run-meta-v2",
        "trial_seq": 0,
        "event_seq": 4,
        "answer_call_id": "call-answer-1",
        "evaluation_law_id": "law-online-v1",
        "target_set_id": "target-set-v1",
        "memory_update_mode": "enabled",
        "trajectory_pair_id": "trajectory-sample-1",
        "checkpoint_index": None,
        "pair_id": "pair-sample-1",
        "checkpoint_ref": None,
        "method_calls": [
            {
                "call_id": "call-answer-1",
                "stage": "rag_generate",
                "messages": prompt_messages,
                "raw_response": "final: 24",
                "model": "replay-model",
                "source_spans": [_logging_v2_span_payload()],
            }
        ],
        "contamination_exposure": {
            "condition": "contaminated",
            "status": "supported",
            "is_exposed": True,
            "answer_call_id": "call-answer-1",
            "target_entry_ids": ["memory-derived-1"],
            "source_entry_ids": ["memory-derived-1"],
            "exposed_source_ids": ["memory-derived-1"],
            "exposure_mode": "final_prompt",
            "reason": "target contaminated source span appears in the answer request",
            "target_set_id": "target-set-v1",
            "exposed_entry_ids": ["memory-derived-1"],
            "exposed_injected_root_ids": ["root-1"],
            "evidence_lineage_status": "exact",
        },
    }
    payload.update(overrides)
    return payload
