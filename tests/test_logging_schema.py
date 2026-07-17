from typing import Any

import pytest
from pydantic import ValidationError

from memcontam.clients.replay import ReplayClient
from memcontam.logging.schema import (
    CallEvent,
    ContaminationExposure,
    EventContext,
    FailureEvent,
    FilterEvent,
    LOGGING_V1,
    MemoryEvent,
    MemoryItemLog,
    MethodCall,
    PromptSourceSpan,
    RetrievalRecord,
    RunMetadata,
    TrialLog,
    VerifierResult,
)
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
