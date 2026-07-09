from typing import Any

import pytest
from pydantic import ValidationError

from memcontam.clients.replay import ReplayClient
from memcontam.logging.schema import ContaminationExposure, TrialLog, VerifierResult
from memcontam.memory.retrieval import retrieve_records
from memcontam.memory.stores import MemoryEntry


EXPOSURE_KEYS = {
    "condition",
    "is_exposed",
    "source_entry_ids",
    "contamination_types",
    "memory_before_entry_ids",
    "retrieved_entry_ids",
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
    assert log.verifier_result.is_correct is True


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
            is_exposed=True,
            source_entry_ids=["m1"],
            contamination_types=["wrong_solution"],
            memory_before_entry_ids=["m1"],
            retrieved_entry_ids=["m1"],
            exposure_mode="retrieved_memory",
            reason="retrieved contaminated memory source",
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
