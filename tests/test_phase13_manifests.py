from __future__ import annotations

import pytest
from pydantic import ValidationError

from memcontam.manifests.phase13 import AnalysisWindowBinding, SourceEvent


def _event() -> dict[str, object]:
    return {
        "event_time": 0,
        "absolute_trial_index": 2,
        "baseline": "fh_bounded",
        "arm": "clean",
        "source_checkpoint_id": "checkpoint-1",
        "branch_checkpoint_id": "branch-1",
        "suffix_id": "sample-2",
        "task": "game24",
        "model": "gpt-4o-2024-11-20",
        "decoding_contract_id": "phase13-decoding-zero-v1",
        "prompt_contract_id": "baseline-fidelity-v2-prompts",
        "tool_contract_id": "text-only-equal-availability-v1",
        "parser_contract_id": "phase13-task-parsers-v1",
        "verifier_contract_id": "phase13-task-verifiers-v1",
        "native_semantics_id": "phase13-native-capacity-v1",
        "session_id": "session-10000",
        "randomness_contract_id": "provider-managed-no-client-seed-v1",
        "future_feedback_cutoff": 0,
        "intervention_id": None,
        "execution_owner_id": "phase13-h10-execution-owner-v1",
        "status": "succeeded",
        "state_before_sha256": "1" * 64,
        "state_after_sha256": "2" * 64,
    }


@pytest.mark.parametrize(
    "field",
    [
        "decoding_contract_id", "prompt_contract_id", "tool_contract_id",
        "parser_contract_id", "verifier_contract_id", "native_semantics_id",
        "session_id", "randomness_contract_id", "future_feedback_cutoff",
        "intervention_id", "execution_owner_id", "status",
    ],
)
def test_source_event_requires_every_provenance_identity(field: str) -> None:
    payload = _event()
    payload.pop(field)

    with pytest.raises(ValidationError):
        SourceEvent.model_validate(payload)


@pytest.mark.parametrize(
    ("length", "end"),
    [(2, 4), (5, 1)],
)
def test_window_schema_rejects_mismatched_exact_range(length: int, end: int) -> None:
    with pytest.raises(ValidationError):
        AnalysisWindowBinding.model_validate(
            {
                "analysis_window_id": "bad-window",
                "window_length": length,
                "event_time_start": 0,
                "event_time_end": end,
                "outcome_family": "verified_accuracy",
                "evidence_status": "descriptive",
                "multiplicity_status": "estimation_only",
            }
        )
