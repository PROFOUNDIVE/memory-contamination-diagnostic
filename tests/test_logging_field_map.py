from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any, cast

from memcontam.baselines.contracts import BaselineExecutionOutcome


def test_outcome_to_logging_v2_map_preserves_failure_evidence_without_new_schema_fields() -> None:
    provenance = importlib.import_module("memcontam.logging.provenance")

    persist = cast(
        Callable[[BaselineExecutionOutcome], dict[str, Any]],
        getattr(provenance, "baseline_outcome_to_logging_v2", None),
    )
    assert callable(persist)

    failed = persist(
        BaselineExecutionOutcome(
            status="failed",
            error_type="ProviderCallFailure",
            failure_disposition="provider_call_failed",
            scientific_ineligibility_reason="provider_call_failed",
            metadata={"baseline_marker": "preserved"},
        )
    )

    assert failed["trial"] == {
        "status": "failed",
        "error_type": "ProviderCallFailure",
        "metadata": {
            "baseline_marker": "preserved",
            "failure_disposition": "provider_call_failed",
            "scientific_ineligibility_reason": "provider_call_failed",
        },
    }
    assert failed["failure"] == {
        "error_type": "ProviderCallFailure",
        "disposition": "provider_call_failed",
    }


def test_valid_incorrect_outcome_serializes_as_success_without_failure_evidence() -> None:
    provenance = importlib.import_module("memcontam.logging.provenance")
    persist = cast(
        Callable[[BaselineExecutionOutcome], dict[str, Any]],
        getattr(provenance, "baseline_outcome_to_logging_v2", None),
    )
    assert callable(persist)

    serialized = persist(
        BaselineExecutionOutcome(
            status="succeeded",
            final_response="wrong",
            parsed_answer="wrong",
            verifier_result=False,
        )
    )

    assert serialized["trial"]["status"] == "succeeded"
    assert serialized["trial"]["metadata"] == {}
    assert serialized["failure"] is None
