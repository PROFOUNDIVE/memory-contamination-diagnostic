from __future__ import annotations

from dataclasses import replace

import pytest

from memcontam.manifests.phase13 import NotExchangeable, PrefixDerivationArtifact
from memcontam.readiness.phase13_calibration_v2_runtime import execute_calibration_trajectory
from memcontam.readiness.phase13_prefix_reuse import derive_prefix_windows
from test_phase13_calibration_v2_runtime import _fixture


def test_runtime_source_derives_registered_prefixes_without_new_dispatch() -> None:
    provider, _, request = _fixture()
    source = execute_calibration_trajectory(request)
    calls_after_source = len(provider.configs)

    result = derive_prefix_windows(request, source)

    assert isinstance(result, PrefixDerivationArtifact)
    assert len(provider.configs) == calls_after_source
    assert {row.analysis_window_id for row in result.rows} == {
        "accuracy-h2-sensitivity",
        "recurrence-h2-descriptive",
        "accuracy-h5-primary",
        "recurrence-h5-secondary",
        "persistence-h5-secondary",
        "propagation-h5-conditional",
        "collapse-h5-exploratory",
    }
    assert all(
        set(event.event_time for event in row.events) == set(range(row.window_length))
        and len(row.events) == row.window_length * 16
        for row in result.rows
    )


@pytest.mark.parametrize(
    ("field", "value", "check_id"),
    [
        ("model", "coordinated-model", "execution_contract_identity"),
        ("decoding_contract_id", "coordinated-decoding", "execution_contract_identity"),
        ("prompt_contract_id", "coordinated-prompt", "execution_contract_identity"),
        ("tool_contract_id", "coordinated-tool", "execution_contract_identity"),
        ("parser_contract_id", "coordinated-parser", "execution_contract_identity"),
        ("verifier_contract_id", "coordinated-verifier", "execution_contract_identity"),
        ("execution_owner_id", "coordinated-owner", "execution_contract_identity"),
        ("future_feedback_cutoff", 1, "future_feedback_cutoff"),
        ("suffix_id", "coordinated-suffix", "suffix_order"),
        ("status", "failed", "source_raw_bytes"),
    ],
)
def test_resigned_runtime_event_drift_is_not_exchangeable(
    field: str, value: str | int, check_id: str
) -> None:
    provider, _, request = _fixture()
    source = execute_calibration_trajectory(request)
    calls_after_source = len(provider.configs)
    events = list(source.events)
    events[0] = replace(events[0], **{field: value})
    mutated = replace(source, events=tuple(events))

    result = derive_prefix_windows(request, mutated)

    assert isinstance(result, NotExchangeable)
    assert result.derived_artifact is None
    assert check_id in {row.check_id for row in result.checks if row.verdict == "fail"}
    assert len(provider.configs) == calls_after_source


def test_coordinated_request_and_event_model_drift_cannot_resign_registry() -> None:
    provider, _, request = _fixture()
    source = execute_calibration_trajectory(request)
    events = tuple(replace(event, model="coordinated-model") for event in source.events)
    contexts = tuple(replace(context, model="coordinated-model") for context in request.contexts)

    result = derive_prefix_windows(replace(request, contexts=contexts), replace(source, events=events))

    assert isinstance(result, NotExchangeable)
    assert "execution_contract_identity" in {
        row.check_id for row in result.checks if row.verdict == "fail"
    }


def test_mutated_window_registry_cannot_define_bad_event_end() -> None:
    _, _, request = _fixture()
    source = execute_calibration_trajectory(request)
    windows = list(request.verified.execution.analysis_windows)
    windows[0] = windows[0].model_copy(update={"event_time_end": 4})
    execution = request.verified.execution.model_copy(update={"analysis_windows": tuple(windows)})
    verified = replace(request.verified, execution=execution)

    result = derive_prefix_windows(replace(request, verified=verified), source)

    assert isinstance(result, PrefixDerivationArtifact)
    row = next(item for item in result.rows if item.analysis_window_id == "accuracy-h2-sensitivity")
    assert row.event_time_range == (0, 1)
