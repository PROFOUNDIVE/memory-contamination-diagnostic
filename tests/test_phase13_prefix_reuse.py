from __future__ import annotations

from dataclasses import replace

import pytest

from memcontam.manifests.phase13 import NotExchangeable, PrefixDerivationArtifact
from memcontam.readiness.phase13_calibration_v2_runtime import execute_calibration_trajectory
from memcontam.readiness.phase13_calibration_v2_runtime_models import (
    CompletedTrajectory,
    TrajectoryRequest,
)
from memcontam.readiness.phase13_prefix_reuse import CHECK_IDS, derive_prefix_windows
from memcontam.readiness.phase13_support_authority import (
    SupportAuthorityError,
    authenticate_conformance,
)
from .test_phase13_calibration_v2_runtime import _fixture


def _completed(request: TrajectoryRequest) -> CompletedTrajectory:
    result = execute_calibration_trajectory(request)
    assert isinstance(result, CompletedTrajectory)
    return result


def test_all_ten_checks_are_named_and_required() -> None:
    _, _, request = _fixture()
    source = _completed(request)

    result = derive_prefix_windows(request, source)

    assert isinstance(result, PrefixDerivationArtifact)
    assert tuple(check.check_id for check in result.checks) == CHECK_IDS
    assert all(check.verdict == "pass" for check in result.checks)


def test_raw_order_and_state_are_recomputed_from_source_events() -> None:
    _, _, request = _fixture()
    source = _completed(request)
    events = list(source.events)
    events[0], events[1] = events[1], events[0]

    result = derive_prefix_windows(request, replace(source, events=tuple(events)))

    assert isinstance(result, NotExchangeable)
    assert {check.check_id for check in result.checks if check.verdict == "fail"} >= {
        "suffix_order", "exact_event_range", "source_raw_bytes"
    }


@pytest.mark.parametrize(
    "window_index",
    [0, 1, 2, 3, 4, 5, 6],
)
def test_unregistered_or_mutated_window_claims_are_ignored(window_index: int) -> None:
    _, _, request = _fixture()
    source = _completed(request)
    windows = list(request.verified.execution.analysis_windows)
    windows[window_index] = windows[window_index].model_copy(
        update={"analysis_window_id": "caller-window", "event_time_end": 9}
    )
    verified = replace(
        request.verified,
        execution=request.verified.execution.model_copy(update={"analysis_windows": tuple(windows)}),
    )

    result = derive_prefix_windows(replace(request, verified=verified), source)

    assert isinstance(result, PrefixDerivationArtifact)
    assert "caller-window" not in {row.analysis_window_id for row in result.rows}


def test_conformance_rejects_caller_stream_hash_drift_without_mocking_authority() -> None:
    _, _, request = _fixture()
    source = _completed(request)
    drifted = replace(request, source_ordered_stream_sha256="0" * 64)
    certificate = derive_prefix_windows(drifted, source)
    assert isinstance(certificate, PrefixDerivationArtifact)

    with pytest.raises(SupportAuthorityError, match="CONFORMANCE_SOURCE_AUTHORITY_HASH_MISMATCH"):
        authenticate_conformance(certificate, drifted, source)
