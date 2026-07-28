from __future__ import annotations

from memcontam.experiment.phase12.filter_challenge.registry_search import Stratum
from memcontam.experiment.phase12.filter_challenge.selection import (
    CoverageEstimabilityInput,
    evaluate_coverage_estimability,
)


def test_missing_required_task_is_not_estimable_without_sensitivity_substitution() -> None:
    # Given: a required task represented only by canonicalization sensitivity evidence.
    required = (
        Stratum(task_family="game24", baseline="full_history"),
        Stratum(task_family="word_sorting", baseline="reflexion_style"),
    )
    value = CoverageEstimabilityInput(
        required_strata=required,
        strict_primary_strata=required[:1],
        canonicalization_sensitivity_strata=required[1:],
        stratum_weights=("task-weight-1", "task-weight-2"),
    )

    # When: the production selection seam evaluates required coverage.
    result = evaluate_coverage_estimability(value)

    # Then: sensitivity cannot fill the task gap or change required strata and weights.
    assert not result.estimable
    assert result.reason_code == "FILTER_V5_PILOT_B_NOT_ESTIMABLE"
    assert result.missing_required_strata == required[1:]
    assert result.retained_required_strata == required
    assert result.retained_required_baselines == ("full_history", "reflexion_style")
    assert result.sensitivity_substitution_applied is False
    assert result.output_weights == value.stratum_weights


def test_missing_required_baseline_is_not_estimable_without_renormalization() -> None:
    # Given: one task remains represented while one of its required baselines is absent.
    required = (
        Stratum(task_family="game24", baseline="full_history"),
        Stratum(task_family="game24", baseline="rag_frozen"),
    )
    value = CoverageEstimabilityInput(
        required_strata=required,
        strict_primary_strata=required[:1],
        canonicalization_sensitivity_strata=(),
        stratum_weights=("baseline-weight-1", "baseline-weight-2"),
    )

    # When: the production selection seam evaluates required coverage.
    result = evaluate_coverage_estimability(value)

    # Then: it retains the missing baseline and original unnormalized weights.
    assert not result.estimable
    assert result.reason_code == "FILTER_V5_PILOT_B_NOT_ESTIMABLE"
    assert result.missing_required_strata == required[1:]
    assert result.retained_required_strata == required
    assert result.retained_required_baselines == ("full_history", "rag_frozen")
    assert result.output_weights == ("baseline-weight-1", "baseline-weight-2")
