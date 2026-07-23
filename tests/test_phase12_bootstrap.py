from __future__ import annotations

import pytest


def test_bootstraps_seed_means_without_trial_resampling() -> None:
    from memcontam.evaluation.bootstrap import BootstrapConfig, bootstrap_seeds

    intervals = bootstrap_seeds(
        {"clean_minus_contam": {1: 0.5, 2: 0.5, 3: 0.0}},
        BootstrapConfig(replicates=200, random_seed=7),
    )

    interval = intervals.intervals["clean_minus_contam"]
    assert isinstance(interval.estimate, float)
    assert isinstance(interval.lower, float)
    assert isinstance(interval.upper, float)
    assert interval.estimate == pytest.approx(1 / 3)
    assert interval.lower <= interval.estimate <= interval.upper
    assert intervals.resampling_unit == "seed"


def test_rejects_trial_resampling_and_keeps_incomplete_panels_not_estimable() -> None:
    from memcontam.evaluation.bootstrap import BootstrapConfig, bootstrap_seeds
    from memcontam.evaluation.phase12_aggregate import AggregateError, NOT_ESTIMABLE

    with pytest.raises(AggregateError, match="TRIAL_RESAMPLING_FORBIDDEN"):
        bootstrap_seeds({"contrast": {1: 1.0}}, BootstrapConfig(unit="trial"))

    intervals = bootstrap_seeds(
        {"contrast": {1: 1.0, 2: None}}, BootstrapConfig(replicates=10, random_seed=1)
    )

    assert intervals.intervals["contrast"].status == NOT_ESTIMABLE
    assert intervals.intervals["contrast"].lower == NOT_ESTIMABLE
