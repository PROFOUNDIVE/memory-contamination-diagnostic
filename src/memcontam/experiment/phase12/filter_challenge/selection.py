from __future__ import annotations

from typing import Final, Literal, TypeAlias

from memcontam.experiment.phase12.filter_challenge.registry_common import StrictRegistry
from memcontam.experiment.phase12.filter_challenge.registry_search import Stratum


FILTER_V5_PILOT_B_NOT_ESTIMABLE: Final = "FILTER_V5_PILOT_B_NOT_ESTIMABLE"
BaselineFamily: TypeAlias = Literal[
    "full_history", "rag_frozen", "bot_style", "reflexion_style"
]


class CoverageEstimabilityInput(StrictRegistry):
    required_strata: tuple[Stratum, ...]
    strict_primary_strata: tuple[Stratum, ...]
    canonicalization_sensitivity_strata: tuple[Stratum, ...]
    stratum_weights: tuple[str, ...]


class CoverageEstimabilityDecision(StrictRegistry):
    estimable: bool
    reason_code: Literal["ESTIMABLE", "FILTER_V5_PILOT_B_NOT_ESTIMABLE"]
    missing_required_strata: tuple[Stratum, ...]
    retained_required_strata: tuple[Stratum, ...]
    retained_required_baselines: tuple[BaselineFamily, ...]
    sensitivity_substitution_applied: Literal[False]
    output_weights: tuple[str, ...]


def evaluate_coverage_estimability(
    value: CoverageEstimabilityInput,
) -> CoverageEstimabilityDecision:
    observed = set(value.strict_primary_strata)
    missing = tuple(stratum for stratum in value.required_strata if stratum not in observed)
    return CoverageEstimabilityDecision(
        estimable=not missing,
        reason_code=FILTER_V5_PILOT_B_NOT_ESTIMABLE if missing else "ESTIMABLE",
        missing_required_strata=missing,
        retained_required_strata=value.required_strata,
        retained_required_baselines=tuple(
            dict.fromkeys(stratum.baseline for stratum in value.required_strata)
        ),
        sensitivity_substitution_applied=False,
        output_weights=value.stratum_weights,
    )


__all__ = (
    "FILTER_V5_PILOT_B_NOT_ESTIMABLE",
    "CoverageEstimabilityDecision",
    "CoverageEstimabilityInput",
    "evaluate_coverage_estimability",
)
