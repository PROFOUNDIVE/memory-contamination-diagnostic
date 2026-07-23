from __future__ import annotations

from decimal import Decimal, ROUND_CEILING
from typing import Literal, Sequence

from memcontam.experiment.phase12.maturity import EligibilityError


TIMING_QUANTILES: dict[Literal["early", "base", "late"], Decimal] = {
    "early": Decimal("0.25"),
    "base": Decimal("0.5"),
    "late": Decimal("0.75"),
}


def select_lower_quantile_checkpoint(eligible_indices: Sequence[int], quantile: Decimal) -> int | None:
    if not isinstance(quantile, Decimal) or not quantile.is_finite() or not Decimal(0) <= quantile <= Decimal(1):
        raise EligibilityError("INVALID_QUANTILE")
    if not eligible_indices:
        return None
    if any(type(index) is not int or index < 1 for index in eligible_indices):
        raise EligibilityError("INVALID_CHECKPOINT_INDEX")
    indices = sorted(eligible_indices)
    if len(indices) != len(set(indices)):
        raise EligibilityError("DUPLICATE_ELIGIBLE_CHECKPOINT")
    rank = int((quantile * len(indices)).to_integral_value(rounding=ROUND_CEILING))
    return indices[max(0, rank - 1)]


def select_timing_checkpoint(
    eligible_indices: Sequence[int], timing_quantile: Literal["early", "base", "late"]
) -> int | None:
    try:
        quantile = TIMING_QUANTILES[timing_quantile]
    except KeyError as error:
        raise EligibilityError("INVALID_TIMING_QUANTILE") from error
    return select_lower_quantile_checkpoint(eligible_indices, quantile)
