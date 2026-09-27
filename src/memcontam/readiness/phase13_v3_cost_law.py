from __future__ import annotations

from collections import Counter
from fractions import Fraction
from math import ceil

from memcontam.readiness.phase13_cost_policy_models import RateCard
from .phase13_v3_cost_models import (
    RATE_CARD, BaseCostInputsV3, CostTotals, ExactStageCost, RequestTokens, UnitProjection,
)


def decimal_string(value: Fraction) -> str:
    """Render a terminating rational exactly, independently of Decimal context."""
    integer, remainder = divmod(value.numerator, value.denominator)
    digits: list[str] = []
    while remainder:
        digit, remainder = divmod(remainder * 10, value.denominator)
        digits.append(str(digit))
    return str(integer) + ("." + "".join(digits) if digits else "")


def exact_request_cost(tokens: RequestTokens, rate: RateCard = RATE_CARD) -> tuple[Fraction, Fraction]:
    long_context = tokens.input_tokens > rate.long_context_threshold_tokens
    input_multiplier = Fraction(rate.long_context_input_multiplier) if long_context else Fraction(1)
    output_multiplier = Fraction(rate.long_context_output_multiplier) if long_context else Fraction(1)
    priced_input = tokens.input_tokens + tokens.cache_write_tokens * (Fraction(rate.cache_write_planning_premium) - 1)
    fx = rate.fx_planning_ceiling_krw_per_usd
    return (
        priced_input * Fraction(rate.input_usd_per_million) * input_multiplier * fx / 1_000_000,
        tokens.output_tokens * Fraction(rate.output_usd_per_million) * output_multiplier * fx / 1_000_000,
    )


def calculate(base: BaseCostInputsV3, order: tuple[str, ...] | None = None) -> tuple[CostTotals, tuple[UnitProjection, ...]]:
    envelopes = {stage.stage_id: stage for stage in base.policy.authority.registry.stages}
    units = {unit.unit_id: unit for unit in base.units}
    sums: dict[str, tuple[Fraction, Fraction]] = {}
    counts: Counter[str] = Counter()
    projections: list[UnitProjection] = []
    for unit_id in units if order is None else order:
        projected = 0
        for group in units[unit_id].stages:
            envelope = envelopes[group.stage_id]
            tokens = RequestTokens(input_tokens=envelope.maximum_input_tokens,
                                   output_tokens=envelope.maximum_output_tokens,
                                   cache_write_tokens=group.cache_write_tokens)
            input_cost, output_cost = exact_request_cost(tokens, base.policy.rate_card)
            before_in, before_out = sums.get(group.stage_id, (Fraction(0), Fraction(0)))
            after_in = before_in + input_cost * group.calls
            after_out = before_out + output_cost * group.calls
            sums[group.stage_id] = after_in, after_out
            counts[group.stage_id] += group.calls
            if order is not None:
                projected += ceil(after_in) - ceil(before_in) + ceil(after_out) - ceil(before_out)
        if order is not None:
            projections.append(UnitProjection(unit_id=unit_id, projected_krw=projected))
    stages = tuple(ExactStageCost(stage_id=stage, semantic_calls=counts[stage],
                                 input_exact_krw=decimal_string(sums[stage][0]),
                                 output_exact_krw=decimal_string(sums[stage][1]),
                                 input_krw_ceiling=ceil(sums[stage][0]),
                                 output_krw_ceiling=ceil(sums[stage][1])) for stage in sorted(sums))
    total = ceil(sum(stage.input_krw_ceiling + stage.output_krw_ceiling for stage in stages))
    margin = base.policy.budget.core_authorization_gate_krw - total
    return CostTotals(stage_costs=stages, semantic_calls=sum(counts.values()),
                      cmax_main_krw=total, gate_margin_krw=margin,
                      gate_result="PASS" if margin >= 0 else "FAIL"), tuple(projections)
