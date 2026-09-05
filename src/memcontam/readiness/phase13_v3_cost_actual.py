from __future__ import annotations

from fractions import Fraction
from math import ceil
import re

from .phase13_v3_cost_law import decimal_string
from .phase13_v3_cost_models import (
    RATE_CARD, ActualCost, CostError, ProviderCostEvidence,
)


def reconcile_actual(evidence: ProviderCostEvidence) -> ActualCost:
    derived: Fraction | None = None
    if evidence.usage is not None:
        usage = evidence.usage
        long_context = usage.input_tokens > RATE_CARD.long_context_threshold_tokens
        input_multiplier = Fraction(RATE_CARD.long_context_input_multiplier) if long_context else Fraction(1)
        output_multiplier = Fraction(RATE_CARD.long_context_output_multiplier) if long_context else Fraction(1)
        derived = ((usage.input_tokens - usage.cached_input_tokens) * Fraction(RATE_CARD.input_usd_per_million)
                   + usage.cached_input_tokens * Fraction(RATE_CARD.cached_input_usd_per_million)) * input_multiplier
        derived += usage.output_tokens * Fraction(RATE_CARD.output_usd_per_million) * output_multiplier
        derived /= 1_000_000
    derived_usd = None if derived is None else decimal_string(derived)
    if evidence.monetary_cost is not None:
        if evidence.currency != "USD":
            raise CostError("MAIN_COST_CURRENCY_INVALID", evidence, derived_usd)
        if re.fullmatch(r"(0|[1-9][0-9]*)(\.[0-9]+)?", evidence.monetary_cost) is None:
            raise CostError("MAIN_COST_NUMERIC_INVALID", evidence, derived_usd)
        selected = Fraction(evidence.monetary_cost)
        if derived is not None and selected != derived:
            raise CostError("MAIN_COST_RECONCILIATION_REQUIRED", evidence, derived_usd)
        return ActualCost(evidence=evidence, derived_usd=derived_usd, source="AUTHORITATIVE_PROVIDER",
                          selected_usd=decimal_string(selected), realized_krw=ceil(selected * 1600))
    if derived is None:
        raise CostError("MAIN_TERMINAL_COST_UNKNOWN", evidence)
    return ActualCost(evidence=evidence, derived_usd=derived_usd, source="DERIVED_FROM_PROVIDER_USAGE",
                      selected_usd=decimal_string(derived), realized_krw=ceil(derived * 1600))
