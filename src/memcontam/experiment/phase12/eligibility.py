from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from memcontam.experiment.phase12.maturity import EligibilityError, MaturityDecision
from memcontam.experiment.phase12.timing import select_lower_quantile_checkpoint


_PRIMARY_FAMILIES = {
    "fh": "full_history",
    "full_history": "full_history",
    "rag": "rag",
    "rag_frozen": "rag",
    "bot": "bot",
    "bot_style": "bot",
    "reflexion": "reflexion",
    "reflexion_style": "reflexion",
}
PRIMARY_FAMILIES = tuple(dict.fromkeys(_PRIMARY_FAMILIES.values()))


@dataclass(frozen=True)
class JointEligibilityResult:
    horizon: int
    baseline_eligible: dict[str, list[int]]
    optional_eligible: dict[str, list[int]]
    primary_intersection: tuple[int, ...]
    joint_eligible_indices: tuple[int, ...]
    ineligibility_reasons: dict[str, tuple[str, ...]]
    estimability_counts: dict[str, int]
    not_estimable: bool

    @property
    def J_s_H(self) -> tuple[int, ...]:
        return self.joint_eligible_indices


def compute_joint_eligibility(
    decisions: Sequence[MaturityDecision],
    horizon: int,
    *,
    required_families: Sequence[str] = PRIMARY_FAMILIES,
) -> JointEligibilityResult:
    _validate_horizon(horizon)
    required = tuple(
        dict.fromkeys(_PRIMARY_FAMILIES.get(family, family) for family in required_families)
    )
    eligible_by_family: dict[str, set[int]] = {}
    reasons_by_family: dict[str, list[str]] = {}
    seen_checkpoints: dict[tuple[str, int], str] = {}
    for decision in decisions:
        if _is_nomem_family(decision.baseline_family):
            continue
        if decision.horizon != horizon:
            raise EligibilityError("RELAXED_MATURITY_THRESHOLD")
        key = (decision.condition_id, decision.checkpoint_index)
        known_checkpoint = seen_checkpoints.setdefault(key, decision.checkpoint_id)
        if known_checkpoint != decision.checkpoint_id:
            raise EligibilityError("MIXED_CHECKPOINTS")
        eligible_by_family.setdefault(decision.baseline_family, set())
        reasons_by_family.setdefault(decision.baseline_family, [])
        if decision.eligible:
            eligible_by_family[decision.baseline_family].add(decision.checkpoint_index)
        else:
            reasons_by_family[decision.baseline_family].extend(decision.reason_codes)

    primary_sets: dict[str, set[int]] = {}
    for family, indices in eligible_by_family.items():
        primary_family = _PRIMARY_FAMILIES.get(family, family)
        if primary_family in required:
            primary_sets[primary_family] = indices
    joint = intersect_eligible_support(primary_sets, required_families=required_families)
    baseline_eligible = {
        family: sorted(indices)
        for family, indices in eligible_by_family.items()
        if _PRIMARY_FAMILIES.get(family, family) in required
    }
    optional_eligible = {
        family: sorted(indices)
        for family, indices in eligible_by_family.items()
        if _PRIMARY_FAMILIES.get(family, family) not in required
    }
    return JointEligibilityResult(
        horizon=horizon,
        baseline_eligible=baseline_eligible,
        optional_eligible=optional_eligible,
        primary_intersection=joint,
        joint_eligible_indices=joint,
        ineligibility_reasons={
            family: tuple(reasons) for family, reasons in reasons_by_family.items() if reasons
        },
        estimability_counts={
            "eligible_checkpoints": len(joint),
            "primary_baselines": len(set(required) & set(primary_sets)),
        },
        not_estimable=not joint,
    )


def intersect_eligible_support(
    eligible_by_family: Mapping[str, set[int]], *, required_families: Sequence[str]
) -> tuple[int, ...]:
    required = tuple(
        dict.fromkeys(_PRIMARY_FAMILIES.get(family, family) for family in required_families)
    )
    canonical = {
        _PRIMARY_FAMILIES.get(family, family): indices
        for family, indices in eligible_by_family.items()
    }
    if not required or any(family not in canonical for family in required):
        return ()
    return tuple(sorted(set.intersection(*(canonical[family] for family in required))))


def _validate_horizon(horizon: int) -> None:
    if type(horizon) is not int or horizon < 1:
        raise EligibilityError("INVALID_HORIZON")


def _is_nomem_family(family: str) -> bool:
    return "".join(character for character in family.lower() if character.isalnum()) in {
        "nomem",
        "nomemory",
    }


__all__ = [
    "EligibilityError",
    "JointEligibilityResult",
    "PRIMARY_FAMILIES",
    "compute_joint_eligibility",
    "intersect_eligible_support",
    "select_lower_quantile_checkpoint",
]
