from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Mapping, Sequence

from memcontam.experiment.phase12.contracts import BaselineConditionSpec
from memcontam.experiment.phase12.eligibility import JointEligibilityResult, compute_joint_eligibility
from memcontam.experiment.phase12.maturity import MaturityDecision, evaluate_maturity
from memcontam.experiment.phase12.timing import select_lower_quantile_checkpoint
from memcontam.memory.checkpoint_v3 import Phase12Checkpoint


MEMORY_BASELINES = ("fh_bounded", "rag_frozen", "bot_style", "reflexion_style")
_EXPECTED_FAMILIES = {
    "fh_bounded": "full_history",
    "rag_frozen": "rag",
    "bot_style": "bot",
    "reflexion_style": "reflexion",
}


class CheckpointSelectionError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class CheckpointRejection:
    baseline: str
    checkpoint_id: str
    checkpoint_index: int
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class CommonCheckpointSelection:
    seed: int
    horizon: int
    decisions: tuple[MaturityDecision, ...]
    rejections: tuple[CheckpointRejection, ...]
    joint_eligibility: JointEligibilityResult
    selected_trial_index: int | None
    selected_checkpoints: dict[str, Phase12Checkpoint]
    suffix_trial_indices: tuple[int, ...]
    block_reason: str | None

    @property
    def blocked(self) -> bool:
        return self.selected_trial_index is None


def select_common_checkpoint(
    *,
    seed: int,
    checkpoints_by_baseline: Mapping[str, Sequence[Phase12Checkpoint]],
    conditions: Mapping[str, BaselineConditionSpec],
    trial_indices: Sequence[int],
    suffix_horizon: int,
) -> CommonCheckpointSelection:
    _validate_inputs(seed, checkpoints_by_baseline, conditions, trial_indices, suffix_horizon)
    positions = {index: position for position, index in enumerate(trial_indices)}
    decisions: list[MaturityDecision] = []
    assessments: list[tuple[str, Phase12Checkpoint, MaturityDecision]] = []
    checkpoints_by_index: dict[tuple[str, int], Phase12Checkpoint] = {}

    for baseline in MEMORY_BASELINES:
        condition = conditions[baseline]
        for checkpoint in sorted(
            checkpoints_by_baseline[baseline], key=lambda item: _checkpoint_index(item)
        ):
            maturity = evaluate_maturity(condition, checkpoint, suffix_horizon)
            index = maturity.checkpoint_index
            if index not in positions:
                raise CheckpointSelectionError("CHECKPOINT_OUTSIDE_PREFIX")
            key = (baseline, index)
            if key in checkpoints_by_index:
                raise CheckpointSelectionError("DUPLICATE_BASELINE_CHECKPOINT")
            checkpoints_by_index[key] = checkpoint
            reasons = list(maturity.reason_codes)
            reasons.extend(_baseline_readiness_reasons(baseline, checkpoint))
            if len(trial_indices) - positions[index] - 1 < suffix_horizon:
                reasons.append("INSUFFICIENT_SUFFIX_HORIZON")
            decision = _decision_with_reasons(maturity, reasons)
            decisions.append(decision)
            assessments.append((baseline, checkpoint, decision))

    joint_eligibility = compute_joint_eligibility(decisions, suffix_horizon)
    selected_trial_index = select_lower_quantile_checkpoint(
        joint_eligibility.joint_eligible_indices, Decimal("0.5")
    )
    rejections = _rejections(assessments, joint_eligibility)
    if selected_trial_index is None:
        return CommonCheckpointSelection(
            seed=seed,
            horizon=suffix_horizon,
            decisions=tuple(decisions),
            rejections=rejections,
            joint_eligibility=joint_eligibility,
            selected_trial_index=None,
            selected_checkpoints={},
            suffix_trial_indices=(),
            block_reason="EMPTY_JOINT_ELIGIBILITY",
        )

    selected_checkpoints = {
        baseline: checkpoints_by_index[(baseline, selected_trial_index)]
        for baseline in MEMORY_BASELINES
    }
    suffix_start = positions[selected_trial_index] + 1
    return CommonCheckpointSelection(
        seed=seed,
        horizon=suffix_horizon,
        decisions=tuple(decisions),
        rejections=rejections,
        joint_eligibility=joint_eligibility,
        selected_trial_index=selected_trial_index,
        selected_checkpoints=selected_checkpoints,
        suffix_trial_indices=tuple(trial_indices[suffix_start : suffix_start + suffix_horizon]),
        block_reason=None,
    )


def _validate_inputs(
    seed: int,
    checkpoints_by_baseline: Mapping[str, Sequence[Phase12Checkpoint]],
    conditions: Mapping[str, BaselineConditionSpec],
    trial_indices: Sequence[int],
    suffix_horizon: int,
) -> None:
    if type(seed) is not int:
        raise CheckpointSelectionError("INVALID_CALIBRATION_SEED")
    if type(suffix_horizon) is not int or suffix_horizon < 1:
        raise CheckpointSelectionError("INVALID_SUFFIX_HORIZON")
    if (
        not trial_indices
        or any(type(index) is not int or index < 1 for index in trial_indices)
        or tuple(trial_indices) != tuple(sorted(trial_indices))
        or len(set(trial_indices)) != len(trial_indices)
    ):
        raise CheckpointSelectionError("INVALID_TRIAL_INDICES")
    if set(checkpoints_by_baseline) != set(MEMORY_BASELINES):
        raise CheckpointSelectionError("PRIMARY_BASELINE_PANEL_REQUIRED")
    if set(conditions) != set(MEMORY_BASELINES):
        raise CheckpointSelectionError("PRIMARY_CONDITION_PANEL_REQUIRED")
    for baseline, family in _EXPECTED_FAMILIES.items():
        if conditions[baseline].baseline_family != family:
            raise CheckpointSelectionError("BASELINE_CONDITION_MISMATCH")


def _checkpoint_index(checkpoint: Phase12Checkpoint) -> int:
    index = checkpoint.state.native_state.get("checkpoint_index")
    if type(index) is not int or index < 1:
        raise CheckpointSelectionError("CHECKPOINT_INDEX_MISSING")
    return index


def _baseline_readiness_reasons(baseline: str, checkpoint: Phase12Checkpoint) -> tuple[str, ...]:
    state = checkpoint.state.native_state
    if baseline == "fh_bounded":
        return (
            ()
            if state.get("first_eviction_trial_id") is None
            else ("FH_POST_INJECTION_VISIBILITY_UNAVAILABLE",)
        )
    if baseline == "rag_frozen":
        return () if state.get("branch") == "clean" else ("RAG_CLEAN_CORPUS_REQUIRED",)
    if baseline == "bot_style":
        templates = state.get("templates")
        competitors = state.get("clean_competitor_ids")
        template_ids = {
            item if isinstance(item, str) else item.get("id")
            for item in templates
            if isinstance(item, str) or isinstance(item, dict)
        } if isinstance(templates, (list, tuple)) else set()
        if (
            not isinstance(competitors, (list, tuple))
            or len(competitors) < 2
            or any(not isinstance(item, str) or item not in template_ids for item in competitors)
        ):
            return ("BOT_CLEAN_COMPETITORS_UNAVAILABLE",)
        return ()
    reflections = state.get("reflections")
    return (
        ()
        if isinstance(reflections, (list, tuple)) and reflections
        else ("REFLEXION_REFLECTIONS_UNAVAILABLE",)
    )


def _decision_with_reasons(
    maturity: MaturityDecision, reasons: Sequence[str]
) -> MaturityDecision:
    reason_codes = tuple(dict.fromkeys(reasons))
    if maturity.eligible and not reason_codes:
        return maturity
    return MaturityDecision(
        condition_id=maturity.condition_id,
        baseline_family=maturity.baseline_family,
        checkpoint_id=maturity.checkpoint_id,
        checkpoint_index=maturity.checkpoint_index,
        horizon=maturity.horizon,
        eligible=False,
        reason_codes=reason_codes,
    )


def _rejections(
    assessments: Sequence[tuple[str, Phase12Checkpoint, MaturityDecision]],
    joint_eligibility: JointEligibilityResult,
) -> tuple[CheckpointRejection, ...]:
    rejected: list[CheckpointRejection] = []
    jointly_eligible = set(joint_eligibility.joint_eligible_indices)
    for baseline, checkpoint, decision in assessments:
        reasons = decision.reason_codes
        if decision.eligible and decision.checkpoint_index not in jointly_eligible:
            reasons = ("NOT_JOINTLY_ELIGIBLE",)
        if reasons:
            rejected.append(
                CheckpointRejection(
                    baseline=baseline,
                    checkpoint_id=checkpoint.identity.checkpoint_id,
                    checkpoint_index=decision.checkpoint_index,
                    reason_codes=reasons,
                )
            )
    return tuple(rejected)


__all__ = [
    "CheckpointRejection",
    "CheckpointSelectionError",
    "CommonCheckpointSelection",
    "MEMORY_BASELINES",
    "select_common_checkpoint",
]
