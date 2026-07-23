from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from memcontam.logging.schema import PromptSourceSpan, TargetContaminationSetSpec
from memcontam.logging.schema_v3 import (
    ContextEvent,
    MemoryBranchTrialLog,
    RetrievalEvent,
    TrialLogV3,
)


class ObservableError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class TargetSetEvidence:
    target_set_id: str
    target_entry_ids: tuple[str, ...] = ()
    answer_call_id: str | None = None
    answer_call_spans: tuple[PromptSourceSpan, ...] = ()
    deterministic_full_history: bool = False
    definition: TargetContaminationSetSpec | None = None

    def __post_init__(self) -> None:
        if not self.target_set_id:
            raise ObservableError("TARGET_SET_ID_REQUIRED")
        if len(set(self.target_entry_ids)) != len(self.target_entry_ids) or any(
            not entry_id for entry_id in self.target_entry_ids
        ):
            raise ObservableError("INVALID_TARGET_ENTRY_IDS")


@dataclass(frozen=True)
class AttributionRule:
    rule_id: str
    version: str
    evaluate: Callable[[Mapping[str, object]], bool]

    def __post_init__(self) -> None:
        if not self.rule_id or not self.version:
            raise ObservableError("ATTRIBUTION_RULE_VERSION_REQUIRED")


@dataclass(frozen=True)
class RetrievalRecord:
    event_ids: tuple[str, ...]
    retrieved_entry_ids: tuple[str, ...]
    retrieved_target_entry_ids: tuple[str, ...]
    is_target_retrieved: bool


@dataclass(frozen=True)
class FinalContextInclusionRecord:
    status: Literal["supported", "unavailable"]
    context_event_id: str | None
    final_entry_ids: tuple[str, ...]
    included_target_entry_ids: tuple[str, ...]
    is_target_included: bool | None


@dataclass(frozen=True)
class TheoryExposureRecord:
    status: Literal["supported", "not_applicable", "unavailable"]
    target_set_id: str
    answer_call_id: str | None
    exposed_entry_ids: tuple[str, ...]
    is_exposed: bool | None
    exposed_non_exposed_contrast_supported: bool


@dataclass(frozen=True)
class AuxiliaryInclusionRecord:
    status: Literal["supported", "not_applicable", "unavailable"]
    included_entry_ids: tuple[str, ...]
    is_included: bool | None


@dataclass(frozen=True)
class OperationalUseRecord:
    status: Literal["supported", "not_attributed", "not_applicable"]
    rule_id: str | None
    rule_version: str | None
    is_used: bool


@dataclass(frozen=True)
class ObservableRecord:
    retrieval: RetrievalRecord
    final_context: FinalContextInclusionRecord
    exposure: TheoryExposureRecord
    auxiliary_inclusion: AuxiliaryInclusionRecord
    use: OperationalUseRecord


def compute_observables(
    trial: TrialLogV3,
    retrievals: Sequence[RetrievalEvent] | RetrievalEvent | None,
    context: ContextEvent | None,
    target_set: TargetSetEvidence | TargetContaminationSetSpec | Mapping[str, Any],
    attribution_rule: AttributionRule | None = None,
) -> ObservableRecord:
    """Compute separately reportable evidence without treating retrieval as exposure or use."""
    evidence = _target_set_evidence(target_set)
    target_ids = _target_ids(evidence)
    retrieval = _retrieval_record(trial, retrievals, target_ids)
    final_context = _final_context_record(trial, context, evidence, target_ids)
    arm = _arm(trial)

    if arm in {"correct", "irrelevant"}:
        if target_ids:
            raise ObservableError("AUXILIARY_THEORY_EXPOSURE_FORBIDDEN")
        auxiliary = _auxiliary_record(final_context)
        return ObservableRecord(
            retrieval=retrieval,
            final_context=final_context,
            exposure=TheoryExposureRecord(
                status="not_applicable",
                target_set_id=evidence.target_set_id,
                answer_call_id=evidence.answer_call_id,
                exposed_entry_ids=(),
                is_exposed=None,
                exposed_non_exposed_contrast_supported=False,
            ),
            auxiliary_inclusion=auxiliary,
            use=OperationalUseRecord("not_applicable", None, None, False),
        )

    exposure = _theory_exposure(arm, final_context, evidence, target_ids)
    use = _operational_use(trial, exposure, attribution_rule)
    return ObservableRecord(
        retrieval=retrieval,
        final_context=final_context,
        exposure=exposure,
        auxiliary_inclusion=AuxiliaryInclusionRecord("not_applicable", (), None),
        use=use,
    )


def _target_set_evidence(
    target_set: TargetSetEvidence | TargetContaminationSetSpec | Mapping[str, Any],
) -> TargetSetEvidence:
    if isinstance(target_set, TargetSetEvidence):
        return target_set
    if isinstance(target_set, TargetContaminationSetSpec):
        return TargetSetEvidence(target_set.target_set_id, definition=target_set)
    if not isinstance(target_set, Mapping):
        raise ObservableError("INVALID_TARGET_SET_EVIDENCE")
    definition_value = target_set.get("definition")
    definition = (
        definition_value
        if isinstance(definition_value, TargetContaminationSetSpec)
        else TargetContaminationSetSpec.model_validate(definition_value)
        if isinstance(definition_value, Mapping)
        else None
    )
    target_set_id = target_set.get("target_set_id") or (
        None if definition is None else definition.target_set_id
    )
    entry_ids = target_set.get("target_entry_ids", target_set.get("target_ids", ()))
    spans = target_set.get("answer_call_spans", target_set.get("source_spans", ()))
    if (
        not isinstance(target_set_id, str)
        or not isinstance(entry_ids, Sequence)
        or isinstance(entry_ids, str)
    ):
        raise ObservableError("INVALID_TARGET_SET_EVIDENCE")
    if not isinstance(spans, Sequence) or isinstance(spans, str):
        raise ObservableError("INVALID_TARGET_SET_EVIDENCE")
    return TargetSetEvidence(
        target_set_id=target_set_id,
        target_entry_ids=tuple(entry_id for entry_id in entry_ids if isinstance(entry_id, str)),
        answer_call_id=target_set.get("answer_call_id"),
        answer_call_spans=tuple(
            span if isinstance(span, PromptSourceSpan) else PromptSourceSpan.model_validate(span)
            for span in spans
        ),
        deterministic_full_history=bool(target_set.get("deterministic_full_history", False)),
        definition=definition,
    )


def _target_ids(evidence: TargetSetEvidence) -> tuple[str, ...]:
    target_ids = list(evidence.target_entry_ids)
    for span in evidence.answer_call_spans:
        if _is_target_span(span, evidence) and span.entry_id not in target_ids:
            target_ids.append(span.entry_id)
    return tuple(target_ids)


def _is_target_span(span: PromptSourceSpan, evidence: TargetSetEvidence) -> bool:
    if span.target_set_id not in {None, evidence.target_set_id}:
        raise ObservableError("TARGET_SET_ID_MISMATCH")
    if evidence.target_entry_ids:
        return span.entry_id in evidence.target_entry_ids
    if evidence.definition is not None:
        return span.contamination_class in evidence.definition.included_classes and (
            not evidence.definition.require_exact_lineage or span.lineage_status == "exact"
        )
    return span.is_target_contamination is True


def _retrieval_record(
    trial: TrialLogV3,
    retrievals: Sequence[RetrievalEvent] | RetrievalEvent | None,
    target_ids: tuple[str, ...],
) -> RetrievalRecord:
    events = (
        ()
        if retrievals is None
        else (retrievals,)
        if isinstance(retrievals, RetrievalEvent)
        else retrievals
    )
    by_id = {event.event_id: event for event in events}
    expected_ids = tuple(trial.retrieval_event_ids)
    if set(by_id) - set(expected_ids):
        raise ObservableError("UNLINKED_RETRIEVAL_EVENT")
    if any(event_id not in by_id for event_id in expected_ids):
        raise ObservableError("MISSING_RETRIEVAL_EVENT")
    retrieved_entry_ids = _unique(
        entry_id for event_id in expected_ids for entry_id in by_id[event_id].retrieved_entry_ids
    )
    retrieved_targets = tuple(
        entry_id for entry_id in retrieved_entry_ids if entry_id in target_ids
    )
    return RetrievalRecord(
        expected_ids, retrieved_entry_ids, retrieved_targets, bool(retrieved_targets)
    )


def _final_context_record(
    trial: TrialLogV3,
    context: ContextEvent | None,
    evidence: TargetSetEvidence,
    target_ids: tuple[str, ...],
) -> FinalContextInclusionRecord:
    expected_id = trial.context_event_id_or_none
    if context is not None and context.event_id != expected_id:
        raise ObservableError("UNLINKED_CONTEXT_EVENT")
    if expected_id is not None and context is None:
        raise ObservableError("MISSING_CONTEXT_EVENT")

    if context is None:
        if not evidence.answer_call_spans:
            return FinalContextInclusionRecord("unavailable", None, (), (), None)
        final_ids = _unique(span.entry_id for span in evidence.answer_call_spans)
        targets = tuple(entry_id for entry_id in final_ids if entry_id in target_ids)
        return FinalContextInclusionRecord("supported", None, final_ids, targets, bool(targets))

    final_ids = tuple(context.final_entry_ids)
    span_ids = _unique(span.entry_id for span in evidence.answer_call_spans)
    if span_ids and not set(span_ids).issubset(final_ids):
        raise ObservableError("FINAL_CONTEXT_SPAN_MISMATCH")
    targets = tuple(entry_id for entry_id in final_ids if entry_id in target_ids)
    return FinalContextInclusionRecord(
        "supported", context.event_id, final_ids, targets, bool(targets)
    )


def _arm(trial: TrialLogV3) -> str | None:
    return trial.execution_key.arm if isinstance(trial, MemoryBranchTrialLog) else None


def _auxiliary_record(final_context: FinalContextInclusionRecord) -> AuxiliaryInclusionRecord:
    if final_context.status == "unavailable":
        return AuxiliaryInclusionRecord("unavailable", (), None)
    return AuxiliaryInclusionRecord(
        "supported", final_context.final_entry_ids, bool(final_context.final_entry_ids)
    )


def _theory_exposure(
    arm: str | None,
    final_context: FinalContextInclusionRecord,
    evidence: TargetSetEvidence,
    target_ids: tuple[str, ...],
) -> TheoryExposureRecord:
    if arm not in {"contam", "filter"}:
        return TheoryExposureRecord(
            "not_applicable", evidence.target_set_id, evidence.answer_call_id, (), None, False
        )
    if final_context.status == "unavailable":
        if target_ids and not evidence.answer_call_spans:
            raise ObservableError("PRESENCE_ONLY_EXPOSURE_INFERENCE")
        return TheoryExposureRecord(
            "unavailable", evidence.target_set_id, evidence.answer_call_id, (), None, False
        )
    if not final_context.is_target_included:
        return TheoryExposureRecord(
            "supported", evidence.target_set_id, evidence.answer_call_id, (), False, True
        )
    if not evidence.answer_call_spans:
        raise ObservableError("PRESENCE_ONLY_EXPOSURE_INFERENCE")
    exposed_ids = tuple(
        span.entry_id
        for span in evidence.answer_call_spans
        if span.entry_id in final_context.included_target_entry_ids
        and _is_target_span(span, evidence)
    )
    if not exposed_ids:
        raise ObservableError("FINAL_CONTEXT_SPAN_MISMATCH")
    return TheoryExposureRecord(
        "supported",
        evidence.target_set_id,
        evidence.answer_call_id,
        exposed_ids,
        True,
        not evidence.deterministic_full_history,
    )


def _operational_use(
    trial: TrialLogV3,
    exposure: TheoryExposureRecord,
    attribution_rule: AttributionRule | None,
) -> OperationalUseRecord:
    if exposure.status == "not_applicable":
        if attribution_rule is not None:
            used = attribution_rule.evaluate(trial.operational_attribution_or_none or {})
            if not isinstance(used, bool):
                raise ObservableError("INVALID_ATTRIBUTION_RESULT")
            if used:
                raise ObservableError("U_GT_Z")
        return OperationalUseRecord("not_applicable", None, None, False)
    if attribution_rule is None:
        return OperationalUseRecord("not_attributed", None, None, False)
    evidence = trial.operational_attribution_or_none or {}
    used = attribution_rule.evaluate(evidence)
    if not isinstance(used, bool):
        raise ObservableError("INVALID_ATTRIBUTION_RESULT")
    if used and exposure.is_exposed is not True:
        raise ObservableError("U_GT_Z")
    return OperationalUseRecord(
        "supported", attribution_rule.rule_id, attribution_rule.version, used
    )


def _unique(entry_ids: Any) -> tuple[str, ...]:
    return tuple(dict.fromkeys(entry_id for entry_id in entry_ids if isinstance(entry_id, str)))


__all__ = [
    "AttributionRule",
    "AuxiliaryInclusionRecord",
    "FinalContextInclusionRecord",
    "ObservableError",
    "ObservableRecord",
    "OperationalUseRecord",
    "RetrievalRecord",
    "TargetSetEvidence",
    "TheoryExposureRecord",
    "compute_observables",
]
