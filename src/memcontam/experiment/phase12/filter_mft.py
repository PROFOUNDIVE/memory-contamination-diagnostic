from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, TypedDict

from memcontam.memory.admission import AdmissionContext, evaluate_admission
from memcontam.memory.cards_v3 import MEMORY_CARD_V3, MemoryCardEnvelopeV3, canonical_content_hash


FILTER_MFT_SCHEMA_VERSION = "phase12_filter_mft_v4"
FILTER_POLICY_VERSION = "operational-evidence-filter-v4"


class FilterMftCase(TypedDict):
    actual_decision: Literal["active", "quarantine"]
    audit_label: str
    baseline: str
    case_id: str
    content_class: Literal["correct", "false"]
    expected_decision: Literal["active", "quarantine"]
    expected_state: Literal["active", "quarantine"]
    observed_state: Literal["active", "quarantine"]
    passed: bool
    policy_version: str
    reason: str
    reason_code: str
    route_class: Literal["contract-invalid direct write", "contract-valid ordinary write"]
    route_valid: bool
    treatment_arm: str


class FilterMftReport(TypedDict):
    cases: list[FilterMftCase]
    evidence_layer: str
    excluded_policy_fields: list[str]
    policy_version: str
    schema_version: str
    scientific_result: bool


_BASELINES = {
    "fh_bounded": ("full_history_transcript", "fh_appender", "full_history_generate", "history"),
    "rag_frozen": ("rag_document", "rag_corpus_loader", "rag_corpus_load", "corpus"),
    "bot_style": ("thought_template", "bot_buffer_manager", "bot_thought_distill", "buffer"),
    "reflexion_style": ("verbal_reflection", "reflexion_reflector", "reflexion_reflect", "reflections"),
}


def _envelope(
    baseline: str, route_valid: bool, content_class: Literal["correct", "false"]
) -> MemoryCardEnvelopeV3:
    semantic_kind, writer_id, writer_stage, native_component = _BASELINES[baseline]
    entry_id = f"{baseline}-{content_class}-{'valid' if route_valid else 'invalid'}"
    content = f"{content_class} calibration content"
    source_trial_id = None if baseline == "rag_frozen" else f"trial-{entry_id}"
    return MemoryCardEnvelopeV3(
        entry_id=entry_id,
        baseline=baseline,
        semantic_kind=semantic_kind,
        schema_version=MEMORY_CARD_V3,
        writer_id=writer_id if route_valid else "protocol_injector",
        writer_event_id=f"event-{entry_id}",
        writer_stage=writer_stage if route_valid else "protocol_inject",
        created_trial_id=source_trial_id,
        source_trial_ids=() if source_trial_id is None else (source_trial_id,),
        source_outcome=None,
        trial_support_ids=() if source_trial_id is None else (source_trial_id,),
        memory_support_ids=(),
        direct_parent_ids=(),
        version_predecessor_id=None,
        order_key=1,
        native_component=native_component,
        content=content,
        content_hash=canonical_content_hash(content),
    )


def build_filter_mft_report() -> FilterMftReport:
    cases: list[FilterMftCase] = []
    for baseline in _BASELINES:
        for route_valid in (False, True):
            for content_class in ("correct", "false"):
                envelope = _envelope(baseline, route_valid, content_class)
                context = AdmissionContext(
                    writer_event_ids=frozenset({envelope.writer_event_id}),
                    trial_record_ids=frozenset(envelope.trial_support_ids),
                    evidence_envelopes=(envelope,),
                )
                decision = evaluate_admission(envelope, context)
                expected_state: Literal["active", "quarantine"] = (
                    "active" if route_valid else "quarantine"
                )
                observed_state: Literal["active", "quarantine"] = (
                    "active" if decision.admitted else "quarantine"
                )
                cases.append(
                    {
                        "actual_decision": observed_state,
                        "audit_label": f"audit-{content_class}",
                        "baseline": baseline,
                        "case_id": envelope.entry_id,
                        "content_class": content_class,
                        "expected_decision": expected_state,
                        "expected_state": expected_state,
                        "observed_state": observed_state,
                        "passed": observed_state == expected_state,
                        "policy_version": FILTER_POLICY_VERSION,
                        "reason": decision.reason,
                        "reason_code": decision.reason,
                        "route_class": (
                            "contract-valid ordinary write"
                            if route_valid
                            else "contract-invalid direct write"
                        ),
                        "route_valid": route_valid,
                        "treatment_arm": "filter",
                    }
                )
    return {
        "cases": cases,
        "evidence_layer": "build_calibration_only",
        "excluded_policy_fields": ["audit_label", "candidate_role", "treatment_arm"],
        "policy_version": FILTER_POLICY_VERSION,
        "schema_version": FILTER_MFT_SCHEMA_VERSION,
        "scientific_result": False,
    }


def write_filter_mft_report(output: Path) -> FilterMftReport:
    report = build_filter_mft_report()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return report
