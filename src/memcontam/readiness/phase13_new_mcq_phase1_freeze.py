from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Literal, assert_never

from pydantic import JsonValue

from memcontam.readiness.phase13_new_mcq_candidate import (
    DisplayedMcq,
    h1_selection,
    h2_selection,
    unicode_provenance,
)
from memcontam.readiness.phase13_new_mcq_phase1_models import (
    ApplicabilityId,
    CertificationId,
    CorrectId,
    OpaqueCertificationRecord,
    Phase1Freeze,
    Phase1SourcePaths,
    Phase1SourceIdentity,
    SplitCertification,
    TaskLocalRelevance,
    TaskPhase1Certification,
    TripletSemantics,
    render_packet_identities,
)
from memcontam.readiness.phase13_new_mcq_phase1_sources import (
    SPLIT_LAW_ID,
    CertificationItem,
    Phase1FreezeError,
    TaskSourcePool,
    load_source_pools,
    split_items,
)
from memcontam.readiness.phase13_core_bundle import CoreTask

_AUTHORITY_STACK: Final = (
    "phase13_theory_revised_v1",
    "phase13_baseline_revised_v5",
    "phase13_protocol_revised_v7",
    "phase13_experiment_revised_v7",
)


@dataclass(frozen=True, slots=True)
class _FrozenTaskPool:
    pool: TaskSourcePool
    build_items: tuple[CertificationItem, ...]
    calibration_items: tuple[CertificationItem, ...]
    split_identity: str
    pool_identity: str
    frozen_at: str


def build_phase1_freeze(paths: Phase1SourcePaths) -> Phase1Freeze:
    pools = load_source_pools(paths)
    split_pools = tuple((pool, *split_items(pool)) for pool in pools)
    pool_identity = _hash(
        {
            "stage": "candidate_pool_before_h1_h2",
            "tasks": [
                {
                    "task": pool.task_id,
                    "source_sha256": pool.facts.source_sha256,
                    "split_identity": split_identity,
                }
                for pool, _, _, split_identity in split_pools
            ],
        }
    )
    tasks: dict[CoreTask, TaskPhase1Certification] = {
        pool.task_id: _certify_task(_FrozenTaskPool(
            pool, build_items, calibration_items, split_identity, pool_identity, paths.frozen_at
        ))
        for pool, build_items, calibration_items, split_identity in split_pools
    }
    freeze_identity = _hash(
        {
            "schema_version": "phase13_new_mcq_phase1_freeze_v1",
            "pool_identity": pool_identity,
            "task_freezes": [tasks[task].candidate_freeze_identity for task in sorted(tasks)],
            "frozen_at": paths.frozen_at,
        }
    )
    return Phase1Freeze(
        schema_version="phase13_new_mcq_phase1_freeze_v1",
        authority_stack=_AUTHORITY_STACK,
        candidate_pool_frozen_before_certification=True,
        candidate_pool_identity=pool_identity,
        tasks=tasks,
        frozen_at=paths.frozen_at,
        freeze_identity=freeze_identity,
    )


def _certify_task(frozen: _FrozenTaskPool) -> TaskPhase1Certification:
    pool = frozen.pool
    build_items = frozen.build_items
    calibration_items = frozen.calibration_items
    split_identity = frozen.split_identity
    pool_identity = frozen.pool_identity
    frozen_at = frozen.frozen_at
    candidate_id = _select_candidate(build_items, calibration_items)
    correct_id: CorrectId
    applicability_id: ApplicabilityId
    counterexample_certification_id: CertificationId
    match candidate_id:
        case "MCQ-H1-LEXICAL-OVERLAP-v1":
            selector = h1_selection
            correct_id = "MCQ-H1-CORRECT-SUBSTANTIVE-CONTENT-v1"
            applicability_id = "MCQ-H1-UNIQUE-MAX-APP-v1"
            counterexample_certification_id = "MCQ-H1-GOLD-DISAGREEMENT-CERT-v1"
        case "MCQ-H2-DETAIL-LENGTH-v1":
            selector = h2_selection
            correct_id = "MCQ-H2-CORRECT-SUBSTANTIVE-CONTENT-v1"
            applicability_id = "MCQ-H2-UNIQUE-MAX-APP-v1"
            counterexample_certification_id = "MCQ-H2-GOLD-DISAGREEMENT-CERT-v1"
        case unreachable:
            assert_never(unreachable)
    records = tuple(
        _record(item, "build", selector(item.displayed)) for item in build_items
    ) + tuple(
        _record(item, "calibration", selector(item.displayed)) for item in calibration_items
    )
    source = Phase1SourceIdentity(
        upstream_dataset_identity=pool.facts.dataset_identity,
        upstream_revision=pool.facts.revision,
        selected_source_config=pool.facts.selected_config,
        source_sha256=pool.facts.source_sha256,
        exclusion_source_sha256=pool.facts.exclusion_source_sha256,
        excluded_canonical_rows=pool.facts.excluded_rows,
        eligible_rows=len(pool.items),
        split_law_id=SPLIT_LAW_ID,
        split_identity=split_identity,
    )
    triplet = TripletSemantics(
        false_candidate_id=candidate_id,
        correct_twin_id=correct_id,
        irrelevant_control_id="MCQ-I1-SINGLETON-OPTION-v1",
        correct_shares_false_applicability=True,
        irrelevant_is_valid=True,
        irrelevant_is_inapplicable_to_target=True,
    )
    relevance = TaskLocalRelevance(
        false_relation=applicability_id,
        correct_relation=applicability_id,
        irrelevant_relation="MCQ-I1-SINGLETON-APP-v1",
        relation_scope="task_local_frozen_displayed_items",
        realized_retrieval_excluded=True,
    )
    render_packets = render_packet_identities(pool.task_id, candidate_id, correct_id)
    certification_hash = _hash(
        {
            "task": pool.task_id,
            "candidate": candidate_id,
            "records": [record.model_dump(mode="json") for record in records],
        }
    )
    freeze_identity = _hash(
        {
            "task": pool.task_id,
            "pool_identity": pool_identity,
            "candidate": candidate_id,
            "certification_record_hash": certification_hash,
            "render_packets": [packet.packet_identity for packet in render_packets],
            "frozen_at": frozen_at,
        }
    )
    provenance = unicode_provenance()
    return TaskPhase1Certification(
        task_id=pool.task_id,
        status="MECHANICALLY_CERTIFIED_PENDING_BLINDED_PANEL",
        source=source,
        pool_identity=pool_identity,
        candidate_family_id="MCQ-SURFACE-CUE-SUFFICIENCY-v1",
        selected_candidate_id=candidate_id,
        mechanical_spec_id="MCQ-SURFACE-HEURISTICS-v1.0.0",
        normalizer_id="MCQ-NORM-NFKC-CASEFOLD-WS-v1",
        tokenizer_id="MCQ-TOK-UNICODE-LNM-RUN-v1",
        overlap_metric_id="MCQ-H1-JACCARD-DISTINCT-TOKENS-v1",
        length_detail_rule_id="MCQ-DETAIL-TOKEN-CODEPOINT-LEX-v1",
        applicability_implementation_id="MCQ-SURFACE-HEURISTICS-v1.0.0",
        counterexample_certification_id=counterexample_certification_id,
        certification_suite_id="MCQ-HEURISTIC-CERT-v1",
        unicode_data_manifest_hash=provenance.unicode_data_manifest_hash,
        implementation_source_hash=provenance.executable_source_sha256,
        test_vector_hash=provenance.conformance_vectors_sha256,
        build=_summary(records, "build", split_identity),
        calibration=_summary(records, "calibration", split_identity),
        records=records,
        triplet=triplet,
        relevance=relevance,
        render_packets=render_packets,
        challenge_suite_key=_hash(
            {"scope": "role_blind_route_blind", "task": pool.task_id, "pool": pool_identity}
        ),
        certification_record_hash=certification_hash,
        candidate_freeze_identity=freeze_identity,
        frozen_at=frozen_at,
    )


def _select_candidate(
    build_items: tuple[CertificationItem, ...], calibration_items: tuple[CertificationItem, ...]
) -> Literal["MCQ-H1-LEXICAL-OVERLAP-v1", "MCQ-H2-DETAIL-LENGTH-v1"]:
    halves = (build_items, calibration_items)
    if all(_split_passes(items, h1_selection) for items in halves):
        return "MCQ-H1-LEXICAL-OVERLAP-v1"
    if all(_split_passes(items, h2_selection) for items in halves):
        return "MCQ-H2-DETAIL-LENGTH-v1"
    raise Phase1FreezeError("PHASE1_CANDIDATE_FAMILY_NOT_READY")


def _split_passes(
    items: tuple[CertificationItem, ...], selector: Callable[[DisplayedMcq], int | None]
) -> bool:
    selections = tuple(selector(item.displayed) for item in items)
    return any(value is not None for value in selections) and any(
        value is not None and value != item.displayed.gold_index
        for item, value in zip(items, selections, strict=True)
    )


def _record(
    item: CertificationItem,
    split: Literal["build", "calibration"],
    selection: int | None,
) -> OpaqueCertificationRecord:
    return OpaqueCertificationRecord(
        opaque_item_id=item.opaque_item_id,
        canonical_local_sample_id=item.canonical_local_sample_id,
        permitted_content_hash=item.permitted_content_hash,
        displayed_option_permutation_id=item.displayed.display_identity,
        split=split,
        option_count=len(item.displayed.options),
        applicable_audit_only=selection is not None,
        counterexample_audit_only=(
            selection is not None and selection != item.displayed.gold_index
        ),
    )


def _summary(
    records: tuple[OpaqueCertificationRecord, ...],
    split: Literal["build", "calibration"],
    split_identity: str,
) -> SplitCertification:
    selected = tuple(record for record in records if record.split == split)
    return SplitCertification(
        split=split,
        split_identity=_hash({"split_registry": split_identity, "split": split}),
        rows=len(selected),
        applicable_rows=sum(record.applicable_audit_only for record in selected),
        counterexample_rows=sum(record.counterexample_audit_only for record in selected),
    )


def canonical_phase1_hash(value: JsonValue) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


_hash = canonical_phase1_hash


__all__ = ["Phase1FreezeError", "Phase1SourcePaths", "build_phase1_freeze", "canonical_phase1_hash"]
