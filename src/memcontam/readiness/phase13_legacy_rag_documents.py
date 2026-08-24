from __future__ import annotations

import hashlib
from typing import Literal, assert_never

from .phase13_legacy_rag_bytes import JsonValue, canonical_json_bytes
from .phase13_legacy_rag_construction import Candidate
from .phase13_legacy_rag_generators import Game24Candidate, MebCandidate, WordSortingCandidate
from .phase13_legacy_rag_models import (
    BuildRegistry,
    CandidateAuditRecord,
    CorpusDocument,
    FeasibleTaskName,
)
from .phase13_legacy_rag_semantics import render_semantic_record, semantic_records


def clean_documents(
    task: FeasibleTaskName,
    candidates: tuple[Candidate, ...],
    registry: BuildRegistry,
) -> tuple[CorpusDocument, ...]:
    documents = [
        _document(task, row.stratum, row.record_id, row.record_id, render_semantic_record(row))
        for row in semantic_records(task)
    ]
    candidates_by_id = {candidate.digest: candidate for candidate in candidates}
    audits_by_id = {audit.candidate_id: audit for audit in registry.candidate_audits}
    selected = tuple(candidates_by_id[candidate_id] for candidate_id in registry.selected_worked_example_ids)
    for ordinal, candidate in enumerate(selected, start=1):
        audit = audits_by_id[candidate.digest]
        match candidate:
            case Game24Candidate(numbers=numbers):
                problem = canonical_json_bytes({"numbers": list(numbers), "target": 24})
                check = (
                    "The expression uses each supplied occurrence exactly once and evaluates "
                    "exactly to 24."
                )
            case MebCandidate(ordered_operands=operands, target_value=target):
                problem = f"{' ? '.join(map(str, operands))} = {target}".encode("ascii")
                check = (
                    "The response preserves the operand order, fills every operator slot with "
                    "an allowed operator, and the complete left-hand expression evaluates "
                    "exactly to the registered right-hand target."
                )
            case WordSortingCandidate(input_words=input_words):
                problem = canonical_json_bytes({"words": list(input_words)})
                check = (
                    "The output preserves the exact input-token multiset and equals the "
                    "registered lexical ordering."
                )
            case unreachable:
                assert_never(unreachable)
        rendered = (
            b"Problem: "
            + problem
            + b"\nCorrect response: "
            + candidate.response.encode("utf-8")
            + b"\nCheck: "
            + check.encode("utf-8")
            + b"\n"
        )
        documents.append(
            _document(
                task,
                "D",
                f"{task}-D{ordinal:02d}",
                candidate.digest,
                rendered,
                registry=registry,
                audit=audit,
            )
        )
    return tuple(documents)


def semantic_registry_hash(task: FeasibleTaskName) -> str:
    payload: list[JsonValue] = [
        {
            "record_id": row.record_id,
            "stratum": row.stratum,
            "title": row.title,
            "body": row.body,
        }
        for row in semantic_records(task)
    ]
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _document(
    task: FeasibleTaskName,
    stratum: Literal["A", "B", "C", "D"],
    suffix: str,
    source: str,
    content: bytes,
    *,
    registry: BuildRegistry | None = None,
    audit: CandidateAuditRecord | None = None,
) -> CorpusDocument:
    is_worked_example = stratum == "D"
    return CorpusDocument(
        schema_version="phase13_legacy_rag_clean_document_v1",
        document_id=f"legacy-rag::{task}::{suffix}",
        task_id=task,
        semantic_stratum=stratum,
        source_registry_id="phase13_postcutoff_addendum" if not is_worked_example else source,
        construction_pool_id="legacy_semantic_registry_v1" if not is_worked_example else "D_build",
        authoring_or_extraction_rule_id=(
            "legacy_rag_text_renderer_v1"
            if not is_worked_example
            else "legacy_rag_worked_example_renderer_v1"
        ),
        canonical_byte_contract_id="legacy_rag_canonical_bytes_v1",
        review_status="PASS",
        leakage_audit_status=audit.leakage_audit_status if audit else "PASS",
        content_hash=hashlib.sha256(content).hexdigest(),
        text=content.decode("utf-8"),
        build_instance_id=source if is_worked_example else None,
        build_generator_id=registry.generator.generator_id if registry else None,
        build_registry_id=registry.build_registry_id if registry else None,
        build_registry_sha256=(
            hashlib.sha256(canonical_json_bytes(registry.model_dump(mode="json"))).hexdigest()
            if registry
            else None
        ),
        generator_implementation_sha256=(
            registry.generator.implementation.sha256 if registry else None
        ),
        canonical_response_constructor_id=(
            "legacy_rag_canonical_response_constructor_v1" if is_worked_example else None
        ),
        semantic_validator_status=audit.semantic_validator_status if audit else None,
    )


__all__ = ["clean_documents", "semantic_registry_hash"]
