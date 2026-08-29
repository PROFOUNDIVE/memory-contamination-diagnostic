from __future__ import annotations

from pathlib import Path

from memcontam.baselines.bot_read import RETRIEVAL_THRESHOLD
from memcontam.memory.embeddings import BgeM3EmbeddingProvider, normalized_dot_top_k
from memcontam.readiness.phase13_legacy_rag_models import IndexBundle
from memcontam.readiness.phase13_readiness0_f1c_contract import (
    ARMS,
    F1CReportError,
    LEGACY_TASKS,
    TASKS,
    QuerySpec,
    canonical_hash,
    file_hash,
    memory_candidates,
    queries,
    text_hash,
)
from memcontam.readiness.phase13_readiness0_f1c_models import (
    Arm,
    F1CReport,
    F1CRetrievalRow,
    F1CRuntimeProof,
    RetrievalBaseline,
    Task,
)
from memcontam.readiness.retrieval_smoke import deny_network, validate_bge_provider


def build_f1c_report(repository_root: Path, cache_root: Path) -> F1CReport:
    with deny_network() as guard:
        provider = BgeM3EmbeddingProvider(cache_folder=cache_root, local_files_only=True)
        runtime_values = validate_bge_provider(provider)
        query_specs = queries(repository_root)
        plans: tuple[tuple[RetrievalBaseline, tuple[Task, ...]], ...] = (
            ("rag_frozen", LEGACY_TASKS),
            ("bot_style", TASKS),
            ("dc_rs", TASKS),
        )
        rows = tuple(
            _retrieval_row(repository_root, provider, query_specs[task], task, baseline, arm)
            for baseline, tasks in plans
            for task in tasks
            for arm in ARMS
        )
    if guard.attempted:
        raise F1CReportError("READINESS0_F1C_NETWORK_ATTEMPT")
    runtime_payload = {
        **runtime_values,
        "vector_dimension": 1024,
        "normalize_embeddings": True,
        "network_attempts": 0,
    }
    runtime = F1CRuntimeProof(**runtime_payload, runtime_hash=canonical_hash(runtime_payload))
    report = F1CReport(
        schema_version="phase13_readiness0_f1c_report_v1",
        status="PASS",
        runtime=runtime,
        row_scope="ACTIVE_CURRENT_MAIN_RETRIEVAL_ARM_CELLS",
        row_count=52,
        rows=rows,
        report_hash="0" * 64,
    )
    payload = report.model_dump(mode="json", exclude={"report_hash"})
    return report.model_copy(update={"report_hash": canonical_hash(payload)})


def _retrieval_row(
    root: Path,
    provider: BgeM3EmbeddingProvider,
    query: QuerySpec,
    task: Task,
    baseline: RetrievalBaseline,
    arm: Arm,
) -> F1CRetrievalRow:
    if baseline == "rag_frozen":
        bundle = IndexBundle.model_validate_json(
            (root / f"data/phase13/rag/legacy/{task}/indices.json").read_bytes()
        )
        branch = bundle.branches[arm]
        scored = normalized_dot_top_k(
            provider.encode_query(query.text),
            [list(branch.vectors[document.id]) for document in branch.documents],
            [document.id for document in branch.documents],
            len(branch.documents),
        )
        state_hash, corpus_hash, index_hash = (
            None,
            branch.corpus_content_hash,
            branch.index_artifact_hash,
        )
        threshold, top_k = None, 3
    else:
        candidates = memory_candidates(root, task, arm, baseline)
        scored = normalized_dot_top_k(
            provider.encode_query(query.text),
            [provider.encode_document(text) for _entry_id, text in candidates],
            [entry_id for entry_id, _text in candidates],
            len(candidates),
        )
        threshold = RETRIEVAL_THRESHOLD if baseline == "bot_style" else None
        top_k = 1 if baseline == "bot_style" else min(3, len(candidates))
        state_hash = canonical_hash(
            {"task": task, "baseline": baseline, "arm": arm, "candidates": candidates}
        )
        corpus_hash = None
        index_hash = canonical_hash(
            {"candidate_ids": [entry_id for entry_id, _text in candidates], "scores": scored}
        )
    selected = tuple(
        entry_id
        for entry_id, score in scored[:top_k]
        if threshold is None or score >= threshold
    )
    candidate_ids = tuple(entry_id for entry_id, _score in scored)
    scores = tuple(score for _entry_id, score in scored)
    return F1CRetrievalRow(
        row_id=canonical_hash(
            {"task": task, "baseline": baseline, "arm": arm, "sample_id": query.sample_id}
        ),
        task=task,
        baseline=baseline,
        arm=arm,
        sample_id=query.sample_id,
        query_sha256=text_hash(query.text),
        query_source=query.source,
        query_source_sha256=file_hash(root / query.source),
        state_identity_sha256=state_hash,
        corpus_identity_sha256=corpus_hash,
        index_identity_sha256=index_hash,
        candidate_ids=candidate_ids,
        scores=scores,
        ranks=tuple(range(1, len(candidate_ids) + 1)),
        tie_policy="score_desc_id_lexical",
        selected_ids=selected,
        threshold=threshold,
        top_k=top_k,
        source_span_ids=selected,
        source_span_join_sha256=canonical_hash({"selected_ids": selected}),
    )


__all__ = ["build_f1c_report"]
