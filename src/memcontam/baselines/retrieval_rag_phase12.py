from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Any, Literal

from memcontam.baselines.common import parse_final_answer
from memcontam.baselines.contracts import BaselineExecutionOutcome
from memcontam.baselines import retrieval_rag_adapter as legacy_rag
from memcontam.clients.base import LLMClient
from memcontam.clients.recording import MethodCallRecorder
from memcontam.logging.schema import RetrievalRecord
from memcontam.logging.schema_v3 import ContextEvent, RetrievalEvent
from memcontam.memory.stores import MemoryEntry, MemoryState
from memcontam.rag.branch_index import BranchIndex
from memcontam.rag.phase12_corpus import BranchCorpus, Document
from memcontam.tasks.base import TaskInstance
from memcontam.tasks.dispatch import canonical_task_json


__all__ = [
    "BaselineStepResultV3",
    "RagExecutionError",
    "RagFrozenPhase12Adapter",
    "RagFrozenStateV3",
    "RagFrozenTrialContextV3",
]


Branch = Literal["clean", "correct", "irrelevant", "contam", "filter"]
_INHERITED_BRANCHES = frozenset({"clean", "contam", "filter"})


class RagExecutionError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class RagFrozenTrialContextV3:
    task: TaskInstance
    client: LLMClient
    model: str
    run_id: str
    trial_id: str
    condition_id: str
    branch: Branch
    rag_mode: str
    included_document_ids: tuple[str, ...] | None = None
    claimed_exposure_document_ids: tuple[str, ...] | None = None
    verifier: Any = None

    def __post_init__(self) -> None:
        if not all((self.run_id, self.trial_id, self.condition_id)):
            raise RagExecutionError("INVALID_TRIAL_CONTEXT")


@dataclass(frozen=True)
class RagFrozenStateV3:
    branch: Branch
    corpus: BranchCorpus | None
    index: BranchIndex | None


@dataclass(frozen=True)
class BaselineStepResultV3:
    outcome: BaselineExecutionOutcome
    retrieval_event: RetrievalEvent
    context_event: ContextEvent
    theory_exposure_document_ids: tuple[str, ...] | None
    auxiliary_inclusion_document_ids: tuple[str, ...] | None


class RagFrozenPhase12Adapter:
    def execute(
        self, trial: RagFrozenTrialContextV3, state: RagFrozenStateV3
    ) -> BaselineStepResultV3:
        _validate_trial_and_state(trial, state)
        assert state.corpus is not None and state.index is not None

        query = canonical_task_json(trial.task)
        candidates = state.index.retrieve(query, 3)
        candidate_ids = tuple(candidate.document_id for candidate in candidates)
        included_ids = _included_ids(trial, candidate_ids)
        removed_ids = tuple(
            candidate_id for candidate_id in candidate_ids if candidate_id not in included_ids
        )
        theory_exposure, auxiliary_inclusion = _exposure_fields(trial, included_ids)

        documents = {document.document_id: document for document in state.index.documents}
        records = _records(candidates, documents, state.index)
        included_records = [record for record in records if record.document_id in included_ids]
        outcome = _run_answer(trial, included_records)
        outcome = replace(
            outcome,
            metadata={
                **outcome.metadata,
                "branch": state.branch,
                "corpus_serialization_id": state.corpus.serialization_id,
                "corpus_version": state.corpus.corpus_version,
                "index_artifact_hash": state.index.artifact_hash,
                "index_serialization_id": state.index.serialization_id,
                "index_version": state.index.index_version,
            },
        )
        retrieval_event = RetrievalEvent(
            record_type="retrieval_event",
            event_id=f"{trial.trial_id}:retrieval",
            run_id=trial.run_id,
            trial_id=trial.trial_id,
            event_seq=0,
            retrieval_id=f"{trial.trial_id}:retrieval",
            query_hash=hashlib.sha256(query.encode("utf-8")).hexdigest(),
            retrieved_entry_ids=list(candidate_ids),
            retrieved_scores=[candidate.score for candidate in candidates],
        )
        context_event = ContextEvent(
            record_type="context_event",
            event_id=f"{trial.trial_id}:context",
            run_id=trial.run_id,
            trial_id=trial.trial_id,
            event_seq=1,
            context_id=f"{trial.trial_id}:context",
            final_entry_ids=list(included_ids),
            removed_entry_ids=list(removed_ids),
        )
        return BaselineStepResultV3(
            outcome=outcome,
            retrieval_event=retrieval_event,
            context_event=context_event,
            theory_exposure_document_ids=theory_exposure,
            auxiliary_inclusion_document_ids=auxiliary_inclusion,
        )


def _validate_trial_and_state(trial: RagFrozenTrialContextV3, state: RagFrozenStateV3) -> None:
    if trial.rag_mode != "frozen":
        raise RagExecutionError("RAG_ONLINE_MODE_FORBIDDEN")
    if state.corpus is None:
        raise RagExecutionError("MISSING_BRANCH_CORPUS")
    if state.index is None:
        raise RagExecutionError("MISSING_BRANCH_INDEX")
    if {trial.branch, state.branch, state.corpus.branch, state.index.branch} != {trial.branch}:
        raise RagExecutionError("BRANCH_IDENTITY_MISMATCH")
    active_ids = tuple(document.document_id for document in state.corpus.active_documents)
    index_ids = tuple(document.document_id for document in state.index.documents)
    if index_ids != active_ids or set(state.index.vectors) != set(active_ids):
        raise RagExecutionError("BRANCH_INDEX_IDENTITY_MISMATCH")
    if trial.branch == "filter" and any(
        document_id not in state.corpus.active_document_ids for document_id in index_ids
    ):
        raise RagExecutionError("FILTER_QUARANTINE_EXPOSURE")


def _included_ids(
    trial: RagFrozenTrialContextV3, candidate_ids: tuple[str, ...]
) -> tuple[str, ...]:
    included = candidate_ids if trial.included_document_ids is None else trial.included_document_ids
    if len(set(included)) != len(included) or any(
        document_id not in candidate_ids for document_id in included
    ):
        raise RagExecutionError("INVALID_FINAL_CONTEXT")
    return tuple(document_id for document_id in candidate_ids if document_id in included)


def _exposure_fields(
    trial: RagFrozenTrialContextV3, included_ids: tuple[str, ...]
) -> tuple[tuple[str, ...] | None, tuple[str, ...] | None]:
    if trial.branch in _INHERITED_BRANCHES:
        if (
            trial.claimed_exposure_document_ids is not None
            and trial.claimed_exposure_document_ids != included_ids
        ):
            raise RagExecutionError("RAG_EXPOSURE_MISMATCH")
        return included_ids, None
    if trial.claimed_exposure_document_ids is not None:
        raise RagExecutionError("AUXILIARY_THEORY_EXPOSURE_FORBIDDEN")
    return None, included_ids


def _records(
    candidates: tuple[Any, ...], documents: dict[str, Document], index: BranchIndex
) -> list[RetrievalRecord]:
    identity = str(index.embedding_contract.get("production_identity", "unknown@unknown"))
    model_id, _, revision = identity.partition("@")
    return [
        RetrievalRecord(
            document_id=candidate.document_id,
            rank=candidate.rank,
            score=candidate.score,
            text=documents[candidate.document_id].text,
            title_or_type="rag_document",
            clean_or_contaminated="branch_document",
            source=index.serialization_id,
            corpus_hash=index.artifact_hash,
            embedding_model_id=model_id,
            embedding_revision=revision,
            embedding_library_version=index.index_version,
        )
        for candidate in candidates
    ]


def _run_answer(
    trial: RagFrozenTrialContextV3, records: list[RetrievalRecord]
) -> BaselineExecutionOutcome:
    memory = MemoryState(entries=[_memory_entry(record) for record in records])
    memory_before = tuple(entry.model_dump() for entry in memory.entries)
    recorder = MethodCallRecorder(trial.client)
    messages, source_spans = legacy_rag._messages(trial.task, records, memory.entries)
    try:
        response = recorder.chat(
            messages,
            model=trial.model,
            config={
                "sample_id": trial.task.sample_id,
                "method_stage": "rag_generate",
                "source_spans": source_spans,
            },
        )
    except Exception:
        return legacy_rag._failed_outcome(
            recorder,
            memory_before,
            memory,
            "ProviderCallFailure",
            "provider_call_failed",
            "provider_call_failed",
            records=records,
        )
    if response is None:
        return legacy_rag._failed_outcome(
            recorder,
            memory_before,
            memory,
            "ProviderCallFailure",
            "provider_call_failed",
            "provider_call_failed",
            records=records,
        )

    method_calls = recorder.get_records()
    if method_calls:
        method_calls[-1].retrieved_records = records
    answer_call_id = legacy_rag._answer_call_id(recorder)
    try:
        parsed_answer = parse_final_answer(response.content)
    except ValueError:
        parsed_answer = ""
    if not parsed_answer:
        return legacy_rag._failed_outcome(
            recorder,
            memory_before,
            memory,
            "BaselineOutputError",
            "rag_invalid_final_answer",
            "invalid_final_answer",
            final_response=response.content,
            answer_call_id=answer_call_id,
            records=records,
        )
    try:
        verifier_result = legacy_rag._verify(trial.verifier, parsed_answer, trial.task)
    except Exception:
        return legacy_rag._failed_outcome(
            recorder,
            memory_before,
            memory,
            "VerifierContractError",
            "verifier_contract_failed",
            "verifier_contract_failed",
            final_response=response.content,
            parsed_answer=parsed_answer,
            answer_call_id=answer_call_id,
            records=records,
        )
    return BaselineExecutionOutcome(
        status="succeeded",
        final_response=response.content,
        parsed_answer=parsed_answer,
        verifier_result=verifier_result,
        answer_call_id=answer_call_id,
        method_calls=tuple(method_calls),
        memory_before=memory_before,
        memory_after=memory_before,
        retrieved_memory=tuple(entry.model_dump() for entry in memory.entries),
        retrieved_scores=tuple(record.score for record in records),
        metadata={"rag_mode": "frozen", "top_k": 3},
    )


def _memory_entry(record: RetrievalRecord) -> MemoryEntry:
    return MemoryEntry(
        entry_id=record.document_id,
        content=record.text,
        memory_type="rag_document",
        clean_or_contaminated=record.clean_or_contaminated,
        metadata={"source": record.source},
    )
