from __future__ import annotations

from memcontam.baselines.retrieval_rag_phase12 import (
    RagFrozenStateV3,
    RagFrozenTrialContextV3,
)
from memcontam.experiment.phase12.filter_challenge.adapters.full_history import (
    FullHistoryPairRequest,
    FullHistoryProvisionalAdapter,
)
from memcontam.experiment.phase12.filter_challenge.adapters.rag_frozen import (
    RagFrozenPairRequest,
    RagFrozenProvisionalAdapter,
)
from memcontam.experiment.phase12.filter_challenge.registry_calibration import ScheduledCall
from memcontam.experiment.phase12.filter_challenge.rootless_local_contract import (
    RootlessContractError,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_native_capture_support import (
    MODEL,
    CaptureClient,
    CaptureEmbedder,
    CapturedMessages,
    candidate_fixture,
    captured,
    checkpoint_fixture,
    task_fixture,
)
from memcontam.memory.checkpoint_v3 import NativeEntry
from memcontam.rag.branch_index import BGE_M3_PRIMARY_IDENTITY, BranchIndex
from memcontam.rag.phase12_corpus import BranchCorpus, Document


def capture_storage_messages(call: ScheduledCall) -> CapturedMessages:
    if call.baseline == "full_history":
        return _capture_full_history(call)
    if call.baseline == "rag_frozen":
        return _capture_rag(call)
    raise RootlessContractError("ROOTLESS_COMPILATION_INVALID")


def _capture_full_history(call: ScheduledCall) -> CapturedMessages:
    task, answer = task_fixture(call)
    checkpoint = checkpoint_fixture(call)
    candidate = candidate_fixture(
        call, checkpoint.canonical_sha256, checkpoint.identity.checkpoint_id
    )
    control = CaptureClient(answer)
    challenge = CaptureClient(answer)
    FullHistoryProvisionalAdapter().execute(
        FullHistoryPairRequest(
            task=task,
            checkpoint=checkpoint,
            candidate=candidate,
            control_client=control,
            challenge_client=challenge,
            model=MODEL,
            context_config={
                "mode": "context_bounded_pair_atomic",
                "token_encoding": "cl100k_base",
                "context_window_tokens": 100_000,
                "max_output_tokens": 512,
                "fixed_prompt_overhead_tokens": 0,
                "safety_margin_tokens": 0,
            },
            verifier=lambda _answer, _task: True,
        )
    )
    return captured(
        control if call.side == "control" else challenge, "full_history_generate"
    )


def _capture_rag(call: ScheduledCall) -> CapturedMessages:
    task, answer = task_fixture(call)
    checkpoint = checkpoint_fixture(call)
    candidate = candidate_fixture(
        call, checkpoint.canonical_sha256, checkpoint.identity.checkpoint_id
    )
    entries = checkpoint.state.entries
    if not all(isinstance(entry, NativeEntry) for entry in entries):
        raise RootlessContractError("ROOTLESS_COMPILATION_INVALID")
    documents = tuple(
        Document.from_mapping({"id": entry.entry_id, "text": entry.content})
        for entry in entries
        if isinstance(entry, NativeEntry)
    )
    embedder = CaptureEmbedder(candidate.candidate_native_content)
    source_ids = tuple(document.document_id for document in documents)
    corpus = BranchCorpus("clean", documents, source_ids, "rootless-native-capture")
    index = BranchIndex(
        "clean",
        documents,
        {"production_identity": BGE_M3_PRIMARY_IDENTITY},
        {document.document_id: (0.0, 1.0) for document in documents},
        "rootless-native-capture",
        embedder,
    )
    control = CaptureClient(answer)
    challenge = CaptureClient(answer)
    RagFrozenProvisionalAdapter().execute(
        RagFrozenPairRequest(
            _trial(task, control, "control", source_ids),
            _trial(
                task,
                challenge,
                "challenge",
                (candidate.candidate_entry_id, *source_ids),
            ),
            RagFrozenStateV3("clean", corpus, index),
            candidate,
        )
    )
    return captured(control if call.side == "control" else challenge, "rag_generate")


def _trial(task, client: CaptureClient, side: str, source_ids: tuple[str, ...]):
    return RagFrozenTrialContextV3(
        task,
        client,
        MODEL,
        f"rootless-{side}",
        f"rootless-{side}",
        "rootless",
        "clean",
        "frozen",
        source_ids,
        verifier=lambda _answer, _task: True,
    )
