from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
from pydantic import ValidationError

from memcontam.baselines.retrieval_rag_phase12 import RagFrozenStateV3
from memcontam.clients.base import LLMResponse
from memcontam.experiment import phase13_ordinary_runtime as ordinary_runtime
from memcontam.logging.schema import MethodCall, PromptSourceSpan
from memcontam.baselines.dynamic_cheatsheet_phase12 import DcRsStateV3
from memcontam.experiment import phase13_dc_rs_runtime
from memcontam.experiment.phase12.live_branch import LiveArmBranch
from memcontam.experiment.phase13_ordinary_runtime import ProspectiveOrdinaryRun, execute_prospective_ordinary
from memcontam.memory.cards_v3 import MEMORY_CARD_V3, MemoryCardEnvelopeV3, canonical_content_hash
from memcontam.memory.checkpoint_v3 import deserialize_checkpoint, serialize_checkpoint
from memcontam.memory.checkpoint_v3 import NativeState
from memcontam.logging.schema_v3 import RetrievalEvent
from memcontam.rag.branch_index import build_branch_indices
from memcontam.rag.phase12_corpus import CleanCorpus, build_branch_corpora
from memcontam.readiness.phase13_production_runtime_evidence import (
    _RuntimeEntryMetadata,
    _RuntimeMemoryEntry,
    _context,
    _lineage,
    _target_spans,
)
from memcontam.readiness.phase13_production_runtime_models import ProductionRuntimeJoinError
from memcontam.readiness.phase13_main_live_evidence import PrefixCheckpointState
from memcontam.readiness.phase13_main_live_evidence import _reflexion_stage_sequences_valid
from memcontam.tasks.base import TaskInstance


def _span(entry_id: str) -> PromptSourceSpan:
    return PromptSourceSpan(
        message_index=0,
        start=0,
        end=1,
        rendered_hash="a" * 64,
        entry_id=entry_id,
        source_ids=[entry_id],
        parent_ids=[],
        lineage_id=entry_id,
        version="v1",
        origin="memory",
        clean_or_contaminated="clean",
    )


def _call(call_id: str, *entry_ids: str) -> MethodCall:
    return MethodCall(
        call_id=call_id,
        stage="generation",
        raw_response="final",
        model="replay",
        source_spans=[_span(entry_id) for entry_id in entry_ids],
    )


def _envelope(
    entry_id: str,
    *,
    parents: tuple[str, ...] = (),
    predecessor: str | None = None,
) -> MemoryCardEnvelopeV3:
    content = f"content:{entry_id}"
    return MemoryCardEnvelopeV3(
        entry_id=entry_id,
        baseline="reflexion_style",
        semantic_kind="verbal_reflection",
        schema_version=MEMORY_CARD_V3,
        writer_id="reflexion",
        writer_event_id=f"write:{entry_id}",
        writer_stage="reflection",
        created_trial_id="trial",
        source_trial_ids=("trial",),
        source_outcome=False,
        trial_support_ids=("trial",),
        memory_support_ids=parents,
        direct_parent_ids=parents,
        version_predecessor_id=predecessor,
        order_key=1,
        native_component="reflections",
        content=content,
        content_hash=canonical_content_hash(content),
    )


def test_selected_answer_call_defines_context_and_target_spans() -> None:
    auxiliary = _call("auxiliary", "target")
    answer = _call("answer", "clean")

    context = _context(None, {}, "run", "trial", (), (auxiliary, answer), "answer")
    spans = _target_spans((auxiliary, answer), "answer", ("target",), "target-set")

    assert context is not None
    assert context.final_entry_ids == ["clean"]
    assert spans == ()


def test_lineage_preserves_version_predecessor_and_independent_origin() -> None:
    entries = (
        _RuntimeMemoryEntry(entry_id="root"),
        _RuntimeMemoryEntry(entry_id="child"),
        _RuntimeMemoryEntry(entry_id="grandchild"),
        _RuntimeMemoryEntry(entry_id="old"),
        _RuntimeMemoryEntry(entry_id="new"),
        _RuntimeMemoryEntry(entry_id="independent"),
    )

    nodes = _lineage(
        entries,
        ("root",),
        (
            _envelope("child", parents=("root",)),
            _envelope("grandchild", parents=("child",)),
            _envelope("new", predecessor="old"),
            _envelope("independent"),
        ),
        ("child", "grandchild", "new", "independent"),
    )

    by_id = {node.entry_id: node for node in nodes}
    assert by_id["grandchild"].injected_root_ids == ("root",)
    assert by_id["new"].version_predecessor_id == "old"
    assert by_id["independent"].direct_parent_ids == ()
    assert by_id["independent"].version_predecessor_id is None


def test_lineage_rejects_missing_claimed_parent() -> None:
    entry = _RuntimeMemoryEntry(
        entry_id="child",
        metadata=_RuntimeEntryMetadata(source_entry_ids=("missing",)),
    )

    with pytest.raises(ProductionRuntimeJoinError, match="PRODUCTION_LINEAGE_PARENT_MISSING"):
        _lineage(
            (entry,),
            (),
            (_envelope("child", parents=("missing",)),),
            ("child",),
        )


def test_lineage_rejects_unproved_independent_write() -> None:
    with pytest.raises(ProductionRuntimeJoinError, match="PRODUCTION_WRITER_ORIGIN_MISSING"):
        _lineage((_RuntimeMemoryEntry(entry_id="new"),), (), (), ("new",))


def test_dc_checkpoint_index_is_outside_strict_native_payload() -> None:
    snapshot = phase13_dc_rs_runtime.serialize(
        DcRsStateV3(archive=[], allow_unparented_strategies=True)
    )

    checkpoint = serialize_checkpoint(snapshot, checkpoint_index=1)

    assert checkpoint.checkpoint_index == 1
    assert "checkpoint_index" not in deserialize_checkpoint(checkpoint).native_state


def test_prefix_checkpoint_index_is_required_envelope_metadata() -> None:
    with pytest.raises(ValidationError, match="checkpoint_index"):
        PrefixCheckpointState.model_validate(
            {
                "schema_version": "phase13_main_prefix_checkpoint_v1",
                "baseline": "dc_rs",
                "checkpoint_id": "checkpoint",
                "checkpoint_identity_sha256": "a" * 64,
                "canonical_sha256": "b" * 64,
                "canonical_state_utf8": "{}",
            }
        )


def test_reflexion_stage_sequences_accept_lawful_early_stops() -> None:
    def call(trial: int, index: int, stage: str) -> MethodCall:
        return MethodCall(
            call_id=f"trial-{trial}:call:{index}",
            stage=stage,
            raw_response="ok",
            model="gpt-5.6-luna",
        )

    calls = (
        call(1, 1, "reflexion_generate"),
        call(2, 1, "reflexion_generate"),
        call(2, 2, "reflexion_reflect"),
        call(2, 3, "reflexion_generate"),
        call(3, 1, "reflexion_generate"),
        call(3, 2, "reflexion_reflect"),
        call(3, 3, "reflexion_generate"),
        call(3, 4, "reflexion_reflect"),
    )
    assert _reflexion_stage_sequences_valid(
        calls, expected_trials=3, kind="MEMORY_BEARING"
    )
    assert not _reflexion_stage_sequences_valid(
        (*calls, call(3, 5, "reflexion_generate")),
        expected_trials=3,
        kind="MEMORY_BEARING",
    )


def test_reflexion_stage_sequences_reject_suffix_only_prefix_stop() -> None:
    calls = (
        MethodCall(call_id="trial-1:call:1", stage="reflexion_generate", raw_response="ok", model="gpt-5.6-luna"),
        MethodCall(call_id="trial-1:call:2", stage="reflexion_reflect", raw_response="ok", model="gpt-5.6-luna"),
    )
    assert not _reflexion_stage_sequences_valid(
        calls, expected_trials=1, kind="MEMORY_BEARING"
    )


def test_reflexion_stage_sequences_reject_relabelled_call_ordinals() -> None:
    calls = (
        MethodCall(call_id="trial-1:call:1", stage="reflexion_generate", raw_response="ok", model="gpt-5.6-luna"),
        MethodCall(call_id="trial-1:call:3", stage="reflexion_reflect", raw_response="ok", model="gpt-5.6-luna"),
    )
    assert not _reflexion_stage_sequences_valid(
        calls, expected_trials=1, kind="CLEAN_PREFIX"
    )


def test_rag_persistent_carrier_is_not_the_retrieved_top_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Embedder:
        embedding_contract = {
            "dimension": 2,
            "normalized": True,
            "production_identity": "BAAI/bge-m3@5617a9f61b028005a4858fdac845db406aefb181",
            "provider": "test",
        }

        def encode_document(self, text: str) -> list[float]:
            del text
            return [1.0, 0.0]

        def encode_query(self, text: str) -> list[float]:
            del text
            return [1.0, 0.0]

    class Client:
        def chat(
            self,
            messages: list[dict[str, str]],
            model: str,
            config: dict[str, Any],
        ) -> LLMResponse:
            del messages, model, config
            return LLMResponse("final: 24", {}, {}, 0)

    monkeypatch.setattr(ordinary_runtime, "_validated_common_capacity_tokens", lambda: 8192)
    documents: list[Mapping[str, Any]] = [
        {"id": f"doc-{index}", "text": f"procedure {index}"}
        for index in range(4)
    ]
    corpora = build_branch_corpora(
        CleanCorpus.from_documents(documents, corpus_id="carrier"),
        {
            "false": {"id": "false", "text": "false"},
            "correct": {"id": "correct", "text": "correct"},
            "irrelevant": {"id": "irrelevant", "text": "irrelevant"},
        },
    )
    embedder = Embedder()
    indices = build_branch_indices(corpora, embedder, filter_policy=None)
    state = RagFrozenStateV3("clean", corpora.branches["clean"], indices.branches["clean"])
    snapshot = ordinary_runtime.PHASE13_CORE_BASELINE_REGISTRY["rag_frozen"].serialize_state(state)
    assert isinstance(snapshot, NativeState)
    checkpoint = serialize_checkpoint(snapshot, checkpoint_index=1)
    branch = LiveArmBranch("clean", "prefix", "prefix", checkpoint, state, 0)
    task = TaskInstance(
        sample_id="game24:rag-carrier",
        task_name="game24",
        input={"numbers": [1, 3, 4, 6], "target": 24},
        verifier_spec={"target": 24},
    )

    result = execute_prospective_ordinary(
        ProspectiveOrdinaryRun(
            task_name="game24",
            baseline="rag_frozen",
            arm="clean",
            branch=branch,
            run_id="rag-carrier",
            model="replay",
            client=Client(),
            allow_test_client=True,
            verifier=lambda _answer, _task: True,
            decoding={"temperature": 0.0},
            tasks=(task,),
        )
    )

    trial = result.trials[0]
    assert trial.state_before is not None
    assert trial.state_after is not None
    persistent_ids = tuple(
        entry if isinstance(entry, str) else entry.entry_id
        for entry in trial.state_before.entries
    )
    assert persistent_ids == tuple(
        entry if isinstance(entry, str) else entry.entry_id
        for entry in trial.state_after.entries
    )
    assert len(persistent_ids) == 4
    assert isinstance(trial.retrieval_event, RetrievalEvent)
    assert len(trial.retrieval_event.retrieved_entry_ids) < len(persistent_ids)
