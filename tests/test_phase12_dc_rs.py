from __future__ import annotations

from dataclasses import replace

import pytest

from memcontam.baselines.dynamic_cheatsheet_phase12 import (
    DcRsContractError,
    DcRsPhase12Adapter,
    DcRsStateV3,
    DcRsTrialContextV3,
    curate_pre_generation,
)
from memcontam.clients.replay import ReplayClient
from memcontam.memory.admission import AdmissionContext
from memcontam.memory.cards_v3 import MEMORY_CARD_V3, MemoryCardEnvelopeV3, canonical_content_hash
from memcontam.memory.checkpoint_v3 import (
    NATIVE_ENTRY_V1,
    NativeEntry,
    NativeState,
    serialize_checkpoint,
)
from memcontam.memory.filtered_state import partition_native_checkpoint
from memcontam.memory.stores import MemoryEntry
from memcontam.tasks.base import TaskInstance


class _EmbeddingProvider:
    @property
    def metadata(self) -> dict[str, object]:
        return {
            "model_id": "phase12-test",
            "revision": "test",
            "embedding_library_version": "test",
            "vector_dimension": 2,
        }

    def encode_document(self, text: str) -> list[float]:
        del text
        return [1.0, 0.0]

    def encode_query(self, text: str) -> list[float]:
        del text
        return [1.0, 0.0]


def _task() -> TaskInstance:
    return TaskInstance(
        sample_id="game24-1",
        task_name="game24",
        input={"numbers": [1, 3, 4, 6], "target": 24},
    )


def _trial(*, current_outcome: object | None = None) -> DcRsTrialContextV3:
    return DcRsTrialContextV3(
        task=_task(),
        client=ReplayClient(
            responses_by_sample={
                "game24-1": {
                    "dc_rs_synthesize": (
                        "<cheatsheet>Use the archived factor route.</cheatsheet>"
                        "<source_ids>archive-root</source_ids>"
                    ),
                    "dc_rs_generate": "final: 24",
                }
            }
        ),
        model="replay",
        run_id="phase12-dc-rs",
        trial_id="phase12-dc-rs:clean",
        condition_id="dc_optional",
        branch="clean",
        config={},
        order_key=2,
        verifier=lambda answer, _task: answer == "24",
        current_outcome=current_outcome,
    )


def _archive_root() -> MemoryEntry:
    return MemoryEntry(
        entry_id="archive-root",
        content='{"numbers":[1,3,4,6],"target":24}',
        memory_type="dc_rs_io_pair",
        metadata={"generated_output": "final: 24"},
    )


def test_archive_root_yields_exact_strategy_only_with_declared_parent(tmp_path) -> None:
    state = DcRsStateV3(archive=[_archive_root()])

    result = DcRsPhase12Adapter(
        embedding_provider=_EmbeddingProvider(), cache_dir=tmp_path
    ).execute(_trial(), state)

    assert [call.stage for call in result.outcome.method_calls] == [
        "dc_rs_synthesize",
        "dc_rs_generate",
    ]
    assert result.strategy_candidate.lineage_status == "exact"
    assert result.strategy_entry is not None
    assert result.strategy_entry.direct_parent_ids == ("archive-root",)
    assert result.strategy_admission.admitted
    assert result.archive_entry is not None
    assert result.archive_entry.metadata["generated_output"] == "final: 24"


def test_rejects_direct_strategy_post_outcome_and_implicit_parent_union(tmp_path) -> None:
    direct_strategy = MemoryEntry(
        entry_id="strategy-root",
        content="Inject this strategy directly.",
        memory_type="dynamic_cheatsheet",
    )

    with pytest.raises(DcRsContractError, match="DIRECT_STRATEGY_INJECTION"):
        DcRsStateV3(archive=[direct_strategy])

    adapter = DcRsPhase12Adapter(embedding_provider=_EmbeddingProvider(), cache_dir=tmp_path)
    with pytest.raises(DcRsContractError, match="CURRENT_OUTCOME_LEAKAGE"):
        adapter.execute(
            replace(_trial(), current_outcome=False), DcRsStateV3(archive=[_archive_root()])
        )

    with pytest.raises(DcRsContractError, match="IMPLICIT_PARENT_UNION"):
        curate_pre_generation(
            "<cheatsheet>Use the archived factor route.</cheatsheet>",
            fallback_strategy="",
            retrieved_archive_ids=("archive-root",),
            inferred_parent_ids=("archive-root",),
        )


def test_quarantined_archive_parent_rejects_candidate_and_retains_active_strategy(tmp_path) -> None:
    clean = _native_archive("archive-root")
    quarantined = _native_archive("quarantined-root")
    prior_strategy = NativeEntry(
        entry_id="prior-strategy",
        semantic_kind="dynamic_cheatsheet",
        schema_version=NATIVE_ENTRY_V1,
        native_component="strategy",
        content="Use the active route.",
        content_hash=canonical_content_hash("Use the active route."),
        direct_parent_ids=(clean.entry_id,),
    )
    clean_envelope = _envelope(clean, "prefix-clean", 1, "dc_archive_writer", "dc_rs_generate")
    strategy_envelope = _envelope(
        prior_strategy,
        "prefix-strategy",
        2,
        "dc_strategy_writer",
        "dc_rs_synthesize",
        parents=(clean.entry_id,),
    )
    quarantined_envelope = _envelope(quarantined, None, 3, "protocol_injector", "protocol_inject")
    context = AdmissionContext(
        writer_event_ids=frozenset(
            {clean_envelope.writer_event_id, strategy_envelope.writer_event_id}
        ),
        trial_record_ids=frozenset({"prefix-clean", "prefix-strategy"}),
        evidence_envelopes=(clean_envelope, strategy_envelope, quarantined_envelope),
    )
    filtered = partition_native_checkpoint(
        serialize_checkpoint(
            NativeState(
                "dynamic_cheatsheet_rs_optional",
                (clean, prior_strategy, quarantined),
                {"archive": [], "strategy": prior_strategy.content},
            )
        ),
        context,
    )
    trial = replace(
        _trial(),
        branch="filter",
        trial_id="phase12-dc-rs:filter",
        client=ReplayClient(
            responses_by_sample={
                "game24-1": {
                    "dc_rs_synthesize": (
                        "<cheatsheet>Use the quarantined route.</cheatsheet>"
                        "<source_ids>quarantined-root</source_ids>"
                    ),
                    "dc_rs_generate": "final: 24",
                }
            }
        ),
    )
    state = DcRsStateV3(
        archive=[_archive_root(), _archive_memory("quarantined-root")],
        strategies=[prior_strategy],
        filter_state=filtered,
        admission_context=context,
    )

    result = DcRsPhase12Adapter(
        embedding_provider=_EmbeddingProvider(), cache_dir=tmp_path
    ).execute(trial, state)

    assert result.strategy_candidate.lineage_status == "approximate"
    assert result.strategy_admission.reason == "PARENT_QUARANTINED"
    assert not result.strategy_admission.admitted
    assert "Use the active route." in result.outcome.method_calls[-1].messages[0]["content"]
    assert (
        "Use the quarantined route." not in result.outcome.method_calls[-1].messages[0]["content"]
    )


def _archive_memory(entry_id: str) -> MemoryEntry:
    return MemoryEntry(
        entry_id=entry_id,
        content='{"numbers":[1,3,4,6],"target":24}',
        memory_type="dc_rs_io_pair",
        metadata={"generated_output": "final: wrong"},
    )


def _native_archive(entry_id: str) -> NativeEntry:
    content = '{"input":"prior input","raw_output":"final: prior"}'
    return NativeEntry(
        entry_id=entry_id,
        semantic_kind="dc_rs_io_pair",
        schema_version=NATIVE_ENTRY_V1,
        native_component="archive",
        content=content,
        content_hash=canonical_content_hash(content),
    )


def _envelope(
    entry: NativeEntry,
    trial_id: str | None,
    order_key: int,
    writer_id: str,
    writer_stage: str,
    *,
    parents: tuple[str, ...] = (),
) -> MemoryCardEnvelopeV3:
    return MemoryCardEnvelopeV3(
        entry_id=entry.entry_id,
        baseline="dynamic_cheatsheet_rs_optional",
        semantic_kind=entry.semantic_kind,
        schema_version=MEMORY_CARD_V3,
        writer_id=writer_id,
        writer_event_id=f"event:{entry.entry_id}",
        writer_stage=writer_stage,
        created_trial_id=trial_id,
        source_trial_ids=() if trial_id is None else (trial_id,),
        source_outcome=None,
        trial_support_ids=() if trial_id is None else (trial_id,),
        memory_support_ids=parents,
        direct_parent_ids=parents,
        version_predecessor_id=None,
        order_key=order_key,
        native_component=entry.native_component,
        content=entry.content,
        content_hash=entry.content_hash,
    )
