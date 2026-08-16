from __future__ import annotations

from pathlib import Path
from dataclasses import replace
from typing import cast

import pytest

from memcontam.baselines import dynamic_cheatsheet_phase12 as dc
from memcontam.baselines.dynamic_cheatsheet_phase12 import DcRsStateV3
from memcontam.clients.replay import ReplayClient
from memcontam.contamination.phase12.registry import load_candidate_registry
from memcontam.contamination.phase12.renderers import RendererError
from memcontam.experiment.phase12.game24_runner import Game24RuntimeContext, RuntimeIdentities
from memcontam.experiment.phase12.live_branch import build_live_reduced_main_branches
from memcontam.experiment.phase12.runtime_registry import (
    PHASE13_CORE_BASELINE_REGISTRY,
    RuntimeStateError,
)
from memcontam.experiment.phase13_dc_rs_runtime import Phase13DcRsContext
from memcontam.memory.checkpoint_v3 import NativeEntry, NativeState, serialize_checkpoint
from memcontam.memory.cards_v3 import canonical_content_hash
from memcontam.memory.stores import MemoryEntry
from memcontam.tasks.base import TaskInstance


REGISTRY_PATH = Path("data/phase12/registries/candidate_registry_v1.json")


class _EmbeddingProvider:
    @property
    def metadata(self) -> dict[str, object]:
        return {
            "model_id": "BAAI/bge-m3",
            "revision": "5617a9f61b028005a4858fdac845db406aefb181",
            "embedding_library_version": "test",
            "vector_dimension": 1024,
            "normalize_embeddings": True,
        }

    def encode_document(self, text: str) -> list[float]:
        del text
        return [1.0, 0.0]

    def encode_query(self, text: str) -> list[float]:
        del text
        return [1.0, 0.0]


class _WrongEmbeddingProvider(_EmbeddingProvider):
    @property
    def metadata(self) -> dict[str, object]:
        return {
            "model_id": "text-embedding-3-small",
            "revision": "provider-default",
            "embedding_library_version": "test",
            "vector_dimension": 1536,
            "normalize_embeddings": True,
        }


class _BombClient:
    def __init__(self) -> None:
        self.calls = 0

    def chat(self, *_args: object, **_kwargs: object) -> None:
        self.calls += 1
        raise AssertionError("invalid state reached LLM")


def _task() -> TaskInstance:
    return TaskInstance(
        sample_id="mmlu_pro_engineering:11775",
        task_name="mmlu_pro_engineering",
        input={"question": "Which option?", "options": ["one", "two", "three", "four"]},
        verifier_spec={"answer_index": 1, "answer_label": "B"},
        metadata={"upstream_question_id": 11775},
    )


def _state() -> DcRsStateV3:
    return DcRsStateV3(
        archive=[
            MemoryEntry(
                entry_id="archive-root",
                content=(
                    '{"input":{"options":["one","two"],"question":"prior"},'
                    '"task_name":"mmlu_pro_engineering"}'
                ),
                memory_type="dc_rs_io_pair",
                source_trial_id="run-1:trial:1:mmlu_pro_engineering:prior",
                metadata={
                    "generated_output": "full prior reasoning\nfinal: A",
                    "parsed_answer": "A",
                },
            )
        ]
    )


def _memory_entry(entry: MemoryEntry | NativeEntry) -> MemoryEntry:
    assert isinstance(entry, MemoryEntry)
    return entry


def _context(*, tool_mode: str = "text_only") -> Phase13DcRsContext:
    return Phase13DcRsContext(
        task=_task(),
        client=ReplayClient(
            responses_by_sample={
                _task().sample_id: {
                    "dc_rs_synthesize": (
                        "<cheatsheet>rewritten engineering strategy</cheatsheet>"
                        "<source_ids>archive-root</source_ids>"
                    ),
                    "dc_rs_generate": "visible current reasoning\nfinal: B",
                }
            }
        ),
        model="replay",
        verifier=lambda answer, task: answer == task.verifier_spec["answer_label"],
        decoding={"temperature": 0.0},
        branch="clean",
        identities=RuntimeIdentities(
            "run-1",
            "run-1:trial:2:mmlu_pro_engineering:11775",
            2,
            "dc_rs",
        ),
        embedding_provider=_EmbeddingProvider(),
        baseline_configs={
            "dc_rs": {
                "embedding_mode": "test_double",
                "tool_mode": tool_mode,
                "serialized_cheatsheet_budget_bytes": 1024,
            }
        },
        initial_states={"dc_rs": _state()},
    )


def test_dc_rs_runtime_is_first_class_text_only_retrieve_synthesize_generate() -> None:
    entry = PHASE13_CORE_BASELINE_REGISTRY["dc_rs"]
    state = entry.initial_state(_context())

    result = entry.execute_trial(_context(), state)

    assert [call.stage for call in result.outcome.method_calls] == [
        "dc_rs_synthesize",
        "dc_rs_generate",
    ]
    synthesis_prompt = result.outcome.method_calls[0].messages[0]["content"]
    generation_prompt = result.outcome.method_calls[1].messages[0]["content"]
    assert "full prior reasoning\nfinal: A" in synthesis_prompt
    assert "rewritten engineering strategy" in generation_prompt
    assert "full prior reasoning\nfinal: A" not in generation_prompt
    assert result.outcome.verifier_result is True
    assert isinstance(result.state, DcRsStateV3)
    archive_entry = _memory_entry(result.state.archive[-1])
    assert archive_entry.metadata["generated_output"] == (
        "visible current reasoning\nfinal: B"
    )
    assert archive_entry.metadata["parsed_answer"] == "B"
    assert result.native_entries[0].native_component == "archive"
    assert result.write_envelopes[0].writer_stage == "dc_rs_generate"


def test_dc_rs_first_trial_generates_from_transient_whole_cheatsheet() -> None:
    task = _task()
    context = replace(
        _context(),
        client=ReplayClient(
            responses_by_sample={
                task.sample_id: {
                    "dc_rs_synthesize": "<cheatsheet>first-trial rewritten guide</cheatsheet>",
                    "dc_rs_generate": "visible current reasoning\nfinal: B",
                }
            }
        ),
        initial_states={"dc_rs": DcRsStateV3(archive=[])},
    )
    entry = PHASE13_CORE_BASELINE_REGISTRY["dc_rs"]

    result = entry.execute_trial(context, entry.initial_state(context))

    generation_prompt = result.outcome.method_calls[1].messages[0]["content"]
    assert "first-trial rewritten guide" in generation_prompt
    assert isinstance(result.state, DcRsStateV3)
    assert result.state.strategies is not None
    assert result.state.strategies[-1].content == "first-trial rewritten guide"
    assert len(result.state.archive) == 1


def test_dc_rs_accepted_rewrite_controls_generation_without_audit_source_tags() -> None:
    task = _task()
    context = replace(
        _context(),
        client=ReplayClient(
            responses_by_sample={
                task.sample_id: {
                    "dc_rs_synthesize": "<cheatsheet>fresh accepted rewrite</cheatsheet>",
                    "dc_rs_generate": "visible current reasoning\nfinal: B",
                }
            }
        ),
    )
    entry = PHASE13_CORE_BASELINE_REGISTRY["dc_rs"]

    result = entry.execute_trial(context, entry.initial_state(context))

    assert "fresh accepted rewrite" in result.outcome.method_calls[1].messages[0]["content"]
    assert isinstance(result.state, DcRsStateV3)
    assert result.state.strategies is not None
    assert result.state.strategies[-1].content == "fresh accepted rewrite"


def test_dc_rs_persists_archive_before_rewritten_strategy() -> None:
    entry = PHASE13_CORE_BASELINE_REGISTRY["dc_rs"]
    context = _context()

    result = entry.execute_trial(context, entry.initial_state(context))

    assert [envelope.writer_stage for envelope in result.write_envelopes] == [
        "dc_rs_generate",
        "dc_rs_synthesize",
    ]
    assert [envelope.order_key for envelope in result.write_envelopes] == [2001, 2002]
    assert [native.native_component for native in result.native_entries] == [
        "archive",
        "strategy",
    ]


def test_dc_rs_runtime_checkpoint_round_trip_preserves_full_visible_responses() -> None:
    entry = PHASE13_CORE_BASELINE_REGISTRY["dc_rs"]
    context = _context()
    executed = entry.execute_trial(context, entry.initial_state(context))

    snapshot = entry.serialize_state(executed.state)
    next_context = replace(
        context,
        identities=RuntimeIdentities(
            "run-1",
            "run-1:trial:3:mmlu_pro_engineering:next",
            3,
            "dc_rs",
        ),
    )
    restored = entry.restore_state(snapshot, next_context)

    assert isinstance(snapshot, NativeState)
    assert snapshot.baseline == "dc_rs"
    assert isinstance(restored, DcRsStateV3)
    assert [_memory_entry(row).metadata["generated_output"] for row in restored.archive] == [
        "full prior reasoning\nfinal: A",
        "visible current reasoning\nfinal: B",
    ]
    assert isinstance(executed.state, DcRsStateV3)
    assert restored.archive == executed.state.archive


@pytest.mark.parametrize("mutation", ["stale_hash", "reordered", "extra_component"])
def test_dc_rs_runtime_rejects_inconsistent_checkpoint_entries(mutation: str) -> None:
    entry = PHASE13_CORE_BASELINE_REGISTRY["dc_rs"]
    context = _context()
    executed = entry.execute_trial(context, entry.initial_state(context))
    snapshot = cast(NativeState, entry.serialize_state(executed.state))
    entries = list(snapshot.entries)
    if mutation == "stale_hash":
        archive = cast(NativeEntry, entries[0])
        entries[0] = replace(archive, content=archive.content + "tampered")
    elif mutation == "reordered":
        entries.reverse()
    else:
        entries.append(
            NativeEntry(
                entry_id="unexpected",
                semantic_kind="dc_rs_io_pair",
                schema_version="phase12_native_entry_v1",
                native_component="unexpected",
                content="unexpected",
                content_hash=canonical_content_hash("unexpected"),
            )
        )

    with pytest.raises(RuntimeStateError, match="INVALID_DC_RS_SNAPSHOT"):
        entry.restore_state(replace(snapshot, entries=tuple(entries)), context)


def test_dc_rs_runtime_rejects_snapshot_with_unresolved_strategy_parent() -> None:
    entry = PHASE13_CORE_BASELINE_REGISTRY["dc_rs"]
    context = _context()
    executed = entry.execute_trial(context, entry.initial_state(context))
    snapshot = cast(NativeState, entry.serialize_state(executed.state))
    entries = list(snapshot.entries)
    strategy = cast(NativeEntry, entries[-1])
    entries[-1] = replace(strategy, direct_parent_ids=("never-retrieved",))

    with pytest.raises(RuntimeStateError, match="INVALID_DC_RS_SNAPSHOT"):
        entry.restore_state(replace(snapshot, entries=tuple(entries)), context)


def test_dc_rs_runtime_rejects_false_core_strategy_mode() -> None:
    entry = PHASE13_CORE_BASELINE_REGISTRY["dc_rs"]
    context = _context()
    executed = entry.execute_trial(context, entry.initial_state(context))
    snapshot = cast(NativeState, entry.serialize_state(executed.state))
    native_state = dict(snapshot.native_state)
    native_state["allow_unparented_strategies"] = False

    with pytest.raises(RuntimeStateError, match="INVALID_DC_RS_SNAPSHOT"):
        entry.restore_state(replace(snapshot, native_state=native_state), context)


@pytest.mark.parametrize(
    "mutation",
    [
        "unresolved_parent",
        "stale_hash",
        "invalid_tool_trace",
        "null_tool_trace",
        "gold_content",
        "strategy_root",
    ],
)
def test_dc_rs_runtime_rejects_invalid_initial_state_before_llm(mutation: str) -> None:
    archive = [_memory_entry(entry) for entry in _state().archive]
    content = "strategy"
    strategy = NativeEntry(
        entry_id="strategy",
        semantic_kind="dynamic_cheatsheet",
        schema_version="phase12_native_entry_v1",
        native_component="strategy",
        content=content,
        content_hash=canonical_content_hash(content),
        direct_parent_ids=("archive-root",),
    )
    if mutation == "unresolved_parent":
        strategy = replace(strategy, direct_parent_ids=("never-retrieved",))
    elif mutation == "stale_hash":
        strategy = replace(strategy, content_hash="0" * 64)
    elif mutation == "invalid_tool_trace":
        archive[0] = archive[0].model_copy(
            update={"metadata": {**archive[0].metadata, "tool_trace": 123}}
        )
    elif mutation == "null_tool_trace":
        archive[0] = archive[0].model_copy(
            update={"metadata": {**archive[0].metadata, "tool_trace": None}}
        )
    elif mutation == "gold_content":
        archive[0] = archive[0].model_copy(
            update={
                "content": (
                    '{"input":{"options":["one","two"],"question":"prior"},'
                    '"task_name":"mmlu_pro_engineering",'
                    '"verifier_spec":{"answer_label":"A"}}'
                )
            }
        )
    client = _BombClient()
    archive_state: list[MemoryEntry | NativeEntry] = [*archive]
    state = DcRsStateV3(
        archive=archive_state,
        strategies=[strategy],
        allow_unparented_strategies=True,
    )
    if mutation == "strategy_root":
        state.injected_root_id = strategy.entry_id
    context = replace(
        _context(),
        client=client,
        initial_states={
            "dc_rs": state
        },
    )
    entry = PHASE13_CORE_BASELINE_REGISTRY["dc_rs"]

    with pytest.raises(RuntimeStateError, match="INVALID_DC_RS_STATE"):
        entry.initial_state(context)
    assert client.calls == 0


@pytest.mark.parametrize(
    "source_trial_id",
    (
        "run-1:trial:3:mmlu_pro_engineering:future",
        "run-2:trial:1:mmlu_pro_engineering:other-run",
        "unproven-prior-trial",
    ),
)
def test_dc_rs_runtime_rejects_archive_without_proven_prior_trajectory(
    source_trial_id: str,
) -> None:
    archive = _memory_entry(_state().archive[0]).model_copy(
        update={"source_trial_id": source_trial_id}
    )
    client = _BombClient()
    context = replace(
        _context(),
        client=client,
        initial_states={"dc_rs": DcRsStateV3(archive=[archive])},
    )
    entry = PHASE13_CORE_BASELINE_REGISTRY["dc_rs"]

    with pytest.raises(RuntimeStateError, match="DC_RS_ORDINARY_HISTORY_UNPROVEN"):
        entry.initial_state(context)
    assert client.calls == 0


def test_dc_rs_runtime_rejects_cross_task_archive_before_curator() -> None:
    archive = _memory_entry(_state().archive[0]).model_copy(
        update={
            "content": (
                '{"input":{"options":["one","two"],"question":"prior"},'
                '"task_name":"mmlu_pro_physics"}'
            )
        }
    )
    client = _BombClient()
    context = replace(
        _context(),
        client=client,
        initial_states={"dc_rs": DcRsStateV3(archive=[archive])},
    )
    entry = PHASE13_CORE_BASELINE_REGISTRY["dc_rs"]

    with pytest.raises(RuntimeStateError, match="DC_RS_ORDINARY_HISTORY_UNPROVEN"):
        entry.initial_state(context)
    assert client.calls == 0


def test_dc_rs_runtime_rejects_unproven_current_identity_on_empty_state() -> None:
    client = _BombClient()
    context = replace(
        _context(),
        client=client,
        identities=RuntimeIdentities("run-1", "trial-1", 1, "dc_rs"),
        initial_states={"dc_rs": DcRsStateV3(archive=[])},
    )
    entry = PHASE13_CORE_BASELINE_REGISTRY["dc_rs"]

    with pytest.raises(RuntimeStateError, match="DC_RS_ORDINARY_HISTORY_UNPROVEN"):
        entry.initial_state(context)
    assert client.calls == 0


def test_dc_rs_runtime_rejects_code_augmented_mode() -> None:
    entry = PHASE13_CORE_BASELINE_REGISTRY["dc_rs"]
    context = _context(tool_mode="python_sandbox")

    with pytest.raises(RuntimeStateError, match="DC_RS_TEXT_ONLY_REQUIRED"):
        entry.execute_trial(context, entry.initial_state(context))


def test_dc_rs_runtime_rejects_non_bge_embedding_substitution() -> None:
    entry = PHASE13_CORE_BASELINE_REGISTRY["dc_rs"]
    context = replace(_context(), embedding_provider=_WrongEmbeddingProvider())

    with pytest.raises(RuntimeStateError, match="DC_RS_BGE_M3_CONTRACT_REQUIRED"):
        entry.execute_trial(context, entry.initial_state(context))


def test_dc_rs_runtime_allows_spoofable_provider_only_in_explicit_replay_mode() -> None:
    context = _context()
    context = replace(
        context,
        baseline_configs={
            "dc_rs": {
                "tool_mode": "text_only",
                "serialized_cheatsheet_budget_bytes": 1024,
            }
        },
    )
    entry = PHASE13_CORE_BASELINE_REGISTRY["dc_rs"]

    with pytest.raises(RuntimeStateError, match="DC_RS_BGE_M3_CONTRACT_REQUIRED"):
        entry.execute_trial(context, entry.initial_state(context))


def test_dc_rs_rejects_source_id_that_was_active_but_not_retrieved() -> None:
    task = _task()
    state = _state()
    state.archive.extend(
        MemoryEntry(
            entry_id=f"archive-{index}",
            content=(
                f'{{"input":{{"options":["one","two"],"question":"prior {index}"}},'
                '"task_name":"mmlu_pro_engineering"}'
            ),
            memory_type="dc_rs_io_pair",
            source_trial_id=f"run-1:trial:{index + 2}:mmlu_pro_engineering:prior-{index}",
            metadata={"generated_output": f"reasoning {index}\nfinal: A"},
        )
        for index in range(3)
    )
    context = replace(
        _context(),
        client=ReplayClient(
            responses_by_sample={
                task.sample_id: {
                    "dc_rs_synthesize": (
                        "<cheatsheet>unsupported rewrite</cheatsheet>"
                        "<source_ids>archive-root</source_ids>"
                    ),
                    "dc_rs_generate": "final: B",
                }
            }
        ),
        initial_states={"dc_rs": state},
        identities=RuntimeIdentities(
            "run-1",
            "run-1:trial:5:mmlu_pro_engineering:11775",
            5,
            "dc_rs",
        ),
    )
    entry = PHASE13_CORE_BASELINE_REGISTRY["dc_rs"]

    with pytest.raises(RuntimeStateError, match="EXPLICIT_PARENT_NOT_ACTIVE"):
        entry.execute_trial(context, entry.initial_state(context))


def test_dc_rs_rejects_rewrite_over_explicit_serialized_budget() -> None:
    task = _task()
    context = replace(
        _context(),
        client=ReplayClient(
            responses_by_sample={
                task.sample_id: {
                    "dc_rs_synthesize": f"<cheatsheet>{'x' * 1025}</cheatsheet>",
                    "dc_rs_generate": "final: B",
                }
            }
        ),
    )
    entry = PHASE13_CORE_BASELINE_REGISTRY["dc_rs"]

    with pytest.raises(RuntimeStateError, match="DC_RS_CHEATSHEET_BUDGET_EXCEEDED"):
        entry.execute_trial(context, entry.initial_state(context))


def test_dc_rs_rejects_oversized_existing_strategy_before_curation() -> None:
    context = replace(
        _context(),
        initial_states={
            "dc_rs": DcRsStateV3(
                archive=[],
                strategies=[
                    NativeEntry(
                        entry_id="oversized",
                        semantic_kind="dynamic_cheatsheet",
                        schema_version="phase12_native_entry_v1",
                        native_component="strategy",
                        content="x" * 1025,
                        content_hash=canonical_content_hash("x" * 1025),
                    )
                ],
                allow_unparented_strategies=True,
            )
        },
    )
    entry = PHASE13_CORE_BASELINE_REGISTRY["dc_rs"]

    with pytest.raises(RuntimeStateError, match="DC_RS_CHEATSHEET_BUDGET_EXCEEDED"):
        entry.execute_trial(context, entry.initial_state(context))


def test_historical_dc_rs_keeps_active_archive_parent_rule() -> None:
    task = _task()
    state = _state()
    state.archive.extend(
        MemoryEntry(
            entry_id=f"archive-{index}",
            content=f'{{"question":"prior {index}"}}',
            memory_type="dc_rs_io_pair",
            source_trial_id=f"prior-{index}",
            metadata={"generated_output": f"reasoning {index}\nfinal: A"},
        )
        for index in range(3)
    )
    client = ReplayClient(
        responses_by_sample={
            task.sample_id: {
                "dc_rs_synthesize": (
                    "<cheatsheet>historical rewrite</cheatsheet>"
                    "<source_ids>archive-root</source_ids>"
                ),
                "dc_rs_generate": "final: B",
            }
        }
    )
    result = dc.DcRsPhase12Adapter(embedding_provider=_EmbeddingProvider()).execute(
        dc.DcRsTrialContextV3(
            task=task,
            client=client,
            model="replay",
            run_id="historical-run",
            trial_id="historical-trial",
            condition_id="clean",
            branch="clean",
            config={"baseline": "dynamic_cheatsheet_rs_optional", "tool_mode": "text_only"},
            order_key=1,
        ),
        state,
    )

    assert result.outcome.status == "succeeded"


def test_dc_rs_runtime_accepts_historical_task_native_context() -> None:
    task = TaskInstance(
        sample_id="game24:1",
        task_name="game24",
        input={"numbers": [1, 3, 4, 6]},
        verifier_spec={"target": 24},
    )
    context = Game24RuntimeContext(
        task=task,
        client=ReplayClient(
            responses_by_sample={
                task.sample_id: {
                    "dc_rs_synthesize": "<cheatsheet>game strategy</cheatsheet>",
                    "dc_rs_generate": "final: (6 / (1 - 3 / 4))",
                }
            }
        ),
        model="replay",
        verifier=lambda _answer, _task: False,
        decoding={"temperature": 0.0},
        branch="clean",
        identities=RuntimeIdentities("run-1", "run-1:trial:1:game24:1", 1),
        embedding_provider=_EmbeddingProvider(),
        baseline_configs={
            "dc_rs": {
                "embedding_mode": "test_double",
                "serialized_cheatsheet_budget_bytes": 1024,
            }
        },
        initial_states={"dc_rs": DcRsStateV3(archive=[])},
    )
    entry = PHASE13_CORE_BASELINE_REGISTRY["dc_rs"]

    result = entry.execute_trial(context, entry.initial_state(context))

    assert result.outcome.status == "succeeded"
    assert isinstance(result.state, DcRsStateV3)
    assert '"task_name":"game24"' in result.state.archive[0].content


def test_dc_rs_runtime_rejects_gold_in_current_input_before_llm() -> None:
    client = _BombClient()
    context = replace(
        _context(),
        task=TaskInstance(
            sample_id="mmlu_pro_engineering:gold",
            task_name="mmlu_pro_engineering",
            input={
                "question": "Which?",
                "options": ["one", "two"],
                "answer_label": "B",
            },
            verifier_spec={"answer_index": 1, "answer_label": "B"},
        ),
        client=client,
    )
    entry = PHASE13_CORE_BASELINE_REGISTRY["dc_rs"]

    with pytest.raises(RuntimeStateError, match="INVALID_DC_RS_TASK"):
        entry.execute_trial(context, entry.initial_state(context))
    assert client.calls == 0


def test_dc_rs_matched_intervention_stays_blocked_without_frozen_raw_interaction() -> None:
    game24 = TaskInstance(
        sample_id="game24-main-1",
        task_name="game24",
        input={"numbers": [1, 3, 4, 6]},
        verifier_spec={"target": 24},
    )
    context = Game24RuntimeContext(
        task=game24,
        client=ReplayClient(responses_by_sample={}),
        model="replay",
        verifier=lambda _answer, _task: False,
        decoding={"temperature": 0.0},
        branch="clean",
        identities=RuntimeIdentities("run-1", "trial-1", 1),
        embedding_provider=_EmbeddingProvider(),
        initial_states={"dc_rs": DcRsStateV3(archive=[])},
    )
    entry = PHASE13_CORE_BASELINE_REGISTRY["dc_rs"]
    prefix = serialize_checkpoint(
        cast(
            NativeState,
            entry.serialize_state(DcRsStateV3([], allow_unparented_strategies=True)),
        )
    )

    with pytest.raises(RendererError, match="DC_RS_INTERVENTION_REGISTRY_REQUIRED"):
        build_live_reduced_main_branches(
            prefix=prefix,
            context=context,
            candidate_registry=load_candidate_registry(REGISTRY_PATH),
            registry=PHASE13_CORE_BASELINE_REGISTRY,
        )
