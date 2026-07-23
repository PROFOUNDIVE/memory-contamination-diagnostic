from __future__ import annotations

import json
from typing import Literal

import pytest

from memcontam.baselines.bot_phase12 import (
    BoTContractError,
    BoTPhase12Adapter,
    BoTStateV3,
    BoTTrialContextV3,
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


_DISTILLED = json.dumps(
    {
        "key_information": "numbers = [1, 3, 4, 6], target = 24",
        "restrictions": "Use every number exactly once.",
        "distilled_task": "Construct an expression equal to 24.",
    }
)
_SOLVED = json.dumps(
    {
        "selected_structure": "retrieved-template",
        "solution_trace": "Use the retrieved procedure.",
        "final_answer": "final: 24",
    }
)


class _EmbeddingProvider:
    def encode_query(self, text: str) -> list[float]:
        return [1.0, 0.0] if text.startswith("{") else [0.0, 1.0]

    def encode_document(self, text: str) -> list[float]:
        del text
        return [1.0, 0.0]


def _task() -> TaskInstance:
    return TaskInstance(
        sample_id="game24-1",
        task_name="game24",
        input={"numbers": [1, 3, 4, 6], "target": 24},
    )


def _trial(
    *,
    branch: Literal["clean", "correct", "irrelevant", "contam", "filter"],
    used_ids: list[str],
    verifier,
) -> BoTTrialContextV3:
    return BoTTrialContextV3(
        task=_task(),
        client=ReplayClient(
            responses_by_sample={
                "game24-1": {
                    "bot_problem_distill": _DISTILLED,
                    "bot_instantiate_solve": _SOLVED,
                    "bot_thought_distill": json.dumps(
                        {
                            "description": "Build useful intermediate values.",
                            "template": "Build factors before combining them.",
                            "category": "procedure-based",
                            "explicitly_used_memory_ids": used_ids,
                        }
                    ),
                }
            }
        ),
        model="replay",
        run_id="phase12-bot",
        trial_id=f"phase12-bot:{branch}",
        condition_id="bot_style",
        branch=branch,
        config={"embedding_provider": _EmbeddingProvider()},
        order_key=2,
        verifier=verifier,
    )


def _native_template(entry_id: str, content: str) -> NativeEntry:
    return NativeEntry(
        entry_id=entry_id,
        semantic_kind="thought_template",
        schema_version=NATIVE_ENTRY_V1,
        native_component="buffer",
        content=content,
        content_hash=canonical_content_hash(content),
    )


def _memory_template(entry_id: str) -> MemoryEntry:
    return MemoryEntry(
        entry_id=entry_id,
        content="Use rational intermediate values.",
        memory_type="thought_template",
        metadata={
            "description": "Use rational intermediate values.",
            "category": "procedure-based",
        },
    )


def _envelope(
    entry: NativeEntry, *, trial_id: str | None, writer_id: str, stage: str
) -> MemoryCardEnvelopeV3:
    return MemoryCardEnvelopeV3(
        entry_id=entry.entry_id,
        baseline="bot_style",
        semantic_kind="thought_template",
        schema_version=MEMORY_CARD_V3,
        writer_id=writer_id,
        writer_event_id=f"event:{entry.entry_id}",
        writer_stage=stage,
        created_trial_id=trial_id,
        source_trial_ids=() if trial_id is None else (trial_id,),
        source_outcome=None,
        trial_support_ids=() if trial_id is None else (trial_id,),
        memory_support_ids=(),
        direct_parent_ids=(),
        version_predecessor_id=None,
        order_key=1 if trial_id is not None else 2,
        native_component="buffer",
        content=entry.content,
        content_hash=entry.content_hash,
    )


def test_rejects_bot_branch_without_two_active_clean_competitors() -> None:
    false_template = _native_template("false-template", "Require integer intermediate values.")

    with pytest.raises(BoTContractError, match="BOT_COMPETITORS_UNAVAILABLE"):
        BoTPhase12Adapter().execute(
            _trial(
                branch="contam", used_ids=[false_template.entry_id], verifier=lambda _answer: True
            ),
            BoTStateV3(entries=[false_template], active_capacity=2),
        )


def test_exposed_false_template_can_create_explicitly_parented_descendant() -> None:
    false_template = _native_template("false-template", "Require integer intermediate values.")
    clean_templates = [
        _native_template("z-clean-template-a", "Use rational intermediate values."),
        _native_template("z-clean-template-b", "Respect operator precedence."),
    ]
    state = BoTStateV3(
        entries=[false_template, *clean_templates],
        clean_competitor_ids=tuple(entry.entry_id for entry in clean_templates),
        active_capacity=4,
    )

    result = BoTPhase12Adapter().execute(
        _trial(branch="contam", used_ids=[false_template.entry_id], verifier=lambda _answer: True),
        state,
    )

    assert [call.stage for call in result.outcome.method_calls] == [
        "bot_problem_distill",
        "bot_instantiate_solve",
        "bot_thought_distill",
    ]
    assert result.prompt_decision.decision == "matched"
    assert result.context_event.final_entry_ids == [false_template.entry_id]
    assert result.native_novelty_decision.admitted
    assert result.lineage_status == "exact"
    assert result.native_entry is not None
    assert result.native_entry.direct_parent_ids == (false_template.entry_id,)
    assert result.write_envelope is not None
    assert result.write_envelope.direct_parent_ids == (false_template.entry_id,)
    assert result.write_envelope.source_outcome is None


def test_rejects_visibility_only_parent_and_verifier_dependent_novelty() -> None:
    visible = _memory_template("visible-template")
    clean_competitors = [
        _memory_template("z-clean-template-a"),
        _memory_template("z-clean-template-b"),
    ]
    without_explicit_parent = BoTPhase12Adapter().execute(
        _trial(branch="contam", used_ids=[], verifier=lambda _answer: True),
        BoTStateV3(
            entries=[visible, *clean_competitors],
            clean_competitor_ids=(
                visible.entry_id,
                *(entry.entry_id for entry in clean_competitors),
            ),
            active_capacity=4,
        ),
    )
    verifier_false = BoTPhase12Adapter().execute(
        _trial(branch="contam", used_ids=[], verifier=lambda _answer: False),
        BoTStateV3(
            entries=[visible, *clean_competitors],
            clean_competitor_ids=(
                visible.entry_id,
                *(entry.entry_id for entry in clean_competitors),
            ),
            active_capacity=4,
        ),
    )

    assert without_explicit_parent.context_event.final_entry_ids == [visible.entry_id]
    assert without_explicit_parent.lineage_status == "unavailable"
    assert without_explicit_parent.native_entry is not None
    assert without_explicit_parent.native_entry.direct_parent_ids == ()
    assert without_explicit_parent.native_novelty_decision == verifier_false.native_novelty_decision
    assert without_explicit_parent.outcome.verifier_result is True
    assert verifier_false.outcome.verifier_result is False


def test_filter_routes_only_the_post_novelty_candidate() -> None:
    clean = _native_template("clean-template", "Use rational intermediate values.")
    clean_competitor = _native_template("clean-template-z", "Respect operator precedence.")
    false = _native_template("false-template", "Require integer intermediate values.")
    clean_envelope = _envelope(
        clean,
        trial_id="prefix-clean",
        writer_id="bot_buffer_manager",
        stage="bot_thought_distill",
    )
    clean_competitor_envelope = _envelope(
        clean_competitor,
        trial_id="prefix-clean-2",
        writer_id="bot_buffer_manager",
        stage="bot_thought_distill",
    )
    false_envelope = _envelope(
        false,
        trial_id=None,
        writer_id="protocol_injector",
        stage="protocol_inject",
    )
    context = AdmissionContext(
        writer_event_ids=frozenset(
            {clean_envelope.writer_event_id, clean_competitor_envelope.writer_event_id}
        ),
        trial_record_ids=frozenset({"prefix-clean", "prefix-clean-2"}),
        evidence_envelopes=(clean_envelope, clean_competitor_envelope, false_envelope),
    )
    filtered = partition_native_checkpoint(
        serialize_checkpoint(
            NativeState("bot_style", (clean, clean_competitor, false), {"templates": []})
        ),
        context,
    )
    state = BoTStateV3(
        entries=[clean, clean_competitor, false],
        clean_competitor_ids=(clean.entry_id, clean_competitor.entry_id),
        active_capacity=3,
        filter_state=filtered,
        admission_context=context,
    )

    result = BoTPhase12Adapter().execute(
        _trial(branch="filter", used_ids=[clean.entry_id], verifier=lambda _answer: True), state
    )

    assert result.native_novelty_decision.admitted
    assert result.context_event.final_entry_ids == [clean.entry_id]
    assert result.filter_transition is not None
    assert result.filter_transition.decision.admitted
    assert false.entry_id not in {
        entry.entry_id if isinstance(entry, NativeEntry) else entry
        for entry in result.filter_transition.reader_entries
    }
