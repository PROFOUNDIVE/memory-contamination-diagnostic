from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from memcontam.contamination.phase12.registry import load_candidate_registry
from memcontam.contamination.phase12.renderers import RendererRegistry
from memcontam.experiment.phase12.branching import BranchSet
from memcontam.experiment.phase12.contracts import MemoryArmExecutionKey, RunTemplateSpec
from memcontam.memory.admission import AdmissionContext
from memcontam.memory.cards_v3 import MemoryCardEnvelopeV3, canonical_content_hash
from memcontam.memory.checkpoint_v3 import NativeEntry, NativeState, serialize_checkpoint
from memcontam.tasks.base import TaskInstance


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "phase12" / "FX-BRANCH-001.json"
REGISTRY_PATH = ROOT / "data" / "phase12" / "registries" / "candidate_registry_v1.json"
_WRITER = ("full_history_transcript", "fh_appender", "full_history_generate", "history")


def _fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _branches() -> BranchSet:
    from memcontam.experiment.phase12.branching import build_matched_branches

    prefix = _fixture()["baseline_prefixes"]["fh_bounded"]["checkpoint"]
    checkpoint = serialize_checkpoint(NativeState.from_mapping(prefix))
    semantic_kind, writer_id, writer_stage, native_component = _WRITER
    envelopes = tuple(
        MemoryCardEnvelopeV3(
            entry_id=entry_id,
            baseline="fh_bounded",
            semantic_kind=semantic_kind,
            schema_version="memory_card_v3",
            writer_id=writer_id,
            writer_event_id=f"event-{entry_id}",
            writer_stage=writer_stage,
            created_trial_id=f"trial-{entry_id}",
            source_trial_ids=(f"trial-{entry_id}",),
            source_outcome=None,
            trial_support_ids=(f"trial-{entry_id}",),
            memory_support_ids=(),
            direct_parent_ids=(),
            version_predecessor_id=None,
            order_key=index,
            native_component=native_component,
            content=f"content for {entry_id}",
            content_hash=canonical_content_hash(f"content for {entry_id}"),
        )
        for index, entry_id in enumerate(prefix["entries"], start=1)
    )
    branches = build_matched_branches(
        checkpoint,
        load_candidate_registry(REGISTRY_PATH).triplets[0],
        RendererRegistry.native(),
        AdmissionContext(
            writer_event_ids=frozenset(envelope.writer_event_id for envelope in envelopes),
            trial_record_ids=frozenset(
                trial_id for envelope in envelopes for trial_id in envelope.trial_support_ids
            ),
            evidence_envelopes=envelopes,
        ),
    )
    assert isinstance(branches, BranchSet)
    return branches


def _spec() -> RunTemplateSpec:
    return RunTemplateSpec(
        run_template_id="suffix-fh",
        layer="core",
        population_layer="core",
        run_family="readiness",
        analysis_status="primary",
        model_snapshot="gpt-4o-v1",
        evidence_layer="build",
        task_family="mixed-suffix",
        baseline_condition_id="fh_bounded",
        execution_key=MemoryArmExecutionKey(kind="memory_arm", arm="clean"),
        sensitivity_cell_ref={"kind": "base", "cell_id": "base"},
        contamination_type="core",
        horizon=3,
        prefix_template_key_or_none="prefix:fh_bounded",
        candidate_and_control_ids=("candidate-a",),
        corpus_index_filter_versions={"corpus": "clean-v1"},
        prompt_version="prompt-v1",
        tool_contract_hash="tool-v1",
        artifact_hash="suffix-template-hash",
    )


def _suffix() -> tuple[TaskInstance, ...]:
    return tuple(
        TaskInstance(
            sample_id=f"suffix-{index}",
            task_name="game24",
            input={"numbers": [index, 2, 3, 4]},
            metadata={
                "absolute_trial_index": index,
                "decoding": {"temperature": 0},
                "event_time": index,
                "resource_limits": {"tokens": 32},
            },
        )
        for index in (3, 4, 5)
    )


class _Policy:
    def __init__(self, runner) -> None:
        self._runner = runner
        self.calls: list[tuple[str, tuple[str, ...], int]] = []

    def execute(self, task, state, seed: int, trial_id: str):
        del trial_id
        entry_ids = tuple(
            entry.entry_id if isinstance(entry, NativeEntry) else entry for entry in state.entries
        )
        self.calls.append((task.sample_id, entry_ids, seed))
        return self._runner.SuffixStep(state=state)


def _factory(runner):
    policies = {
        arm: _Policy(runner) for arm in ("clean", "correct", "irrelevant", "contam", "filter")
    }
    return runner.SuffixWriterFactory(policies), policies


def test_executes_matched_three_trial_suffix_over_five_arms() -> None:
    from memcontam.experiment.phase12 import suffix_runner

    branches = _branches()
    factory, policies = _factory(suffix_runner)

    result = suffix_runner.run_matched_suffix(branches, _suffix(), _spec(), factory, seed=17)

    assert tuple(run.arm for run in result.runs) == (
        "clean",
        "correct",
        "irrelevant",
        "contam",
        "filter",
    )
    assert {run.pair_id for run in result.runs} == {result.pair_id}
    assert {run.checkpoint_id for run in result.runs} == {
        branches.source_checkpoint.identity.checkpoint_id
    }
    for run in result.runs:
        assert [(trial.absolute_trial_index, trial.event_time) for trial in run.trials] == [
            (3, 3),
            (4, 4),
            (5, 5),
        ]
        assert [trial.branch_id for trial in run.trials] == [run.arm] * 3
        assert len(run.checkpoints) == 3
    assert [call[0] for call in policies["clean"].calls] == ["suffix-3", "suffix-4", "suffix-5"]
    assert all(call[2] == 17 for policy in policies.values() for call in policy.calls)
    filter_entries = {
        entry.entry_id if isinstance(entry, NativeEntry) else entry
        for entry in branches.filter.quarantine.state.entries
    }
    assert filter_entries.isdisjoint(
        {entry_id for _, entries, _ in policies["filter"].calls for entry_id in entries}
    )


def test_rejects_suffix_drift_quarantine_exposure_and_duplicate_nomem() -> None:
    from memcontam.experiment.phase12 import suffix_runner

    branches = _branches()
    factory, _ = _factory(suffix_runner)
    suffix = _suffix()

    drifted = {arm: suffix for arm in ("clean", "correct", "irrelevant", "contam", "filter")}
    drifted["correct"] = tuple(reversed(suffix))
    with pytest.raises(suffix_runner.SuffixExecutionError, match="SUFFIX_TASK_DRIFT"):
        suffix_runner.run_matched_suffix(branches, drifted, _spec(), factory)

    exposed_filter = replace(branches.filter, active=branches.contam.checkpoint)
    object.__setattr__(branches, "filter", exposed_filter)
    with pytest.raises(suffix_runner.SuffixExecutionError, match="QUARANTINE_EXPOSURE"):
        suffix_runner.run_matched_suffix(branches, suffix, _spec(), factory)
