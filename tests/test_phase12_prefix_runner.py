from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pytest

from memcontam.experiment.phase12.contracts import PrefixExecutionKey, PrefixTemplateSpec
from memcontam.memory.cards_v3 import MEMORY_CARD_V3, MemoryCardEnvelopeV3, canonical_content_hash
from memcontam.memory.checkpoint_v3 import NATIVE_ENTRY_V1, NativeEntry, NativeState


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "phase12" / "FX-BRANCH-001.json"


def _fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _template(
    *,
    baseline_condition_id: str,
    model_snapshot: str = "gpt-4o-v1",
    evidence_layer: Literal["build", "calibration", "main", "extension"] = "build",
    tool_contract_hash: str = "tool-v1",
    corpus_version: str = "clean-corpus-v1",
    capacity_contract_id: str = "capacity-v1",
):
    return PrefixTemplateSpec(
        prefix_template_key=f"prefix:{baseline_condition_id}",
        execution_key=PrefixExecutionKey(kind="branch_free_prefix"),
        model_snapshot=model_snapshot,
        evidence_layer=evidence_layer,
        task_family="mixed-burn-in",
        baseline_condition_id=baseline_condition_id,
        sensitivity_cell_ref={"kind": "base", "cell_id": "base"},
        prompt_version="prompt-v1",
        tool_contract_hash=tool_contract_hash,
        corpus_version=corpus_version,
        capacity_contract_id=capacity_contract_id,
        artifact_hash="template-hash",
    )


class _FixturePolicy:
    def __init__(self, runner, checkpoint: dict) -> None:
        self._runner = runner
        self._checkpoint = checkpoint
        self.calls: list[tuple[str, int, str]] = []

    def initial_state(self, spec, seed: int):
        del spec, seed
        return NativeState(
            baseline=self._checkpoint["baseline"],
            entries=(),
            native_state=self._checkpoint["native_state"],
        )

    def execute(self, task, state, seed: int, trial_id: str):
        self.calls.append((task.task_id, seed, trial_id))
        entry_count = task.absolute_trial_index
        return self._runner.PrefixStep(
            state=NativeState(
                baseline=state.baseline,
                entries=tuple(self._checkpoint["entries"][:entry_count]),
                native_state=self._checkpoint["native_state"],
            )
        )


def _fixture_spec(runner, baseline: str):
    fixture = _fixture()
    return runner.PrefixRunSpec(
        template=_template(baseline_condition_id=baseline),
        tasks=tuple(
            runner.PrefixTask(
                absolute_trial_index=row["absolute_trial_index"],
                task_id=row["task_id"],
                input_value=row["input"],
            )
            for row in fixture["burn_in_task_sequence"]
        ),
    )


def test_runs_one_deterministic_prefix_per_primary_baseline() -> None:
    from memcontam.experiment.phase12 import checkpoint_store, prefix_runner

    fixture = _fixture()
    store = checkpoint_store.CheckpointStore()

    for baseline, prefix in fixture["baseline_prefixes"].items():
        spec = _fixture_spec(prefix_runner, baseline)
        policy = _FixturePolicy(prefix_runner, prefix["checkpoint"])
        ledger = prefix_runner.PrefixEventLedger(store)

        result = prefix_runner.run_clean_prefix(spec, seed=7, policy=policy, writer=ledger)
        repeated = prefix_runner.run_clean_prefix(
            spec,
            seed=7,
            policy=policy,
            writer=prefix_runner.PrefixEventLedger(store),
        )

        assert result is repeated
        assert len(policy.calls) == len(fixture["burn_in_task_sequence"])
        assert result.checkpoint.identity.checkpoint_id == prefix["expected_checkpoint_id"]
        assert result.checkpoint.identity.sha256 == prefix["expected_checkpoint_sha256"]
        assert [event.checkpoint_index for event in result.checkpoint_events] == [1, 2]
        assert result.checkpoint_events[-1].memory_hash == prefix["expected_checkpoint_sha256"]
        assert all(trial.execution_key.kind == "branch_free_prefix" for trial in result.trials)
        assert all("arm" not in trial.execution_key.model_dump() for trial in result.trials)
        assert result.planning_summary()["evidence_layer"] == "build"
        assert result.is_suffix_aggregate_eligible is False
        assert "outcome" not in json.dumps(result.planning_summary(), sort_keys=True)


def test_rejects_prefix_reuse_on_identity_drift() -> None:
    from memcontam.experiment.phase12 import checkpoint_store, prefix_runner

    fixture_prefix = _fixture()["baseline_prefixes"]["fh_bounded"]["checkpoint"]
    store = checkpoint_store.CheckpointStore()
    spec = _fixture_spec(prefix_runner, "fh_bounded")
    policy = _FixturePolicy(prefix_runner, fixture_prefix)
    prefix_runner.run_clean_prefix(
        spec, seed=11, policy=policy, writer=prefix_runner.PrefixEventLedger(store)
    )

    drifted_templates = (
        _template(baseline_condition_id="fh_bounded", model_snapshot="gpt-4o-v2"),
        _template(baseline_condition_id="fh_bounded", tool_contract_hash="tool-v2"),
        _template(baseline_condition_id="fh_bounded", corpus_version="clean-corpus-v2"),
        _template(baseline_condition_id="fh_bounded", capacity_contract_id="capacity-v2"),
        _template(baseline_condition_id="fh_bounded", evidence_layer="calibration"),
    )
    for template in drifted_templates:
        drifted = prefix_runner.PrefixRunSpec(
            template=template,
            tasks=spec.tasks,
        )
        with pytest.raises(prefix_runner.PrefixReuseError, match="PREFIX_IDENTITY_DRIFT"):
            prefix_runner.run_clean_prefix(
                drifted,
                seed=11,
                policy=policy,
                writer=prefix_runner.PrefixEventLedger(store),
            )


class _WritingPolicy:
    def __init__(self, runner) -> None:
        self._runner = runner

    def initial_state(self, spec, seed: int):
        del spec, seed
        return NativeState("fh_bounded", (), {"records": []})

    def execute(self, task, state, seed: int, trial_id: str):
        del seed
        entry = NativeEntry(
            entry_id="fh-prefix-1",
            semantic_kind="full_history_transcript",
            schema_version=NATIVE_ENTRY_V1,
            native_component="history",
            content="2 3 4 5 -> 24",
            content_hash=canonical_content_hash("2 3 4 5 -> 24"),
        )
        envelope = MemoryCardEnvelopeV3(
            entry_id=entry.entry_id,
            baseline="fh_bounded",
            semantic_kind=entry.semantic_kind,
            schema_version=MEMORY_CARD_V3,
            writer_id="fh_appender",
            writer_event_id="fh-native-update-1",
            writer_stage="full_history_generate",
            created_trial_id=trial_id,
            source_trial_ids=(trial_id,),
            source_outcome=None,
            trial_support_ids=(trial_id,),
            memory_support_ids=(),
            direct_parent_ids=(),
            version_predecessor_id=None,
            order_key=task.absolute_trial_index,
            native_component=entry.native_component,
            content=entry.content,
            content_hash=entry.content_hash,
        )
        return self._runner.PrefixStep(
            state=NativeState(state.baseline, (*state.entries, entry), state.native_state),
            writes=(self._runner.PrefixMemoryWrite(entry=entry, envelope=envelope),),
        )


def test_records_native_write_and_admission_ids_on_prefix_trials() -> None:
    from memcontam.experiment.phase12 import checkpoint_store, prefix_runner

    spec = prefix_runner.PrefixRunSpec(
        template=_template(baseline_condition_id="fh_bounded"),
        tasks=(prefix_runner.PrefixTask(1, "game24-001", "2 3 4 5"),),
    )
    ledger = prefix_runner.PrefixEventLedger(checkpoint_store.CheckpointStore())

    result = prefix_runner.run_clean_prefix(
        spec, seed=3, policy=_WritingPolicy(prefix_runner), writer=ledger
    )

    trial = result.trials[0]
    assert trial.memory_event_ids == [ledger.memory_events[0].event_id]
    assert trial.admission_event_ids == [ledger.admission_events[0].event_id]
    assert ledger.memory_events[0].writer_event_id == "fh-native-update-1"
    assert ledger.admission_events[0].decision == "admit"
    assert result.checkpoint.state.entries == (ledger.memory_events[0].entry,)


class _ExternalWriter:
    def __init__(self) -> None:
        self.trials = []
        self.events = []

    def append_trial(self, trial_id, trial) -> None:
        self.trials.append((trial_id, trial))

    def append_event(self, event) -> None:
        self.events.append(event)


def test_accepts_a_v3_style_writer_without_runner_specific_state() -> None:
    from memcontam.experiment.phase12 import prefix_runner

    fixture_prefix = _fixture()["baseline_prefixes"]["fh_bounded"]["checkpoint"]
    writer = _ExternalWriter()
    result = prefix_runner.run_clean_prefix(
        _fixture_spec(prefix_runner, "fh-bounded-external"),
        seed=29,
        policy=_FixturePolicy(prefix_runner, fixture_prefix),
        writer=writer,
    )

    assert [trial_id for trial_id, _ in writer.trials] == [trial.trial_id for trial in writer.events]
    assert result.checkpoint_events == tuple(
        event for event in writer.events if event.record_type == "checkpoint_event"
    )
