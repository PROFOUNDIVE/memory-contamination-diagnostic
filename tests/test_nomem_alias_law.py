from __future__ import annotations

from pathlib import Path

import pytest
from memcontam.contamination.phase12.registry import load_candidate_registry
from memcontam.contamination.phase12.renderers import RendererRegistry
from memcontam.experiment.phase12.contracts import NoMemExecutionKey, RunTemplateSpec
from memcontam.memory.admission import AdmissionContext
from memcontam.memory.checkpoint_v3 import NativeState, Phase12Checkpoint, Phase12CheckpointIdentity
from memcontam.tasks.base import TaskInstance


def _nomem_spec() -> RunTemplateSpec:
    return RunTemplateSpec(
        run_template_id="suffix-nomem",
        layer="core",
        population_layer="core",
        run_family="readiness",
        analysis_status="primary",
        model_snapshot="gpt-4o-v1",
        evidence_layer="build",
        task_family="mixed-suffix",
        baseline_condition_id="nomem",
        execution_key=NoMemExecutionKey(kind="nomem_singleton", key="*"),
        sensitivity_cell_ref={"kind": "base", "cell_id": "base"},
        contamination_type="not_applicable",
        horizon=2,
        prefix_template_key_or_none=None,
        candidate_and_control_ids=(),
        corpus_index_filter_versions={},
        prompt_version="prompt-v1",
        tool_contract_hash="tool-v1",
        artifact_hash="suffix-template-hash",
    )


def _suffix() -> tuple[TaskInstance, ...]:
    return tuple(
        TaskInstance(
            sample_id=f"nomem-{index}",
            task_name="game24",
            input={"numbers": [index, 2, 3, 4]},
            metadata={"absolute_trial_index": index, "event_time": index},
        )
        for index in (1, 2)
    )


class _NoMemPolicy:
    def __init__(self, runner) -> None:
        self._runner = runner
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def execute(self, task, state, seed: int, trial_id: str):
        del seed, trial_id
        self.calls.append((task.sample_id, tuple(state.entries)))
        return self._runner.SuffixStep(state=state)


def test_nomem_executes_once_and_materializes_only_display_aliases() -> None:
    from memcontam.experiment.phase12 import suffix_runner
    from memcontam.experiment.phase12.branching import build_matched_branches

    checkpoint = Phase12Checkpoint(
        identity=Phase12CheckpointIdentity("nomem", "no_memory", "unused"),
        state=NativeState("no_memory", (), {}),
        canonical_bytes=b"",
        canonical_sha256="",
    )
    aliases = build_matched_branches(
        checkpoint,
        load_candidate_registry(
            Path("data/phase12/registries/candidate_registry_v1.json")
        ).triplets[0],
        RendererRegistry.native(),
        AdmissionContext(),
    )
    policy = _NoMemPolicy(suffix_runner)
    factory = suffix_runner.SuffixWriterFactory({"nomem": policy})

    result = suffix_runner.run_matched_suffix(aliases, _suffix(), _nomem_spec(), factory, seed=5)
    materialized = suffix_runner.materialize_nomem_aliases(result)

    assert result.nomem is not None
    assert result.nomem.underlying_execution_count == 1
    assert [call[0] for call in policy.calls] == ["nomem-1", "nomem-2"]
    assert all(entries == () for _, entries in policy.calls)
    assert [(alias.display_arm, alias.execution_key.kind) for alias in materialized] == [
        ("clean", "nomem_singleton"),
        ("correct", "nomem_singleton"),
        ("irrelevant", "nomem_singleton"),
        ("contam", "nomem_singleton"),
        ("filter", "nomem_singleton"),
    ]
    assert len(policy.calls) == 2
    with pytest.raises(suffix_runner.SuffixExecutionError, match="DUPLICATE_NOMEM_EXECUTION"):
        suffix_runner.run_matched_suffix(aliases, _suffix(), _nomem_spec(), factory, seed=5)
