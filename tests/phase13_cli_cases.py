from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from memcontam.evaluation.phase13_observability_registration import registered_verifier_result
from memcontam.experiment import phase13_ordinary_runtime
from memcontam.experiment.phase12.runtime_registry import PHASE13_CORE_BASELINE_REGISTRY
from memcontam.experiment.phase13_snapshot_diagnostic import (
    execute_snapshot_unit,
    snapshot_input_sha256,
)
from memcontam.memory.checkpoint_v3 import NativeEntry, NativeState, serialize_checkpoint
from memcontam.tasks.base import TaskInstance

from .phase13_snapshot_durable_fixtures import ContractTransport
from .test_phase13_snapshot_bot import make_fixture
from .test_phase13_snapshot_dc_rs import _context, _native_embedder, _shared_fixture


def native_case(folder: Path, index: int):
    arm = ("clean", "correct", "irrelevant", "contam")[index % 4]
    task = (TaskInstance(sample_id="qa-meb", task_name="math_equation_balancer", input={"input": "1 ? 1 = 2"},
                        verifier_spec={"target": "1 + 1 = 2", "target_value": 2}) if index % 8 < 4 else
            TaskInstance(sample_id="qa-words", task_name="word_sorting", input={"words": ["pear", "apple"]},
                         verifier_spec={"sorted_words": ["apple", "pear"]}))
    if index < 8:
        run, bindings, _ = make_fixture(folder, _native_embedder())
        assert run.branch is not None
        entries = tuple(replace(entry, render_id=f"synthetic-{arm}") if isinstance(entry, NativeEntry) and arm != "clean" else entry
                        for entry in run.branch.checkpoint.state.entries)
        native = replace(run.branch.checkpoint.state, entries=entries)
        checkpoint = serialize_checkpoint(native, checkpoint_index=0)
        state = PHASE13_CORE_BASELINE_REGISTRY["bot_style"].restore_state(native, None)
        branch = replace(run.branch, arm=arm, state=state, checkpoint=checkpoint, root_count=0 if arm == "clean" else 1,
            injected_root_id=None if arm == "clean" else "fixture-template", candidate_triplet_id=None if arm == "clean" else "synthetic-qa",
            native_render_id=None if arm == "clean" else f"synthetic-{arm}")
        run = replace(run, task_name=task.task_name, tasks=(task,), arm=arm, branch=branch,
            verifier=lambda answer, row: registered_verifier_result(row, answer).is_correct)
    else:
        run, bindings = _shared_fixture(_context(task, arm=arm))
    run = replace(run, run_id=f"qa-observation-{index}", client=ContractTransport(run.client), decoding={"temperature": 0.0, "top_p": 1.0})
    assert run.branch is not None
    if index >= 8:
        native = NativeState.from_mapping(json.loads(run.branch.checkpoint.canonical_bytes.decode().replace("qa:trial:", run.run_id + ":trial:")))
        checkpoint = serialize_checkpoint(native, checkpoint_index=0)
        state = PHASE13_CORE_BASELINE_REGISTRY["dc_rs"].restore_state(native, phase13_ordinary_runtime._context(run, task, 1))
        run = replace(run, branch=replace(run.branch, checkpoint=checkpoint, state=state))
    assert run.branch is not None
    bindings = replace(bindings, state_sha256=run.branch.checkpoint.canonical_sha256, input_sha256=snapshot_input_sha256(run),
        archive_identity=bindings.archive_identity.model_copy(update={
            "checkpoint_registry_sha256": run.branch.checkpoint.canonical_sha256,
            "execution_template_id": f"{task.task_name}:{run.baseline}:{arm}",
            "ordered_sample_ids_sha256": hashlib.sha256(json.dumps((task.sample_id,), separators=(",", ":")).encode()).hexdigest()}))
    return execute_snapshot_unit(run, bindings, folder)
