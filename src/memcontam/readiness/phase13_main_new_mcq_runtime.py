from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Final, Literal, Mapping, assert_never

from memcontam.baselines.bot_phase12 import BoTStateV3
from memcontam.baselines.dynamic_cheatsheet_phase12 import DcRsStateV3, _archive_entry
from memcontam.baselines.full_history_phase12 import FullHistoryStateV3
from memcontam.baselines.reflexion_phase12 import ReflexionStateV3
from memcontam.experiment.phase12.game24_runner import Game24RuntimeContext
from memcontam.experiment.phase12.live_branch import (
    Arm,
    LiveArmBranch,
    LiveBranchEvent,
    LiveThreeArmBranches,
)
from memcontam.experiment.phase12.runtime_registry import RuntimeEntry
from memcontam.memory.cards_v3 import canonical_content_hash
from memcontam.memory.checkpoint_v3 import (
    NATIVE_ENTRY_V1,
    NativeEntry,
    Phase12Checkpoint,
    append_native_entry,
)
from memcontam.memory.serializer_registry import SerializerRegistry
from memcontam.memory.stores import MemoryEntry
from memcontam.readiness.phase13_authority_files import read_regular_nofollow
from memcontam.readiness.phase13_core_bundle import CoreTask
from memcontam.readiness.phase13_new_mcq_candidate_evidence_v2_rendering import (
    RenderInput,
    native_payload,
    render_config,
)
from memcontam.readiness.phase13_new_mcq_phase1_models import BaselineId, NativeKind
from memcontam.readiness.phase13_new_mcq_rag_models import (
    AuthoritySelection,
    InterventionRegistry,
)


TreatmentArm = Literal["correct", "irrelevant", "contam"]
BaselineState = FullHistoryStateV3 | BoTStateV3 | ReflexionStateV3 | DcRsStateV3
_ARMS: Final[tuple[Arm, ...]] = ("clean", "correct", "irrelevant", "contam")
_TREATMENT_ARMS: Final[tuple[TreatmentArm, ...]] = ("correct", "irrelevant", "contam")
_BASELINES: Final[dict[str, tuple[BaselineId, NativeKind]]] = {
    "fh_bounded": ("FH-bounded", "raw_interaction"),
    "bot_style": ("BoT-style", "thought_template"),
    "reflexion_style": ("Reflexion-style", "reflection"),
    "dc_rs": ("DC-RS adapted", "raw_interaction"),
}


class NewMcqRuntimeError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def load_new_mcq_runtime_registry(repository_root: Path) -> InterventionRegistry:
    package = repository_root / "data/phase13/rag/new_mcq"
    authority_raw = read_regular_nofollow(package / "authority_selection_v1.json")
    registry = InterventionRegistry.model_validate_json(
        read_regular_nofollow(package / "intervention_registry_v1.json")
    )
    authority = AuthoritySelection.model_validate_json(authority_raw)
    if (
        hashlib.sha256(authority_raw).hexdigest() != registry.authority_selection_sha256
        or authority.task_selections
        != {task: value.selected_candidate_id for task, value in registry.tasks.items()}
    ):
        raise NewMcqRuntimeError("MAIN_NEW_MCQ_SELECTION_BINDING_INVALID")
    return registry


def new_mcq_native_entries(
    task: CoreTask,
    baseline: str,
    registry: InterventionRegistry,
) -> dict[TreatmentArm, NativeEntry]:
    try:
        baseline_id, native_kind = _BASELINES[baseline]
        treatments = registry.tasks[task]
        schema = SerializerRegistry.native().schema_for(baseline)
    except KeyError as error:
        raise NewMcqRuntimeError("MAIN_NEW_MCQ_RUNTIME_CELL_INVALID") from error
    entries: dict[TreatmentArm, NativeEntry] = {}
    for arm in _TREATMENT_ARMS:
        document = treatments.documents[arm]
        role = "false" if arm == "contam" else arm
        payload = native_payload(
            RenderInput(
                treatments.selected_candidate_id,
                task,
                baseline_id,
                native_kind,
                role,
                document.semantic_id,
                document.text,
                f"main-a::{task}",
            )
        )
        if baseline == "dc_rs" and isinstance(payload, dict):
            payload = {
                **payload,
                "input": payload["query"],
                "raw_output": payload["response"],
            }
        content = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        render = render_config(
            RenderInput(
                treatments.selected_candidate_id,
                task,
                baseline_id,
                native_kind,
                role,
                document.semantic_id,
                document.text,
                f"main-a::{task}",
            )
        )
        render_id = hashlib.sha256(
            json.dumps(
                {"config": render, "native_entry": payload},
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        ).hexdigest()
        entries[arm] = NativeEntry(
            entry_id=f"phase13-new-mcq::{task}::{arm}::{document.semantic_id}",
            semantic_kind=schema.semantic_kind,
            schema_version=NATIVE_ENTRY_V1,
            native_component=schema.native_component,
            content=content,
            content_hash=canonical_content_hash(content),
            render_id=render_id,
        )
    return entries


def build_new_mcq_live_branches(
    *,
    prefix: Phase12Checkpoint,
    context: Game24RuntimeContext,
    task: CoreTask,
    registry: InterventionRegistry,
    runtime_registry: Mapping[str, RuntimeEntry],
) -> LiveThreeArmBranches:
    baseline = prefix.state.baseline
    try:
        runtime = runtime_registry[baseline]
    except KeyError as error:
        raise NewMcqRuntimeError("MAIN_NEW_MCQ_RUNTIME_CELL_INVALID") from error
    clean_state = _baseline_state(runtime.restore_state(prefix.state, context))
    roots = new_mcq_native_entries(task, baseline, registry)
    states: dict[Arm, BaselineState] = {"clean": clean_state}
    checkpoints: dict[Arm, Phase12Checkpoint] = {"clean": prefix}
    for arm, root in roots.items():
        state = deepcopy(clean_state)
        _inject_root(state, root)
        states[arm] = state
        checkpoints[arm] = append_native_entry(prefix, root)
    triplet_id = f"phase13-new-mcq::{task}::{registry.tasks[task].selected_candidate_id}"
    branches: dict[Arm, LiveArmBranch] = {
        arm: LiveArmBranch(
            arm,
            prefix.identity.checkpoint_id,
            checkpoints[arm].identity.checkpoint_id,
            checkpoints[arm],
            states[arm],
            0 if arm == "clean" else 1,
            None if arm == "clean" else roots[arm].entry_id,
            None if arm == "clean" else triplet_id,
            None if arm == "clean" else roots[arm].render_id,
        )
        for arm in _ARMS
    }
    events = tuple(
        LiveBranchEvent(
            "branch_constructed",
            arm,
            prefix.identity.checkpoint_id,
            branches[arm].source_identity,
            branches[arm].injected_root_id,
            None,
            None,
        )
        for arm in _ARMS
    ) + tuple(
        LiveBranchEvent(
            "intervention_applied",
            arm,
            prefix.identity.checkpoint_id,
            branches[arm].source_identity,
            roots[arm].entry_id,
            triplet_id,
            roots[arm].render_id,
        )
        for arm in _TREATMENT_ARMS
    )
    return LiveThreeArmBranches(baseline, context.model, dict(context.decoding), branches, events)


def _baseline_state(state: object) -> BaselineState:
    match state:
        case FullHistoryStateV3() | BoTStateV3() | ReflexionStateV3() | DcRsStateV3():
            return state
        case _:
            raise NewMcqRuntimeError("MAIN_NEW_MCQ_BRANCH_STATE_INVALID") from None


def _inject_root(state: BaselineState, root: NativeEntry) -> None:
    match state:
        case FullHistoryStateV3():
            state.records.append(
                MemoryEntry(
                    entry_id=root.entry_id,
                    content=root.content,
                    memory_type=root.semantic_kind,
                    clean_or_contaminated="injected",
                )
            )
            state.injected_root_id = root.entry_id
        case BoTStateV3():
            state.entries.append(root)
        case ReflexionStateV3():
            state.reflections.append(root)
            state.injected_root_id = root.entry_id
        case DcRsStateV3():
            state.archive.append(_archive_entry(root))
            state.injected_root_id = root.entry_id
        case unreachable:
            assert_never(unreachable)


__all__ = [
    "NewMcqRuntimeError",
    "build_new_mcq_live_branches",
    "load_new_mcq_runtime_registry",
    "new_mcq_native_entries",
]
