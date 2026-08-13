from __future__ import annotations

from typing import Final

from memcontam.contamination.phase12.controls import construct_correct_control, construct_irrelevant_control
from memcontam.contamination.phase12.registry import load_candidate_registry
from memcontam.contamination.phase12.renderers import RendererRegistry
from memcontam.experiment.phase12.live_branch import LiveArmBranch
from memcontam.memory.checkpoint_v3 import NativeEntry, append_native_entry
from memcontam.readiness.phase13_calibration_v2_runtime_models import TrajectoryRequest
from memcontam.readiness.phase13_structural_authority import registered_checkpoints
from memcontam.readiness.phase13_provider_runtime import Phase13V2ProviderRuntime

BASELINES: Final = ("fh_bounded", "rag_frozen", "bot_style", "reflexion_style")
ARMS: Final = ("clean", "correct", "irrelevant", "contam")


class RuntimeValidationError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def validate_trajectory_request(request: TrajectoryRequest) -> None:
    execution = request.verified.execution
    stream = next((row for row in execution.task_streams if row.task == request.task), None)
    suffix = None if stream is None else next(
        (row for row in stream.suffixes if row.seed_id == request.seed_id), None
    )
    if suffix is None or suffix.source_ordered_stream_sha256 != request.source_ordered_stream_sha256:
        raise RuntimeValidationError("SOURCE_STREAM_IDENTITY_INVALID")
    if request.stream_id != f"{request.task}-seed-{request.seed_id}":
        raise RuntimeValidationError("SOURCE_STREAM_IDENTITY_INVALID")
    if tuple(request.branches_by_baseline) != BASELINES:
        raise RuntimeValidationError("BASELINE_PANEL_INVALID")
    registered = {row.baseline: row for row in registered_checkpoints(request.stream_id)}
    for baseline, branches in request.branches_by_baseline.items():
        if branches.baseline != baseline:
            raise RuntimeValidationError("BRANCH_BASELINE_MISMATCH")
        if tuple(branches.arms) != ARMS or "filter" in branches.arms:
            raise RuntimeValidationError("FILTER_BRANCH_FORBIDDEN")
        authority = registered[baseline]
        expected = _expected_checkpoints(request, baseline, branches.arms["clean"].checkpoint)
        for arm, branch in branches.arms.items():
            _validate_branch(
                branch,
                arm,
                authority.checkpoint_id,
                authority.sha256,
                expected[arm],
            )
        _validate_interventions(branches)
    if len(request.contexts) != execution.timing.H_run:
        raise RuntimeValidationError("HORIZON_INVALID")
    if tuple(context.identities.order_key for context in request.contexts) != tuple(range(2, 12)):
        raise RuntimeValidationError("EVENT_RANGE_INVALID")
    expected_suffix = request.verified.ordered_suffixes.get((request.task, request.seed_id))
    if tuple(context.task.sample_id for context in request.contexts) != expected_suffix:
        raise RuntimeValidationError("SUFFIX_TASK_DRIFT")
    _validate_providers(request)


def _validate_branch(
    branch: LiveArmBranch,
    arm: str,
    source_id: str,
    source_hash: str,
    expected_checkpoint,
) -> None:
    checkpoint = branch.checkpoint
    if branch.arm != arm or branch.prefix_identity != source_id:
        raise RuntimeValidationError("BRANCH_LINEAGE_MISMATCH")
    if checkpoint != expected_checkpoint:
        raise RuntimeValidationError("BRANCH_LINEAGE_MISMATCH")
    if checkpoint.canonical_sha256 == source_hash:
        if arm != "clean" or branch.source_identity != source_id or branch.injected_root_id is not None:
            raise RuntimeValidationError("BRANCH_LINEAGE_MISMATCH")
        return
    root = checkpoint.state.entries[-1] if checkpoint.state.entries else None
    if (
        arm == "clean"
        or not isinstance(root, NativeEntry)
        or checkpoint.identity.checkpoint_id != branch.source_identity
        or root.entry_id != branch.injected_root_id
        or root.direct_parent_ids
    ):
        raise RuntimeValidationError("BRANCH_LINEAGE_MISMATCH")


def _expected_checkpoints(request: TrajectoryRequest, baseline: str, source):  # noqa: ANN001, ANN202
    registry = load_candidate_registry(
        request.verified.root / "data/phase12/registries/candidate_registry_v1.json"
    )
    triplet = next(row for row in registry.triplets if row.task == request.task)
    renderers = RendererRegistry.native()
    return {
        "clean": source,
        "correct": append_native_entry(source, construct_correct_control(baseline, triplet, source)),
        "irrelevant": append_native_entry(
            source, construct_irrelevant_control(baseline, triplet, source)
        ),
        "contam": append_native_entry(source, renderers.render_false(baseline, triplet, source)),
    }


def _validate_providers(request: TrajectoryRequest) -> None:
    required = {(baseline, arm) for baseline in BASELINES for arm in ARMS} | {("nomem", "clean")}
    if set(request.providers) != required:
        raise RuntimeValidationError("OWNED_PROVIDER_PANEL_INVALID")
    templates = {
        (row.baseline, row.arm_key): row
        for row in request.verified.execution.execution_templates
        if row.task == request.task
    }
    arm_names = {"clean": "Clean", "correct": "Correct", "irrelevant": "Irrelevant", "contam": "Contam"}
    for (baseline, arm), provider in request.providers.items():
        if not isinstance(provider, Phase13V2ProviderRuntime):
            raise RuntimeValidationError("OWNED_PROVIDER_REQUIRED")
        key = ("nomem", "star_NoMem") if baseline == "nomem" else (baseline, arm_names[arm])
        template = templates[key]
        if (
            provider.execution_template_id != template.template_id
            or provider.execution_owner_id != template.owner_id
        ):
            raise RuntimeValidationError("PROVIDER_TEMPLATE_AUTHORITY_MISMATCH")


def _validate_interventions(branches) -> None:  # noqa: ANN001
    interventions = {
        event.arm: event
        for event in branches.events
        if event.kind == "intervention_applied"
    }
    if set(interventions) != {"correct", "irrelevant", "contam"}:
        raise RuntimeValidationError("INTERVENTION_LINEAGE_MISMATCH")
    triplet_ids = {event.candidate_triplet_id for event in interventions.values()}
    if len(triplet_ids) != 1 or None in triplet_ids:
        raise RuntimeValidationError("INTERVENTION_LINEAGE_MISMATCH")
    for arm, event in interventions.items():
        branch = branches.arms[arm]
        root = branch.checkpoint.state.entries[-1]
        if (
            event.prefix_identity != branch.prefix_identity
            or event.source_identity != branch.source_identity
            or event.injected_root_id != branch.injected_root_id
            or not isinstance(root, NativeEntry)
            or event.native_render_id != root.render_id
        ):
            raise RuntimeValidationError("INTERVENTION_LINEAGE_MISMATCH")
