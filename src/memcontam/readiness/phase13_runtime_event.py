from __future__ import annotations

from typing import Literal

from memcontam.experiment.phase12.game24_runner import Game24RuntimeContext
from memcontam.experiment.phase12.live_branch import Arm
from memcontam.readiness.phase13_calibration_v2_runtime_models import (
    TrajectoryEvent,
    TrajectoryRequest,
)


def build_trajectory_event(
    request: TrajectoryRequest,
    context: Game24RuntimeContext,
    baseline: str,
    arm: Arm,
    state_hashes: tuple[str, str],
    outcome: tuple[Literal["succeeded", "failed"], Literal[0, 1]],
) -> TrajectoryEvent:
    branch = request.branches_by_baseline[baseline].arms[arm]
    absolute = int(context.identities.order_key)
    return TrajectoryEvent(
        absolute - 2, absolute, baseline, arm, branch.prefix_identity,
        branch.checkpoint.identity.checkpoint_id, context.task.sample_id, request.task,
        context.model,
        request.verified.execution.identities.decoding_contract_id,
        request.verified.execution.identities.prompt_contract_id,
        request.verified.execution.identities.tool_contract_id,
        request.verified.execution.identities.parser_contract_id,
        request.verified.execution.identities.verifier_contract_id,
        request.verified.execution.identities.native_capacity_registry_id,
        request.session_id, "provider-managed-no-client-seed-v1", 0,
        branch.injected_root_id, request.verified.execution.execution_owner_id,
        *outcome, *state_hashes,
    )


__all__ = ("build_trajectory_event",)
