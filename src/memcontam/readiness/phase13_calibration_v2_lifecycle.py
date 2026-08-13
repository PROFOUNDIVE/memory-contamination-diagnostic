from __future__ import annotations

from dataclasses import asdict, replace
from datetime import UTC, datetime
from typing import Callable, Final, Literal

from memcontam.readiness.phase13_calibration_v2 import prepare_calibration_v2, validate_calibration_v2
from memcontam.readiness.phase13_calibration_v2_authorization import (
    CalibrationV2AuthorizationError,
    VerifiedCalibrationV2Authorization,
    verify_calibration_v2_authorization,
)
from memcontam.readiness.phase13_support_planning import clopper_pearson_lower
from memcontam.readiness.phase13_terminal import render_terminal

from .phase13_calibration_v2_lifecycle_models import (
    AuthorizedLifecycleEvidence,
    LifecycleInvocation,
    LifecycleReport,
    ResourceValues,
    SourceTrajectoryEvidence,
)
from .phase13_calibration_v2_lifecycle_report import lifecycle_identities, seal_lifecycle_report


TASKS: Final = ("game24", "math_equation_balancer", "word_sorting")
SEEDS: Final = tuple(range(10000, 10012))
PROVIDER_OWNER: Final = "phase13-h10-execution-owner-v1"
OFFLINE_OWNER: Final = "phase13-offline-compute-owner-v1"
CAPACITY_FIELDS: Final = (
    "maximum_cost_microusd",
    "per_request_timeout_seconds",
    "maximum_latency_milliseconds",
    "maximum_storage_bytes",
    "maximum_wall_clock_seconds",
    "provider_requests_per_minute",
    "provider_concurrency",
)
ZERO_RESOURCES: Final = ResourceValues(0, 0, 0, 0, 0, 0, 0, 0)


def run_calibration_v2_lifecycle(
    invocation: LifecycleInvocation,
    executor_factory: Callable[[], AuthorizedLifecycleEvidence],
) -> LifecycleReport:
    deterministic = {
        "validate": render_terminal(validate_calibration_v2(invocation.config_path)),
        "prepare": "AWAITING_CALIBRATION_V2_AUTHORIZATION",
    }
    prepare_calibration_v2(invocation.config_path)
    identities = lifecycle_identities(
        invocation.config_path, invocation.request_path, invocation.authorization_path
    )
    verified = _authorize(invocation)
    if verified is None:
        unavailable = _unavailable(invocation)
        report = _base_report(
            invocation, deterministic, identities,
            terminal="CALIBRATION_V2_EXTERNAL_BLOCK",
            reason="EXTERNAL_PREREQUISITES_UNAVAILABLE",
            unavailable_fields=unavailable,
        )
        return seal_lifecycle_report(invocation.report_path, report)
    evidence = executor_factory()
    reason = _invalid_reason(evidence, verified)
    if reason is not None:
        report = replace(
            _base_report(
                invocation, deterministic, identities,
                terminal="CALIBRATION_V2_INVALIDATED", reason=reason,
            ),
            synthetic_qa_only=True,
            provider_construction_count=1,
        )
        return seal_lifecycle_report(invocation.report_path, report)
    report = _completed_report(invocation, deterministic, identities, evidence, verified)
    return seal_lifecycle_report(invocation.report_path, report)


def _authorize(invocation: LifecycleInvocation) -> VerifiedCalibrationV2Authorization | None:
    if (
        invocation.request_path is None
        or invocation.authorization_path is None
        or invocation.expected_authorization_sha256 is None
    ):
        return None
    try:
        return verify_calibration_v2_authorization(
            config_path=invocation.config_path,
            request_path=invocation.request_path,
            authorization_path=invocation.authorization_path,
            expected_authorization_sha256=invocation.expected_authorization_sha256,
            allow_live_calls=invocation.allow_live_calls,
            environment=invocation.environment,
            now=invocation.now,
        )
    except CalibrationV2AuthorizationError:
        return None


def _unavailable(invocation: LifecycleInvocation) -> tuple[str, ...]:
    unavailable = []
    if invocation.request_path is None:
        unavailable.append("request_path")
    if invocation.authorization_path is None:
        unavailable.append("authorization_path")
    if invocation.expected_authorization_sha256 is None:
        unavailable.append("authorization_sha256")
    if not invocation.environment.get("OPENAI_API_KEY"):
        unavailable.append("credential:OPENAI_API_KEY")
    if not invocation.environment.get("MEMCONTAM_BGE_CACHE_DIR"):
        unavailable.append("cache_runtime_access")
    unavailable.extend(f"operator_capacity:{field}" for field in CAPACITY_FIELDS)
    return tuple(unavailable)


def _invalid_reason(
    evidence: AuthorizedLifecycleEvidence,
    verified: VerifiedCalibrationV2Authorization,
) -> str | None:
    expected = tuple((task, seed) for task in TASKS for seed in SEEDS)
    observed = tuple((row.task, row.seed_id) for row in evidence.sources)
    if len(evidence.sources) != 36 or len(set(row.source_run_id for row in evidence.sources)) != 36:
        return "TRAJECTORY_INVENTORY_INVALID"
    if observed != expected:
        return "TRAJECTORY_SEED_INVENTORY_INVALID"
    if any(row.event_times != tuple(range(10)) for row in evidence.sources):
        return "SOURCE_H10_INCOMPLETE"
    if any(
        row.source_execution_count != 1
        or len(row.state_lineage) != 10
        or any(
            left[1] != right[0]
            for left, right in zip(row.state_lineage, row.state_lineage[1:], strict=False)
        )
        for row in evidence.sources
    ):
        return "SOURCE_EXECUTION_MULTIPLICITY_INVALID"
    call_ids = tuple(call_id for row in evidence.sources for call_id in row.provider_call_ids)
    settled_ids = tuple(call_id for row in evidence.sources for call_id in row.settled_call_ids)
    attempts = tuple(attempt for row in evidence.sources for attempt in row.transport_attempt_ids)
    if (
        any(row.accounting_status != "closed_complete" for row in evidence.sources)
        or call_ids != settled_ids
        or len(call_ids) != len(set(call_ids))
        or len(attempts) != len(set(attempts))
        or len(attempts) < len(call_ids)
    ):
        return "OWNER_SETTLEMENT_INCOMPLETE"
    if any(
        row.provider_owner_id != PROVIDER_OWNER or row.offline_owner_id != OFFLINE_OWNER
        for row in evidence.sources
    ):
        return "OWNER_RECONCILIATION_INVALID"
    if any(row.short_window_provider_calls or row.derived_window_executions for row in evidence.sources):
        return "SHORT_WINDOW_EXECUTION_FORBIDDEN"
    if any(not row.archive_valid or row.claim_status != "synthetic_qa_only" for row in evidence.sources):
        return "ARCHIVE_INVALID"
    totals = (
        len(settled_ids),
        len(attempts),
        evidence.observed_input_tokens,
        evidence.observed_output_tokens,
        evidence.observed_cost_microusd,
        evidence.observed_latency_milliseconds,
        evidence.observed_storage_bytes,
        evidence.observed_wall_clock_seconds,
    )
    ceilings = _capacity(verified)
    if any(value > ceiling for value, ceiling in zip(totals, asdict(ceilings).values(), strict=True)):
        return "CAPACITY_OVERRUN"
    if evidence.support_successes != tuple((task, 12) for task in TASKS):
        return "SUPPORT_RECONCILIATION_INVALID"
    if evidence.attempted_seeds != tuple((task, 13) for task in TASKS):
        return "ATTEMPTED_SEED_RECONCILIATION_INVALID"
    return None


def _completed_report(
    invocation: LifecycleInvocation,
    deterministic: dict[str, str],
    identities: dict[str, str | None],
    evidence: AuthorizedLifecycleEvidence,
    verified: VerifiedCalibrationV2Authorization,
) -> LifecycleReport:
    semantic_calls = sum(len(row.settled_call_ids) for row in evidence.sources)
    transports = sum(len(row.transport_attempt_ids) for row in evidence.sources)
    observed = ResourceValues(
        semantic_calls, transports, evidence.observed_input_tokens, evidence.observed_output_tokens,
        evidence.observed_cost_microusd, evidence.observed_latency_milliseconds,
        evidence.observed_storage_bytes, evidence.observed_wall_clock_seconds,
    )
    return replace(
        _base_report(
            invocation, deterministic, identities,
            terminal="CALIBRATION_V2_COMPLETED", reason=None,
        ),
        synthetic_qa_only=True,
        provider_construction_count=1,
        provider_dispatch_count=transports,
        run_root=verified.request.output_root,
        archive_status="valid",
        claim_status="synthetic_qa_only",
        trajectory_count=36,
        tasks=TASKS,
        seeds=SEEDS,
        events_per_source=10,
        source_execution_count=36,
        settled_semantic_calls=semantic_calls,
        settled_transport_attempts=transports,
        provider_owner_id=PROVIDER_OWNER,
        offline_owner_id=OFFLINE_OWNER,
        archive_valid=True,
        support_cp95=tuple((task, str(clopper_pearson_lower(12, 12))) for task in TASKS),
        attempted_seeds=evidence.attempted_seeds,
        observed_resources=observed,
        capacity_ceilings=_capacity(verified),
    )


def _capacity(verified: VerifiedCalibrationV2Authorization) -> ResourceValues:
    row = verified.request.capacity
    return ResourceValues(
        row.maximum_semantic_calls, row.maximum_transport_attempts,
        row.maximum_input_tokens, row.maximum_output_tokens, row.maximum_cost_microusd,
        row.maximum_latency_milliseconds, row.maximum_storage_bytes, row.maximum_wall_clock_seconds,
    )


def _base_report(
    invocation: LifecycleInvocation,
    deterministic: dict[str, str],
    identities: dict[str, str | None],
    *,
    terminal: Literal[
        "CALIBRATION_V2_EXTERNAL_BLOCK",
        "CALIBRATION_V2_INVALIDATED",
        "CALIBRATION_V2_COMPLETED",
    ],
    reason: str | None,
    unavailable_fields: tuple[str, ...] = (),
) -> LifecycleReport:
    timestamp = (invocation.now or datetime.now(UTC)).astimezone(UTC).isoformat()
    return LifecycleReport(
        "phase13_calibration_v2_lifecycle_report_v1", terminal, reason,
        "MAIN_A_EXECUTION_FORBIDDEN", False, unavailable_fields, deterministic, identities,
        0, 0, None, "absent", "absent", 0, (), (), 0, 0, 0, 0, 0, 0,
        None, None, False, (), (), ZERO_RESOURCES, ZERO_RESOURCES, timestamp,
        identities["implementation_commit"] or "unavailable", "",
    )


__all__ = (
    "AuthorizedLifecycleEvidence",
    "LifecycleInvocation",
    "SourceTrajectoryEvidence",
    "run_calibration_v2_lifecycle",
)
