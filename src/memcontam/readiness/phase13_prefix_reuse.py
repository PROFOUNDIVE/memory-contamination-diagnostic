from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Final

from memcontam.manifests.phase13 import (
    AnalysisWindowBinding, ConformanceCheck, DerivedWindowRow, NotExchangeable,
    NotExchangeableWindow, PrefixDerivationArtifact, SourceEvent,
)
from memcontam.readiness.phase13_calibration_v2_runtime_models import (
    CompletedTrajectory, TrajectoryEvent, TrajectoryRequest,
)
from memcontam.readiness.phase13_prefix_authority import PrefixAuthority, load_prefix_authority


CHECKER_VERSION: Final = "phase13-prefix-checker-v2"
CHECK_IDS: Final = (
    "checkpoint_source_identity", "suffix_order", "execution_contract_identity",
    "native_semantics", "session_randomness", "intervention_identity",
    "future_feedback_cutoff", "source_manifest_identity", "exact_event_range",
    "source_raw_bytes",
)


def _canonical_events(events: tuple[TrajectoryEvent, ...]) -> bytes:
    return b"".join(
        (json.dumps(asdict(event), sort_keys=True, separators=(",", ":")) + "\n").encode()
        for event in events
    )


def _checks(
    request: TrajectoryRequest, source: CompletedTrajectory, authority: PrefixAuthority
) -> tuple[ConformanceCheck, ...]:
    events = source.events
    expected_checkpoints = {row.checkpoint_id for row in authority.checkpoints}
    expected_suffix = authority.ordered_suffix
    identities = authority.identities
    groups = {(event.baseline, event.arm) for event in events}
    predicates = (
        bool(events) and all(event.source_checkpoint_id in expected_checkpoints for event in events),
        all(event.suffix_id == expected_suffix[event.event_time] for event in events)
        and all(
            tuple(event.suffix_id for event in events if (event.baseline, event.arm) == group)
            == expected_suffix for group in groups
        ),
        all(
            (event.task, event.model, event.decoding_contract_id, event.prompt_contract_id,
             event.tool_contract_id, event.parser_contract_id, event.verifier_contract_id,
             event.execution_owner_id)
            == (request.task, identities["model_snapshot_id"], identities["decoding_contract_id"],
                identities["prompt_contract_id"], identities["tool_contract_id"],
                identities["parser_contract_id"], identities["verifier_contract_id"],
                authority.execution_owner_id)
            for event in events
        ),
        all(_state_chain(events, group) for group in groups)
        and all(event.native_semantics_id == identities["native_capacity_registry_id"] for event in events),
        all(event.session_id == request.session_id and event.randomness_contract_id == "provider-managed-no-client-seed-v1" for event in events),
        all(event.intervention_id == request.branches_by_baseline[event.baseline].arms[event.arm].injected_root_id for event in events),
        all(event.future_feedback_cutoff == 0 for event in events),
        source.source_manifest_id == request.stream_id
        and source.source_seal.execution_registry_hash == authority.execution_registry_hash,
        len(events) == 160 and all(
            tuple(event.event_time for event in events if (event.baseline, event.arm) == group)
            == tuple(range(10)) for group in groups
        ),
        hashlib.sha256(_canonical_events(events)).hexdigest() == source.source_raw_sha256
        == source.source_seal.source_raw_sha256,
    )
    return tuple(
        ConformanceCheck(
            check_id=check_id,
            verdict="pass" if passed else "fail",
            evidence_sha256=hashlib.sha256(
                f"{check_id}:{passed}:{authority.execution_registry_hash}:{source.source_raw_sha256}".encode()
            ).hexdigest(),
            checker_version=CHECKER_VERSION,
            source_run_id=request.stream_id,
            source_manifest_id=source.source_manifest_id,
            source_raw_sha256=source.source_raw_sha256,
        )
        for check_id, passed in zip(CHECK_IDS, predicates, strict=True)
    )


def _state_chain(events: tuple[TrajectoryEvent, ...], group: tuple[str, str]) -> bool:
    selected = tuple(event for event in events if (event.baseline, event.arm) == group)
    return all(
        left.state_after_sha256 == right.state_before_sha256
        for left, right in zip(selected, selected[1:], strict=False)
    )


def derive_prefix_windows(
    request: TrajectoryRequest, source: CompletedTrajectory
) -> PrefixDerivationArtifact | NotExchangeable:
    authority = load_prefix_authority(request)
    checks = _checks(request, source, authority)
    if any(check.verdict == "fail" for check in checks):
        return NotExchangeable(
            status="not_exchangeable", checks=checks,
            registered_windows=tuple(
                NotExchangeableWindow(
                    analysis_window_id=window.analysis_window_id,
                    evidence_status=window.evidence_status,
                    multiplicity_status=window.multiplicity_status,
                    realization_disposition="not_exchangeable",
                ) for window in authority.windows
            ),
            derived_artifact=None, provider_calls=0, task_presentations=0, memory_evolutions=0,
        )
    rows = tuple(_row(window, request, source, authority) for window in authority.windows)
    return PrefixDerivationArtifact(
        schema_version="phase13_prefix_derivation_v2",
        conformance_id="phase13-ten-condition-prefix-v1",
        execution_registry_hash=authority.execution_registry_hash,
        source_raw_sha256=source.source_raw_sha256,
        checks=checks,
        rows=rows,
    )


def _row(window, request, source, authority) -> DerivedWindowRow:  # noqa: ANN001, ANN202
    end = window.window_length - 1
    selected = tuple(event for event in source.events if event.event_time <= end)
    return DerivedWindowRow(
        analysis_window=AnalysisWindowBinding(
            analysis_window_id=window.analysis_window_id,
            window_length=window.window_length,
            event_time_start=0,
            event_time_end=end,
            outcome_family=window.outcome_family,
            evidence_status=window.evidence_status,
            multiplicity_status=window.multiplicity_status,
        ),
        source_run_id=request.stream_id,
        source_manifest_id=source.source_manifest_id,
        source_raw_sha256=source.source_raw_sha256,
        source_execution_contract_id=authority.execution_contract_id,
        source_execution_owner_id=authority.execution_owner_id,
        source_ordered_stream_sha256=authority.ordered_stream_sha256,
        event_time_range=(0, end),
        events=tuple(SourceEvent.model_validate(asdict(event)) for event in selected),
        conformance_id="phase13-ten-condition-prefix-v1",
        realization_disposition="prefix_view",
        no_new_provider_execution=True,
        provider_calls=0,
        task_presentations=0,
        memory_evolutions=0,
    )


__all__ = ("CHECKER_VERSION", "derive_prefix_windows")
