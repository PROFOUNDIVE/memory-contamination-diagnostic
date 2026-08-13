from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Final

from memcontam.manifests.phase13 import (
    ConformanceAuthority,
    ConformanceCheck,
    DerivedWindowRow,
    NotExchangeable,
    NotExchangeableWindow,
    PrefixDerivationArtifact,
    SourceEvent,
    SourceTrajectoryManifest,
    load_conformance_authority,
    load_source_manifest,
    read_exact_no_follow,
)


CHECKER_VERSION: Final = "phase13-prefix-checker-v1"
WINDOWS: Final = {
    "accuracy-h2": (2, "prespecified_sensitivity", "descriptive_no_inferential_family"),
    "recurrence-h2": (2, "descriptive", "estimation_only"),
    "accuracy-h5": (5, "confirmatory_primary", "primary_holm_family"),
    "recurrence-h5": (5, "confirmatory_secondary", "estimation_only"),
}
IdentityCheck = Callable[[ConformanceAuthority, SourceTrajectoryManifest], bool]


def _same_fields(*names: str) -> IdentityCheck:
    return lambda authority, source: all(
        getattr(authority, name) == getattr(source, name) for name in names
    )


CHECKS: Final[tuple[tuple[str, IdentityCheck], ...]] = (
    (
        "checkpoint_source_identity",
        _same_fields("source_run_id", "source_checkpoint_id", "source_checkpoint_sha256"),
    ),
    ("suffix_order", _same_fields("source_suffix_id", "source_ordered_stream_sha256")),
    (
        "execution_contract_identity",
        _same_fields(
            "task",
            "model_snapshot_id",
            "decoding_contract_id",
            "prompt_contract_id",
            "tool_contract_id",
            "parser_contract_id",
            "verifier_contract_id",
            "source_execution_contract_id",
            "source_execution_owner_id",
        ),
    ),
    ("native_semantics", _same_fields("native_semantics_id")),
    ("session_randomness", _same_fields("session_contract_id", "randomness_contract_id")),
    ("intervention_identity", _same_fields("intervention_id")),
    ("future_feedback_cutoff", _same_fields("future_feedback_cutoff")),
    (
        "source_manifest_identity",
        lambda authority, source: authority.source_manifest_id == source.source_manifest_id
        and {
            row.analysis_window_id: (
                row.window_length,
                row.evidence_status,
                row.multiplicity_status,
            )
            for row in authority.analysis_windows
        }
        == WINDOWS,
    ),
    ("exact_event_range", lambda _authority, source: source.event_count == 10),
    ("source_raw_bytes", lambda _authority, _source: True),
)


def _evidence_hash(
    check_id: str, authority: ConformanceAuthority, source: SourceTrajectoryManifest, passed: bool
) -> str:
    payload = {
        "check_id": check_id,
        "authority_id": authority.authority_id,
        "source_run_id": source.source_run_id,
        "source_manifest_id": source.source_manifest_id,
        "source_manifest_sha256": authority.source_manifest_sha256,
        "source_raw_sha256": source.source_raw_sha256,
        "verdict": "pass" if passed else "fail",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _events(raw: bytes) -> tuple[SourceEvent, ...]:
    return tuple(SourceEvent.model_validate(json.loads(line)) for line in raw.splitlines() if line)


def _event_identity_matches(
    events: tuple[SourceEvent, ...], source: SourceTrajectoryManifest
) -> bool:
    return all(
        event.source_checkpoint_id == source.source_checkpoint_id
        and event.source_suffix_id == source.source_suffix_id
        and event.task == source.task
        and event.model_snapshot_id == source.model_snapshot_id
        and event.session_contract_id == source.session_contract_id
        and event.intervention_id == source.intervention_id
        for event in events
    )


def _state_chain_matches(events: tuple[SourceEvent, ...]) -> bool:
    return all(
        previous.state_after_sha256 == current.state_before_sha256
        for previous, current in zip(events, events[1:], strict=False)
    )


def derive_prefix_windows(
    authority_path: Path,
    authority_sha256: str,
    source_manifest_path: Path,
    source_manifest_sha256: str,
) -> PrefixDerivationArtifact | NotExchangeable:
    authority = load_conformance_authority(authority_path, authority_sha256)
    source = load_source_manifest(source_manifest_path, source_manifest_sha256)
    raw_hash_matches = False
    raw = b""
    try:
        raw = read_exact_no_follow(Path(source.source_raw_path), source.source_raw_sha256, "SOURCE_RAW")
        raw_hash_matches = True
    except ValueError:
        pass
    try:
        parsed_events = _events(raw) if raw_hash_matches else ()
    except (TypeError, ValueError, json.JSONDecodeError):
        parsed_events = ()
    event_range_matches = (
        source.event_count == 10
        and len(parsed_events) == 10
        and tuple(event.event_index for event in parsed_events) == tuple(range(10))
    )
    checks: list[ConformanceCheck] = []
    for check_id, predicate in CHECKS:
        passed = predicate(authority, source)
        if check_id == "source_manifest_identity":
            passed = passed and authority.source_manifest_sha256 == source_manifest_sha256
        if check_id == "exact_event_range":
            passed = event_range_matches and _event_identity_matches(parsed_events, source)
        if check_id == "native_semantics":
            passed = passed and _state_chain_matches(parsed_events)
        if check_id == "source_raw_bytes":
            passed = raw_hash_matches
        checks.append(
            ConformanceCheck(
                check_id=check_id,
                verdict="pass" if passed else "fail",
                evidence_sha256=_evidence_hash(check_id, authority, source, passed),
                checker_version=CHECKER_VERSION,
                source_run_id=source.source_run_id,
                source_manifest_id=source.source_manifest_id,
                source_manifest_sha256=source_manifest_sha256,
            )
        )
    typed_checks = tuple(checks)
    if any(check.verdict == "fail" for check in typed_checks):
        return NotExchangeable(
            status="not_exchangeable",
            checks=typed_checks,
            registered_windows=tuple(
                NotExchangeableWindow(
                    analysis_window_id=window.analysis_window_id,
                    evidence_status=window.evidence_status,
                    multiplicity_status=window.multiplicity_status,
                    realization_disposition="not_exchangeable",
                )
                for window in authority.analysis_windows
            ),
            derived_artifact=None,
            provider_calls=0,
            task_presentations=0,
            memory_evolutions=0,
        )
    rows = tuple(
        DerivedWindowRow(
            analysis_window_id=window.analysis_window_id,
            conformance_id=authority.conformance_id,
            checker_script_sha256=authority.checker_script_sha256,
            checker_config_sha256=authority.checker_config_sha256,
            repository_commit=authority.repository_commit,
            source_run_id=source.source_run_id,
            source_manifest_id=source.source_manifest_id,
            source_manifest_sha256=source_manifest_sha256,
            source_checkpoint_id=source.source_checkpoint_id,
            source_checkpoint_sha256=source.source_checkpoint_sha256,
            source_suffix_id=source.source_suffix_id,
            source_ordered_stream_sha256=source.source_ordered_stream_sha256,
            source_raw_path=source.source_raw_path,
            source_raw_sha256=source.source_raw_sha256,
            source_execution_contract_id=source.source_execution_contract_id,
            analysis_window=window,
            window_length=window.window_length,
            event_time_range=(0, window.event_time_end),
            events=parsed_events[: window.window_length],
            evidence_status=window.evidence_status,
            multiplicity_status=window.multiplicity_status,
            realization_disposition="prefix_view",
            no_new_provider_execution=True,
            provider_calls=0,
            task_presentations=0,
            memory_evolutions=0,
        )
        for window in authority.analysis_windows
    )
    return PrefixDerivationArtifact(
        schema_version="phase13_prefix_derivation_v1",
        conformance_id=authority.conformance_id,
        checker_script_sha256=authority.checker_script_sha256,
        checker_config_sha256=authority.checker_config_sha256,
        repository_commit=authority.repository_commit,
        checks=typed_checks,
        rows=rows,
    )


__all__ = ("CHECKER_VERSION", "derive_prefix_windows")
