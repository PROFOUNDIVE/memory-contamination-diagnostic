from __future__ import annotations

import math
from statistics import mean

from memcontam.manifests.phase13_archive_authority import ArchiveProjection
from memcontam.manifests.phase13_archive_models import Phase13Archive, SourceAttemptRow


class ReconstructionError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def reconstruct_archive(
    archive: Phase13Archive,
    projection: ArchiveProjection,
    sources: dict[str, SourceAttemptRow],
) -> int:
    scores = _validate_windows(archive, projection, sources)
    _validate_ledgers(archive, projection, sources)
    _validate_aggregates(archive, projection, scores)
    _validate_historical(archive, projection)
    return 4 + sum(len(row.events) for row in sources.values()) + len(archive.provider_ledger) + len(
        archive.offline_ledger
    ) + len(archive.derived_windows) + len(archive.aggregates) + len(archive.claims)


def _validate_windows(
    archive: Phase13Archive,
    projection: ArchiveProjection,
    sources: dict[str, SourceAttemptRow],
) -> dict[str, tuple[int, ...]]:
    registered = {row.analysis_window_id: row for row in projection.windows if row.window_length in (2, 5)}
    expected_ids = set(registered)
    observed: set[str] = set()
    scores: dict[str, tuple[int, ...]] = {}
    for row in archive.derived_windows:
        if row.window_id in observed:
            raise ReconstructionError("DUPLICATE_DERIVED_WINDOW_ID")
        observed.add(row.window_id)
        window = registered.get(row.analysis_window_id)
        if window is None:
            raise ReconstructionError("DERIVED_WINDOW_UNREGISTERED")
        source = sources.get(row.source_run_id)
        if source is None:
            raise ReconstructionError("DERIVED_SOURCE_RUN_MISSING")
        end = window.window_length - 1
        selected = source.events[: window.window_length]
        if row.window_length != window.window_length or row.source_event_range != (0, end):
            raise ReconstructionError("DERIVED_EVENT_RANGE_MISMATCH")
        if row.event_ids != tuple(event.event_id for event in selected):
            raise ReconstructionError("DERIVED_EVENT_RANGE_MISMATCH")
        if row.source_raw_sha256 != source.source_raw_sha256:
            raise ReconstructionError("DERIVED_SOURCE_HASH_MISMATCH")
        if (row.evidence_status, row.multiplicity_status) != (
            window.evidence_status,
            window.multiplicity_status,
        ):
            raise ReconstructionError("WINDOW_STATUS_MISMATCH")
        if row.owner_id != projection.analysis.offline_compute.owner_id or row.provider_calls != 0:
            raise ReconstructionError("OFFLINE_OWNER_MISMATCH")
        scores[row.window_id] = tuple(event.verified_score for event in selected)
    if {row.analysis_window_id for row in archive.derived_windows} != expected_ids:
        raise ReconstructionError("DERIVED_WINDOW_INVENTORY_MISMATCH")
    return scores


def _validate_ledgers(
    archive: Phase13Archive,
    projection: ArchiveProjection,
    sources: dict[str, SourceAttemptRow],
) -> None:
    event_calls = [event.semantic_call_id for source in sources.values() for event in source.events]
    ledger_calls = [row.semantic_call_id for row in archive.provider_ledger]
    if len(event_calls) != len(set(event_calls)) or len(ledger_calls) != len(set(ledger_calls)):
        raise ReconstructionError("DUPLICATE_SEMANTIC_CALL_ID")
    if event_calls != ledger_calls:
        raise ReconstructionError("PROVIDER_LEDGER_MISMATCH")
    attempts = [attempt for row in archive.provider_ledger for attempt in row.transport_attempt_ids]
    if len(attempts) != len(set(attempts)) or any(not row.transport_attempt_ids for row in archive.provider_ledger):
        raise ReconstructionError("PROVIDER_LEDGER_MISMATCH")
    if any(row.execution_owner_id != projection.execution.execution_owner_id for row in archive.provider_ledger):
        raise ReconstructionError("PROVIDER_OWNER_MISMATCH")
    expected = tuple((row.operation, row.owner_id) for row in projection.analysis.offline_compute.rows)
    observed = tuple((row.operation, row.owner_id) for row in archive.offline_ledger)
    if observed != expected or len(set(observed)) != len(observed):
        raise ReconstructionError("OFFLINE_LEDGER_MISMATCH")
    if any(row.provider_calls != 0 or row.cost_microusd != 0 for row in archive.offline_ledger):
        raise ReconstructionError("OFFLINE_PROVIDER_WORK_FORBIDDEN")


def _validate_aggregates(
    archive: Phase13Archive,
    projection: ArchiveProjection,
    scores: dict[str, tuple[int, ...]],
) -> None:
    aggregates = {}
    for row in archive.aggregates:
        if row.aggregate_id in aggregates:
            raise ReconstructionError("DUPLICATE_AGGREGATE_ID")
        if row.original_weights != row.weights:
            raise ReconstructionError("WEIGHT_RENORMALIZATION_FORBIDDEN")
        if any(value <= 0 for value in row.weights.values()):
            raise ReconstructionError("AGGREGATE_WEIGHT_INVALID")
        if not math.isclose(sum(row.weights.values()), 1.0):
            raise ReconstructionError("AGGREGATE_WEIGHT_INVALID")
        task = next(iter(row.weights)).split("-seed-", maxsplit=1)[0]
        if row.family_id != projection.primary_families.get(task):
            raise ReconstructionError("AGGREGATE_FAMILY_MISMATCH")
        source_runs = {
            window.source_run_id
            for window in archive.derived_windows
            if window.window_id in row.source_ids
        }
        if row.status != "ESTIMABLE" or set(row.weights) != source_runs:
            raise ReconstructionError("AGGREGATE_STATUS_PROMOTION_FORBIDDEN")
        try:
            estimate = mean(mean(scores[source_id]) for source_id in row.source_ids)
        except KeyError as error:
            raise ReconstructionError("AGGREGATE_SOURCE_MISSING") from error
        if not isinstance(row.estimate, float) or not math.isclose(row.estimate, estimate):
            raise ReconstructionError("AGGREGATE_RECONSTRUCTION_MISMATCH")
        aggregates[row.aggregate_id] = row
    claim_ids: set[str] = set()
    for claim in archive.claims:
        if claim.claim_id in claim_ids:
            raise ReconstructionError("DUPLICATE_CLAIM_ID")
        claim_ids.add(claim.claim_id)
        aggregate = aggregates.get(claim.aggregate_id)
        if aggregate is None or claim.status != "supported":
            raise ReconstructionError("CLAIM_RECONSTRUCTION_MISMATCH")
        if claim.family_id != aggregate.family_id or claim.estimate != aggregate.estimate:
            raise ReconstructionError("CLAIM_RECONSTRUCTION_MISMATCH")


def _validate_historical(archive: Phase13Archive, projection: ArchiveProjection) -> None:
    reference = archive.historical_reference
    if (
        reference.run_id != projection.historical_run_id
        or reference.availability != projection.historical_availability
        or reference.imported
    ):
        raise ReconstructionError("HISTORICAL_REFERENCE_INVALID")


__all__ = ("ReconstructionError", "reconstruct_archive")
