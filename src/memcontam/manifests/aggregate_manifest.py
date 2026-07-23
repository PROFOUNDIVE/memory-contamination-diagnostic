from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from memcontam.experiment.phase12.contracts import canonical_json_hash
from memcontam.manifests.run_manifest import RunManifest, RunManifestRow


class AggregateManifestError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


AggregateStatus = Literal["supported", "not_estimable", "unsupported"]


@dataclass(frozen=True)
class AggregateManifestRow:
    aggregate_id: str
    estimand: str
    population: Mapping[str, str | None]
    evidence_layer: str
    value: float | str
    status: AggregateStatus
    run_ids: tuple[str, ...]
    seed_ids: tuple[int, ...]
    original_weights: Mapping[str, float] | None
    weights: Mapping[str, float] | None
    exclusions: tuple[str, ...]
    metadata_kind: str | None
    run_template_registry_id: str | None
    run_template_registry_hash: str | None
    route_selection_manifest_id: str | None
    route_selection_manifest_hash: str | None
    seed_allocation_manifest_id: str | None
    seed_allocation_manifest_hash: str | None
    exploratory_activation_manifest_id: str | None
    exploratory_activation_manifest_hash: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AggregateManifest:
    rows: tuple[AggregateManifestRow, ...]


def build_aggregate_manifest(
    aggregates: Sequence[AggregateManifestRow | Mapping[str, Any]], run_manifest: RunManifest
) -> AggregateManifest:
    rows = tuple(_build_row(aggregate, run_manifest) for aggregate in aggregates)
    manifest = AggregateManifest(rows)
    validate_aggregate_manifest(manifest, run_manifest)
    return manifest


def validate_aggregate_manifest(manifest: AggregateManifest, run_manifest: RunManifest) -> None:
    _validate_manifest_rows(manifest.rows)
    run_rows = _run_rows(run_manifest)
    for row in manifest.rows:
        _validate_row(row, run_rows)


def write_aggregate_manifest(manifest: AggregateManifest, path: Path | str) -> str:
    _validate_manifest_rows(manifest.rows)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(_canonical_json(row.to_dict()) for row in manifest.rows), encoding="utf-8")
    return canonical_json_hash([row.to_dict() for row in manifest.rows])


def read_aggregate_manifest(path: Path | str) -> AggregateManifest:
    try:
        rows = tuple(
            _row_from_dict(json.loads(line))
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        if isinstance(error, AggregateManifestError):
            raise
        raise AggregateManifestError("AGGREGATE_MANIFEST_SCHEMA_INVALID") from error
    manifest = AggregateManifest(rows)
    _validate_manifest_rows(manifest.rows)
    return manifest


def _build_row(value: AggregateManifestRow | Mapping[str, Any], run_manifest: RunManifest) -> AggregateManifestRow:
    source = value.to_dict() if isinstance(value, AggregateManifestRow) else dict(value)
    run_rows = _run_rows(run_manifest)
    run_ids = _strings(source.get("run_ids", ()), "AGGREGATE_MANIFEST_SCHEMA_INVALID")
    selected_rows = _resolve_run_rows(run_ids, run_rows)
    status = _status(source.get("status", "supported"))
    if status == "supported" and not selected_rows:
        raise AggregateManifestError("AGGREGATE_RUN_ROWS_REQUIRED")
    seed_ids = _seed_ids(source.get("seed_ids", ()), selected_rows)
    _validate_weights(source.get("original_weights"), source.get("weights"), seed_ids)
    governance = _governance(selected_rows)
    return AggregateManifestRow(
        aggregate_id=_required_string(source.get("aggregate_id")),
        estimand=_required_string(source.get("estimand")),
        population=_population(source.get("population")),
        evidence_layer=_required_string(source.get("evidence_layer")),
        value=_value(source.get("value"), status),
        status=status,
        run_ids=run_ids,
        seed_ids=seed_ids,
        original_weights=_weights(source.get("original_weights")),
        weights=_weights(source.get("weights")),
        exclusions=_strings(source.get("exclusions", ()), "AGGREGATE_MANIFEST_SCHEMA_INVALID"),
        **governance,
    )


def _validate_manifest_rows(rows: Sequence[AggregateManifestRow]) -> None:
    aggregate_ids: set[str] = set()
    for row in rows:
        if not row.aggregate_id or row.aggregate_id in aggregate_ids:
            raise AggregateManifestError("DUPLICATE_AGGREGATE_ID")
        aggregate_ids.add(row.aggregate_id)


def _validate_row(row: AggregateManifestRow, run_rows: Mapping[str, RunManifestRow]) -> None:
    selected_rows = _resolve_run_rows(row.run_ids, run_rows)
    if row.status == "supported" and not selected_rows:
        raise AggregateManifestError("AGGREGATE_RUN_ROWS_REQUIRED")
    expected_seeds = tuple(sorted({candidate.trajectory_seed for candidate in selected_rows}))
    if row.seed_ids != expected_seeds:
        raise AggregateManifestError("AGGREGATE_SEED_ROWS_MISMATCH")
    _validate_weights(row.original_weights, row.weights, row.seed_ids)
    expected_governance = _governance(selected_rows)
    if any(getattr(row, name) != value for name, value in expected_governance.items()):
        raise AggregateManifestError("AGGREGATE_COMPATIBILITY_MISMATCH")


def _run_rows(manifest: RunManifest) -> dict[str, RunManifestRow]:
    rows: dict[str, RunManifestRow] = {}
    for row in manifest.rows:
        if row.run_id in rows:
            raise AggregateManifestError("DUPLICATE_RUN_ID")
        rows[row.run_id] = row
    return rows


def _resolve_run_rows(
    run_ids: Sequence[str], run_rows: Mapping[str, RunManifestRow]
) -> tuple[RunManifestRow, ...]:
    if len(set(run_ids)) != len(run_ids):
        raise AggregateManifestError("DUPLICATE_AGGREGATE_RUN_ID")
    try:
        return tuple(run_rows[run_id] for run_id in run_ids)
    except KeyError as error:
        raise AggregateManifestError("AGGREGATE_RUN_ROW_MISSING") from error


def _governance(rows: Sequence[RunManifestRow]) -> dict[str, str | None]:
    names = (
        "metadata_kind",
        "run_template_registry_id",
        "run_template_registry_hash",
        "route_selection_manifest_id",
        "route_selection_manifest_hash",
        "seed_allocation_manifest_id",
        "seed_allocation_manifest_hash",
        "exploratory_activation_manifest_id",
        "exploratory_activation_manifest_hash",
    )
    if not rows:
        return {name: None for name in names}
    values = {name: {getattr(row, name) for row in rows} for name in names}
    if any(len(items) != 1 for items in values.values()):
        raise AggregateManifestError("AGGREGATE_COMPATIBILITY_MISMATCH")
    return {name: next(iter(items)) for name, items in values.items()}


def _required_string(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise AggregateManifestError("AGGREGATE_MANIFEST_SCHEMA_INVALID")
    return value


def _strings(value: Any, code: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or any(not isinstance(item, str) or not item for item in value):
        raise AggregateManifestError(code)
    return tuple(value)


def _population(value: Any) -> Mapping[str, str | None]:
    if not isinstance(value, Mapping) or not value or any(
        not isinstance(key, str) or not isinstance(item, str | type(None)) for key, item in value.items()
    ):
        raise AggregateManifestError("AGGREGATE_MANIFEST_SCHEMA_INVALID")
    return dict(value)


def _status(value: Any) -> AggregateStatus:
    if value not in {"supported", "not_estimable", "unsupported"}:
        raise AggregateManifestError("AGGREGATE_MANIFEST_SCHEMA_INVALID")
    return value


def _value(value: Any, status: AggregateStatus) -> float | str:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise AggregateManifestError("AGGREGATE_MANIFEST_SCHEMA_INVALID")
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise AggregateManifestError("AGGREGATE_MANIFEST_SCHEMA_INVALID")
        return float(value)
    if status == "supported" or not value:
        raise AggregateManifestError("AGGREGATE_MANIFEST_SCHEMA_INVALID")
    return value


def _seed_ids(value: Any, rows: Sequence[RunManifestRow]) -> tuple[int, ...]:
    expected = tuple(sorted({row.trajectory_seed for row in rows}))
    if value in (None, ()):
        return expected
    if not isinstance(value, (list, tuple)) or any(type(seed) is not int for seed in value):
        raise AggregateManifestError("AGGREGATE_MANIFEST_SCHEMA_INVALID")
    seeds = tuple(value)
    if seeds != expected:
        raise AggregateManifestError("AGGREGATE_SEED_ROWS_MISMATCH")
    return seeds


def _weights(value: Any) -> Mapping[str, float] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str)
        or not key
        or isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or not math.isfinite(weight)
        for key, weight in value.items()
    ):
        raise AggregateManifestError("AGGREGATE_MANIFEST_SCHEMA_INVALID")
    return {key: float(weight) for key, weight in value.items()}


def _validate_weights(original: Any, reported: Any, seed_ids: Sequence[int]) -> None:
    original_weights = _weights(original)
    reported_weights = _weights(reported)
    if original_weights != reported_weights:
        raise AggregateManifestError("WEIGHT_RENORMALIZATION_FORBIDDEN")
    if original_weights is not None and set(original_weights) != {str(seed) for seed in seed_ids}:
        raise AggregateManifestError("WEIGHT_RENORMALIZATION_FORBIDDEN")


def _row_from_dict(value: Mapping[str, Any]) -> AggregateManifestRow:
    payload = dict(value)
    for name in ("run_ids", "seed_ids", "exclusions"):
        if isinstance(payload.get(name), list):
            payload[name] = tuple(payload[name])
    return AggregateManifestRow(**payload)


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"


__all__ = [
    "AggregateManifest",
    "AggregateManifestError",
    "AggregateManifestRow",
    "build_aggregate_manifest",
    "read_aggregate_manifest",
    "validate_aggregate_manifest",
    "write_aggregate_manifest",
]
