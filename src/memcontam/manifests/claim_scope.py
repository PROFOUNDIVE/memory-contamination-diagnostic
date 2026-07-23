from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from memcontam.experiment.phase12.contracts import canonical_json_hash
from memcontam.manifests.aggregate_manifest import AggregateManifest, AggregateManifestRow


class ClaimScopeError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


ClaimStatus = Literal["supported", "nonclaim", "unsupported"]


@dataclass(frozen=True)
class ClaimScopeRow:
    claim_id: str
    aggregate_ids: tuple[str, ...]
    estimand: str
    population: Mapping[str, str | None]
    evidence_layer: str
    exclusions: tuple[str, ...]
    prohibited_extrapolations: tuple[str, ...]
    status: ClaimStatus
    scope: str | None
    original_weights: Mapping[str, float] | None
    weights: Mapping[str, float] | None
    route_selection_manifest_id: str | None
    route_selection_manifest_hash: str | None
    seed_allocation_manifest_id: str | None
    seed_allocation_manifest_hash: str | None
    exploratory_activation_manifest_id: str | None
    exploratory_activation_manifest_hash: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ClaimScopeLedger:
    rows: tuple[ClaimScopeRow, ...]


def build_claim_scope(
    claims: Sequence[ClaimScopeRow | Mapping[str, Any]], aggregate_manifest: AggregateManifest
) -> ClaimScopeLedger:
    aggregates = _aggregate_rows(aggregate_manifest)
    rows = tuple(_build_row(claim, aggregates) for claim in claims)
    ledger = ClaimScopeLedger(rows)
    validate_claim_scope(ledger, aggregate_manifest)
    return ledger


def validate_claim_scope(ledger: ClaimScopeLedger, aggregate_manifest: AggregateManifest) -> None:
    aggregates = _aggregate_rows(aggregate_manifest)
    claim_ids: set[str] = set()
    for row in ledger.rows:
        if not row.claim_id or row.claim_id in claim_ids:
            raise ClaimScopeError("DUPLICATE_CLAIM_ID")
        claim_ids.add(row.claim_id)
        _validate_row(row, aggregates)


def write_claim_scope(ledger: ClaimScopeLedger, path: Path | str) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(_canonical_json(row.to_dict()) for row in ledger.rows), encoding="utf-8"
    )
    return canonical_json_hash([row.to_dict() for row in ledger.rows])


def read_claim_scope(path: Path | str) -> ClaimScopeLedger:
    try:
        rows = tuple(
            _row_from_dict(json.loads(line))
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        if isinstance(error, ClaimScopeError):
            raise
        raise ClaimScopeError("CLAIM_SCOPE_SCHEMA_INVALID") from error
    return ClaimScopeLedger(rows)


def _build_row(
    value: ClaimScopeRow | Mapping[str, Any], aggregates: Mapping[str, AggregateManifestRow]
) -> ClaimScopeRow:
    source = value.to_dict() if isinstance(value, ClaimScopeRow) else dict(value)
    aggregate_ids = _strings(source.get("aggregate_ids", ()), "UNSUPPORTED_CLAIM")
    linked = _linked_aggregates(aggregate_ids, aggregates)
    row = ClaimScopeRow(
        claim_id=_required_string(source.get("claim_id")),
        aggregate_ids=aggregate_ids,
        estimand=_required_string(source.get("estimand")),
        population=_population(source.get("population")),
        evidence_layer=_required_string(source.get("evidence_layer")),
        exclusions=_strings(source.get("exclusions", ()), "UNSUPPORTED_CLAIM"),
        prohibited_extrapolations=_strings(
            source.get("prohibited_extrapolations", source.get("prohibited", ())),
            "UNSUPPORTED_CLAIM",
        ),
        status=_status(source.get("status", "supported")),
        scope=_optional_string(source.get("scope")),
        original_weights=_weights(source.get("original_weights")),
        weights=_weights(source.get("weights")),
        route_selection_manifest_id=_optional_string(source.get("route_selection_manifest_id")),
        route_selection_manifest_hash=_optional_string(source.get("route_selection_manifest_hash")),
        seed_allocation_manifest_id=_optional_string(source.get("seed_allocation_manifest_id")),
        seed_allocation_manifest_hash=_optional_string(source.get("seed_allocation_manifest_hash")),
        exploratory_activation_manifest_id=_optional_string(
            source.get("exploratory_activation_manifest_id")
        ),
        exploratory_activation_manifest_hash=_optional_string(
            source.get("exploratory_activation_manifest_hash")
        ),
    )
    _validate_row(row, aggregates, linked)
    return row


def _validate_row(
    row: ClaimScopeRow,
    aggregates: Mapping[str, AggregateManifestRow],
    linked: Sequence[AggregateManifestRow] | None = None,
) -> None:
    linked = _linked_aggregates(row.aggregate_ids, aggregates) if linked is None else linked
    if (
        any(
            aggregate.estimand != row.estimand
            or dict(aggregate.population) != dict(row.population)
            or aggregate.evidence_layer != row.evidence_layer
            for aggregate in linked
        )
        or not row.prohibited_extrapolations
    ):
        raise ClaimScopeError("UNSUPPORTED_CLAIM")
    _validate_weights(row.original_weights, row.weights)
    if row.status == "supported" and any(aggregate.status != "supported" for aggregate in linked):
        raise ClaimScopeError("UNSUPPORTED_CLAIM")
    if any(set(aggregate.exclusions) - set(row.exclusions) for aggregate in linked):
        raise ClaimScopeError("UNSUPPORTED_CLAIM")
    _validate_route_bundle(row, linked)
    _validate_activation(row, linked)


def _aggregate_rows(manifest: AggregateManifest) -> dict[str, AggregateManifestRow]:
    rows: dict[str, AggregateManifestRow] = {}
    for row in manifest.rows:
        if row.aggregate_id in rows:
            raise ClaimScopeError("DUPLICATE_AGGREGATE_ID")
        rows[row.aggregate_id] = row
    return rows


def _linked_aggregates(
    aggregate_ids: Sequence[str], aggregates: Mapping[str, AggregateManifestRow]
) -> tuple[AggregateManifestRow, ...]:
    if not aggregate_ids or len(set(aggregate_ids)) != len(aggregate_ids):
        raise ClaimScopeError("UNSUPPORTED_CLAIM")
    try:
        return tuple(aggregates[aggregate_id] for aggregate_id in aggregate_ids)
    except KeyError as error:
        raise ClaimScopeError("UNSUPPORTED_CLAIM") from error


def _validate_route_bundle(row: ClaimScopeRow, aggregates: Sequence[AggregateManifestRow]) -> None:
    names = (
        "route_selection_manifest_id",
        "route_selection_manifest_hash",
        "seed_allocation_manifest_id",
        "seed_allocation_manifest_hash",
    )
    expected = {name: {getattr(aggregate, name) for aggregate in aggregates} for name in names}
    if any(len(values) != 1 for values in expected.values()):
        raise ClaimScopeError("UNSUPPORTED_CLAIM")
    values = {name: next(iter(items)) for name, items in expected.items()}
    if values["route_selection_manifest_id"] is None:
        if any(getattr(row, name) is not None for name in names):
            raise ClaimScopeError("UNSUPPORTED_CLAIM")
        return
    if row.route_selection_manifest_id is None:
        raise ClaimScopeError("ROUTE_SELECTION_REQUIRED")
    if row.seed_allocation_manifest_id is None:
        raise ClaimScopeError("SEED_ALLOCATION_REQUIRED")
    if any(getattr(row, name) != value for name, value in values.items()):
        raise ClaimScopeError("UNSUPPORTED_CLAIM")


def _validate_activation(row: ClaimScopeRow, aggregates: Sequence[AggregateManifestRow]) -> None:
    names = ("exploratory_activation_manifest_id", "exploratory_activation_manifest_hash")
    expected = {name: {getattr(aggregate, name) for aggregate in aggregates} for name in names}
    if any(len(values) != 1 for values in expected.values()):
        raise ClaimScopeError("UNSUPPORTED_CLAIM")
    values = {name: next(iter(items)) for name, items in expected.items()}
    if values["exploratory_activation_manifest_id"] is None:
        if any(getattr(row, name) is not None for name in names):
            raise ClaimScopeError("UNSUPPORTED_CLAIM")
        return
    if row.exploratory_activation_manifest_id is None:
        raise ClaimScopeError("EXPLORATORY_ACTIVATION_REQUIRED")
    if any(getattr(row, name) != value for name, value in values.items()):
        raise ClaimScopeError("UNSUPPORTED_CLAIM")


def _required_string(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ClaimScopeError("UNSUPPORTED_CLAIM")
    return value


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ClaimScopeError("UNSUPPORTED_CLAIM")
    return value


def _strings(value: Any, code: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ClaimScopeError(code)
    return tuple(value)


def _population(value: Any) -> Mapping[str, str | None]:
    if (
        not isinstance(value, Mapping)
        or not value
        or any(
            not isinstance(key, str) or not isinstance(item, str | type(None))
            for key, item in value.items()
        )
    ):
        raise ClaimScopeError("UNSUPPORTED_CLAIM")
    return dict(value)


def _status(value: Any) -> ClaimStatus:
    if value not in {"supported", "nonclaim", "unsupported"}:
        raise ClaimScopeError("UNSUPPORTED_CLAIM")
    return value


def _weights(value: Any) -> Mapping[str, float] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str)
        or not key
        or isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        for key, weight in value.items()
    ):
        raise ClaimScopeError("UNSUPPORTED_CLAIM")
    return {key: float(weight) for key, weight in value.items()}


def _validate_weights(
    original: Mapping[str, float] | None, reported: Mapping[str, float] | None
) -> None:
    if original != reported:
        raise ClaimScopeError("WEIGHT_RENORMALIZATION_FORBIDDEN")


def _row_from_dict(value: Mapping[str, Any]) -> ClaimScopeRow:
    payload = dict(value)
    for name in ("aggregate_ids", "exclusions", "prohibited_extrapolations"):
        if isinstance(payload.get(name), list):
            payload[name] = tuple(payload[name])
    return ClaimScopeRow(**payload)


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"


__all__ = [
    "ClaimScopeError",
    "ClaimScopeLedger",
    "ClaimScopeRow",
    "build_claim_scope",
    "read_claim_scope",
    "validate_claim_scope",
    "write_claim_scope",
]
