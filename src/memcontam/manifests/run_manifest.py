from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence, cast

from memcontam.experiment.phase12.contracts import (
    ExploratoryActivationManifest,
    PrefixTemplateSpec,
    RouteSelectionManifest,
    RunTemplateSpec,
    SeedAllocationManifest,
    canonical_json_hash,
)
from memcontam.logging.schema_v3 import (
    NonScientificExploratoryCodeRunMetadata,
    PreRouteRunMetadata,
    RunMetadataV3,
    ScientificExploratoryCodeRunMetadata,
    SelectedRouteRunMetadata,
    ScientificAdmissionReference,
    parse_log_record_v3,
)


class ManifestError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


TemplateSpec = PrefixTemplateSpec | RunTemplateSpec


@dataclass(frozen=True)
class RunArtifactRef:
    run_id: str
    metadata: RunMetadataV3
    run_template: TemplateSpec
    git_commit: str
    seed_slot: str | None
    run_template_registry_id: str
    run_template_registry_hash: str
    config_path: Path
    config_hash: str
    raw_log_path: Path
    raw_log_hash: str
    raw_record_range: tuple[int, int] | None
    output_path: Path
    output_hash: str
    public_artifact_manifest_path: Path
    public_artifact_manifest_hash: str
    route_selection_manifest_hash: str | None = None
    seed_allocation_manifest_hash: str | None = None
    exploratory_activation_manifest_hash: str | None = None
    rerun_parent_id: str | None = None


@dataclass(frozen=True)
class _RunManifestRowBase:
    run_id: str
    metadata_kind: str
    git_commit: str
    run_template_id: str
    run_template_hash: str
    run_template_registry_id: str
    run_template_registry_hash: str
    trajectory_seed: int
    seed_slot: str | None
    config_path: str
    config_hash: str
    raw_log_path: str
    raw_log_hash: str
    raw_record_range: tuple[int, int]
    output_path: str
    output_hash: str
    public_artifact_manifest_path: str
    public_artifact_manifest_hash: str
    scientific_result: bool
    scientific_admission_hash_or_none: str | None
    rerun_parent_id: str | None
    route_selection_manifest_id: str | None
    route_selection_manifest_hash: str | None
    seed_allocation_manifest_id: str | None
    seed_allocation_manifest_hash: str | None
    exploratory_activation_manifest_id: str | None
    exploratory_activation_manifest_hash: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreRouteRunManifestRow(_RunManifestRowBase):
    metadata_kind: Literal["pre_route"]


@dataclass(frozen=True)
class SelectedRouteRunManifestRow(_RunManifestRowBase):
    metadata_kind: Literal["selected_route"]


@dataclass(frozen=True)
class ExploratoryRunManifestRow(_RunManifestRowBase):
    metadata_kind: Literal["exploratory_code_non_scientific", "exploratory_code_scientific"]


RunManifestRow = PreRouteRunManifestRow | SelectedRouteRunManifestRow | ExploratoryRunManifestRow


@dataclass(frozen=True)
class RunManifest:
    rows: tuple[RunManifestRow, ...]


def build_run_manifest(rows: Sequence[RunArtifactRef]) -> RunManifest:
    manifest = RunManifest(tuple(_build_row(ref) for ref in rows))
    _validate_local(manifest)
    return manifest


def write_run_manifest(manifest: RunManifest, path: Path | str) -> str:
    _validate_local(manifest)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(_canonical_json(row.to_dict()) for row in manifest.rows), encoding="utf-8"
    )
    return canonical_json_hash([row.to_dict() for row in manifest.rows])


def read_run_manifest(path: Path | str) -> RunManifest:
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        manifest = RunManifest(
            tuple(_row_from_dict(json.loads(line)) for line in lines if line.strip())
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        if isinstance(error, ManifestError):
            raise
        raise ManifestError("MANIFEST_SCHEMA_INVALID") from error
    _validate_local(manifest)
    return manifest


def validate_run_manifest(
    manifest: RunManifest,
    route_manifests: Mapping[str, RouteSelectionManifest],
    seed_allocations: Mapping[str, SeedAllocationManifest],
    activations: Mapping[str, ExploratoryActivationManifest],
) -> None:
    _validate_local(manifest)
    for row in manifest.rows:
        if isinstance(row, PreRouteRunManifestRow):
            _validate_pre_route(row)
        elif isinstance(row, SelectedRouteRunManifestRow):
            _validate_selected_route(row, route_manifests, seed_allocations)
        else:
            _validate_exploratory(row, route_manifests, seed_allocations, activations)


def load_run_artifact_refs(path: Path | str) -> tuple[RunArtifactRef, ...]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        records = payload["rows"] if isinstance(payload, dict) else payload
        return tuple(_artifact_ref_from_dict(record) for record in records)
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise ManifestError("MANIFEST_INPUT_INVALID") from error


def _build_row(ref: RunArtifactRef) -> RunManifestRow:
    _validate_artifact_ref(ref)
    metadata = ref.metadata
    admission: ScientificAdmissionReference | None
    if isinstance(metadata, (PreRouteRunMetadata, NonScientificExploratoryCodeRunMetadata)):
        admission = metadata.scientific_admission_ref_or_none
    else:
        admission = metadata.scientific_admission_ref
    common: dict[str, Any] = dict(
        run_id=ref.run_id,
        git_commit=ref.git_commit,
        run_template_id=_template_id(ref.run_template),
        run_template_hash=ref.run_template.artifact_hash,
        run_template_registry_id=ref.run_template_registry_id,
        run_template_registry_hash=ref.run_template_registry_hash,
        trajectory_seed=metadata.trajectory_seed,
        seed_slot=ref.seed_slot,
        config_path=str(ref.config_path.resolve()),
        config_hash=ref.config_hash,
        raw_log_path=str(ref.raw_log_path.resolve()),
        raw_log_hash=ref.raw_log_hash,
        raw_record_range=ref.raw_record_range,
        output_path=str(ref.output_path.resolve()),
        output_hash=ref.output_hash,
        public_artifact_manifest_path=str(ref.public_artifact_manifest_path.resolve()),
        public_artifact_manifest_hash=ref.public_artifact_manifest_hash,
        scientific_result=metadata.scientific_result,
        scientific_admission_hash_or_none=(
            None if admission is None else canonical_json_hash(admission.model_dump(mode="json"))
        ),
        rerun_parent_id=ref.rerun_parent_id,
    )
    if isinstance(metadata, PreRouteRunMetadata):
        return PreRouteRunManifestRow(
            metadata_kind="pre_route",
            route_selection_manifest_id=None,
            route_selection_manifest_hash=None,
            seed_allocation_manifest_id=None,
            seed_allocation_manifest_hash=None,
            exploratory_activation_manifest_id=None,
            exploratory_activation_manifest_hash=None,
            **common,
        )
    if isinstance(metadata, SelectedRouteRunMetadata):
        return SelectedRouteRunManifestRow(
            metadata_kind="selected_route",
            route_selection_manifest_id=metadata.route_selection_manifest_id,
            route_selection_manifest_hash=_required_governance_hash(
                ref.route_selection_manifest_hash, "ROUTE_SELECTION_MISMATCH"
            ),
            seed_allocation_manifest_id=metadata.seed_allocation_manifest_id,
            seed_allocation_manifest_hash=_required_governance_hash(
                ref.seed_allocation_manifest_hash, "SEED_ALLOCATION_MISMATCH"
            ),
            exploratory_activation_manifest_id=None,
            exploratory_activation_manifest_hash=None,
            **common,
        )
    if isinstance(metadata, NonScientificExploratoryCodeRunMetadata):
        return ExploratoryRunManifestRow(
            metadata_kind="exploratory_code_non_scientific",
            route_selection_manifest_id=None,
            route_selection_manifest_hash=None,
            seed_allocation_manifest_id=None,
            seed_allocation_manifest_hash=None,
            exploratory_activation_manifest_id=None,
            exploratory_activation_manifest_hash=None,
            **common,
        )
    if isinstance(metadata, ScientificExploratoryCodeRunMetadata):
        return ExploratoryRunManifestRow(
            metadata_kind="exploratory_code_scientific",
            route_selection_manifest_id=metadata.source_route_selection_manifest_id,
            route_selection_manifest_hash=_required_governance_hash(
                ref.route_selection_manifest_hash, "ROUTE_SELECTION_MISMATCH"
            ),
            seed_allocation_manifest_id=metadata.source_seed_allocation_manifest_id,
            seed_allocation_manifest_hash=_required_governance_hash(
                ref.seed_allocation_manifest_hash, "SEED_ALLOCATION_MISMATCH"
            ),
            exploratory_activation_manifest_id=metadata.exploratory_activation_manifest_id,
            exploratory_activation_manifest_hash=_required_governance_hash(
                ref.exploratory_activation_manifest_hash, "ACTIVATION_MISMATCH"
            ),
            **common,
        )
    raise ManifestError("UNKNOWN_METADATA_KIND")


def _validate_artifact_ref(ref: RunArtifactRef) -> None:
    if not ref.run_id:
        raise ManifestError("MANIFEST_SCHEMA_INVALID")
    if ref.raw_record_range is None:
        raise ManifestError("MISSING_RAW_RANGE")
    _validate_template(ref.metadata, ref.run_template)
    if ref.seed_slot != ref.metadata.abstract_seed_slot_or_none:
        raise ManifestError("SEED_ASSIGNMENT_MISMATCH")
    _validate_artifact_paths(
        config_path=ref.config_path,
        config_hash=ref.config_hash,
        raw_log_path=ref.raw_log_path,
        raw_log_hash=ref.raw_log_hash,
        raw_record_range=ref.raw_record_range,
        output_path=ref.output_path,
        output_hash=ref.output_hash,
        public_artifact_manifest_path=ref.public_artifact_manifest_path,
        public_artifact_manifest_hash=ref.public_artifact_manifest_hash,
    )


def _required_governance_hash(value: str | None, code: str) -> str:
    if not value:
        raise ManifestError(code)
    return value


def _validate_template(metadata: RunMetadataV3, template: TemplateSpec) -> None:
    if (
        metadata.run_template_id != _template_id(template)
        or metadata.evidence_layer != template.evidence_layer
        or metadata.task_family != template.task_family
        or metadata.baseline_condition_id != template.baseline_condition_id
        or canonical_json_hash(metadata.execution_key.model_dump(mode="json"))
        != canonical_json_hash(template.execution_key.model_dump(mode="json"))
        or canonical_json_hash(
            _sensitivity_payload(metadata.sensitivity_cell_ref.model_dump(mode="json"))
        )
        != canonical_json_hash(template.sensitivity_cell_ref)
    ):
        raise ManifestError("RUN_TEMPLATE_MISMATCH")


def _template_id(template: TemplateSpec) -> str:
    return (
        template.prefix_template_key
        if isinstance(template, PrefixTemplateSpec)
        else template.run_template_id
    )


def _sensitivity_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item is not None}


def _validate_local(manifest: RunManifest) -> None:
    seen: dict[str, RunManifestRow] = {}
    for row in manifest.rows:
        if row.run_id in seen:
            raise ManifestError("DUPLICATE_RUN_ID")
        _validate_row_paths(row)
        if row.scientific_result and not row.scientific_admission_hash_or_none:
            raise ManifestError("SCIENTIFIC_ADMISSION_REQUIRED")
        if not row.scientific_result and row.scientific_admission_hash_or_none is not None:
            raise ManifestError("ADMISSION_EVIDENCE_FORBIDDEN")
        if row.rerun_parent_id is not None:
            parent = seen.get(row.rerun_parent_id)
            if parent is None:
                raise ManifestError("ORPHAN_RERUN")
            if (
                parent.run_template_id != row.run_template_id
                or parent.trajectory_seed != row.trajectory_seed
                or parent.seed_slot != row.seed_slot
                or parent.config_hash != row.config_hash
            ):
                raise ManifestError("RERUN_PARENT_MISMATCH")
        seen[row.run_id] = row


def _validate_row_paths(row: RunManifestRow) -> None:
    _validate_artifact_paths(
        config_path=Path(row.config_path),
        config_hash=row.config_hash,
        raw_log_path=Path(row.raw_log_path),
        raw_log_hash=row.raw_log_hash,
        raw_record_range=row.raw_record_range,
        output_path=Path(row.output_path),
        output_hash=row.output_hash,
        public_artifact_manifest_path=Path(row.public_artifact_manifest_path),
        public_artifact_manifest_hash=row.public_artifact_manifest_hash,
    )


def _validate_artifact_paths(
    *,
    config_path: Path,
    config_hash: str,
    raw_log_path: Path,
    raw_log_hash: str,
    raw_record_range: tuple[int, int],
    output_path: Path,
    output_hash: str,
    public_artifact_manifest_path: Path,
    public_artifact_manifest_hash: str,
) -> None:
    _require_hash(config_path, config_hash, "CONFIG_HASH_MISMATCH", canonical=True)
    _require_hash(raw_log_path, raw_log_hash, "RAW_LOG_HASH_MISMATCH")
    _require_hash(output_path, output_hash, "OUTPUT_HASH_MISMATCH")
    _require_hash(
        public_artifact_manifest_path,
        public_artifact_manifest_hash,
        "PUBLIC_ARTIFACT_MANIFEST_HASH_MISMATCH",
        canonical=True,
    )
    if (
        len(raw_record_range) != 2
        or any(type(offset) is not int for offset in raw_record_range)
        or raw_record_range[0] < 0
        or raw_record_range[0] > raw_record_range[1]
    ):
        raise ManifestError("MISSING_RAW_RANGE")
    try:
        record_count = sum(
            1 for line in raw_log_path.read_text(encoding="utf-8").splitlines() if line.strip()
        )
    except OSError as error:
        raise ManifestError("RAW_LOG_HASH_MISMATCH") from error
    if raw_record_range[1] >= record_count:
        raise ManifestError("MISSING_RAW_RANGE")


def _require_hash(path: Path, expected: str, code: str, *, canonical: bool = False) -> None:
    try:
        if canonical:
            observed = canonical_json_hash(json.loads(path.read_text(encoding="utf-8")))
        else:
            observed = hashlib.sha256(path.read_bytes()).hexdigest()
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ManifestError(code) from error
    if not expected or observed != expected:
        raise ManifestError(code)


def _validate_pre_route(row: PreRouteRunManifestRow) -> None:
    if row.route_selection_manifest_id is not None or row.route_selection_manifest_hash is not None:
        raise ManifestError("ROUTE_SELECTION_FORBIDDEN_PRE_ROUTE")
    if row.seed_allocation_manifest_id is not None or row.seed_allocation_manifest_hash is not None:
        raise ManifestError("SEED_ALLOCATION_FORBIDDEN_PRE_ROUTE")
    if (
        row.exploratory_activation_manifest_id is not None
        or row.exploratory_activation_manifest_hash is not None
    ):
        raise ManifestError("EXPLORATORY_ACTIVATION_FORBIDDEN")


def _validate_selected_route(
    row: SelectedRouteRunManifestRow,
    route_manifests: Mapping[str, RouteSelectionManifest],
    seed_allocations: Mapping[str, SeedAllocationManifest],
) -> None:
    selection = _route_selection(row, route_manifests)
    allocation = _seed_allocation(row, seed_allocations)
    if (
        selection.seed_allocation_manifest_id != allocation.manifest_id
        or selection.seed_allocation_manifest_hash != allocation.artifact_hash
        or selection.selected_route != allocation.selected_route
    ):
        raise ManifestError("SEED_ALLOCATION_MISMATCH")
    _validate_seed_assignment(row, allocation.slot_to_seed)


def _validate_exploratory(
    row: ExploratoryRunManifestRow,
    route_manifests: Mapping[str, RouteSelectionManifest],
    seed_allocations: Mapping[str, SeedAllocationManifest],
    activations: Mapping[str, ExploratoryActivationManifest],
) -> None:
    if row.metadata_kind == "exploratory_code_non_scientific":
        if any(
            value is not None
            for value in (
                row.route_selection_manifest_id,
                row.route_selection_manifest_hash,
                row.seed_allocation_manifest_id,
                row.seed_allocation_manifest_hash,
                row.exploratory_activation_manifest_id,
                row.exploratory_activation_manifest_hash,
            )
        ):
            raise ManifestError("EXPLORATORY_ACTIVATION_FORBIDDEN")
        return
    selection = _route_selection(row, route_manifests)
    allocation = _seed_allocation(row, seed_allocations)
    activation_id = row.exploratory_activation_manifest_id
    if not activation_id:
        raise ManifestError("EXPLORATORY_ACTIVATION_REQUIRED")
    activation = activations.get(activation_id)
    if (
        activation is None
        or not row.exploratory_activation_manifest_hash
        or row.exploratory_activation_manifest_hash != activation.artifact_hash
    ):
        raise ManifestError("ACTIVATION_MISMATCH")
    if (
        activation.route_selection_manifest_id != selection.manifest_id
        or activation.route_selection_manifest_hash != selection.artifact_hash
        or activation.seed_allocation_manifest_id != allocation.manifest_id
        or activation.seed_allocation_manifest_hash != allocation.artifact_hash
    ):
        raise ManifestError("ACTIVATION_MISMATCH")
    _validate_seed_assignment(row, activation.exploratory_slot_to_seed)


def _route_selection(
    row: RunManifestRow, route_manifests: Mapping[str, RouteSelectionManifest]
) -> RouteSelectionManifest:
    selection_id = row.route_selection_manifest_id
    if not selection_id:
        raise ManifestError("ROUTE_SELECTION_REQUIRED")
    if not row.route_selection_manifest_hash:
        raise ManifestError("ROUTE_SELECTION_MISMATCH")
    selection = route_manifests.get(selection_id)
    if selection is None or row.route_selection_manifest_hash != selection.artifact_hash:
        raise ManifestError("ROUTE_SELECTION_MISMATCH")
    return selection


def _seed_allocation(
    row: RunManifestRow, seed_allocations: Mapping[str, SeedAllocationManifest]
) -> SeedAllocationManifest:
    allocation_id = row.seed_allocation_manifest_id
    if not allocation_id:
        raise ManifestError("SEED_ALLOCATION_REQUIRED")
    if not row.seed_allocation_manifest_hash:
        raise ManifestError("SEED_ALLOCATION_MISMATCH")
    allocation = seed_allocations.get(allocation_id)
    if allocation is None or row.seed_allocation_manifest_hash != allocation.artifact_hash:
        raise ManifestError("SEED_ALLOCATION_MISMATCH")
    return allocation


def _validate_seed_assignment(row: RunManifestRow, assignments: Mapping[str, int]) -> None:
    if not row.seed_slot or assignments.get(row.seed_slot) != row.trajectory_seed:
        raise ManifestError("SEED_ASSIGNMENT_MISMATCH")


def _row_from_dict(value: Mapping[str, Any]) -> RunManifestRow:
    payload = dict(value)
    raw_range = payload.get("raw_record_range")
    if isinstance(raw_range, list):
        payload["raw_record_range"] = tuple(raw_range)
    kind = payload.get("metadata_kind")
    if kind == "pre_route":
        return PreRouteRunManifestRow(**payload)
    if kind == "selected_route":
        return SelectedRouteRunManifestRow(**payload)
    if kind in {"exploratory_code_non_scientific", "exploratory_code_scientific"}:
        return ExploratoryRunManifestRow(**payload)
    raise ManifestError("UNKNOWN_METADATA_KIND")


def _artifact_ref_from_dict(value: Mapping[str, Any]) -> RunArtifactRef:
    parsed = parse_log_record_v3(value["metadata"])
    if not isinstance(
        parsed,
        (
            PreRouteRunMetadata,
            SelectedRouteRunMetadata,
            NonScientificExploratoryCodeRunMetadata,
            ScientificExploratoryCodeRunMetadata,
        ),
    ):
        raise ManifestError("MANIFEST_INPUT_INVALID")
    metadata = cast(RunMetadataV3, parsed)
    template_payload = value["run_template"]
    if template_payload["execution_key"]["kind"] == "branch_free_prefix":
        template: TemplateSpec = PrefixTemplateSpec.model_validate(template_payload)
    else:
        template = RunTemplateSpec.model_validate(template_payload)
    raw_range = value.get("raw_record_range")
    return RunArtifactRef(
        run_id=value["run_id"],
        metadata=metadata,
        run_template=template,
        git_commit=value["git_commit"],
        seed_slot=value.get("seed_slot"),
        run_template_registry_id=value["run_template_registry_id"],
        run_template_registry_hash=value["run_template_registry_hash"],
        config_path=Path(value["config_path"]),
        config_hash=value["config_hash"],
        raw_log_path=Path(value["raw_log_path"]),
        raw_log_hash=value["raw_log_hash"],
        raw_record_range=None if raw_range is None else tuple(raw_range),
        output_path=Path(value["output_path"]),
        output_hash=value["output_hash"],
        public_artifact_manifest_path=Path(value["public_artifact_manifest_path"]),
        public_artifact_manifest_hash=value["public_artifact_manifest_hash"],
        route_selection_manifest_hash=value.get("route_selection_manifest_hash"),
        seed_allocation_manifest_hash=value.get("seed_allocation_manifest_hash"),
        exploratory_activation_manifest_hash=value.get("exploratory_activation_manifest_hash"),
        rerun_parent_id=value.get("rerun_parent_id"),
    )


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"


__all__ = [
    "ExploratoryRunManifestRow",
    "ManifestError",
    "PreRouteRunManifestRow",
    "RunArtifactRef",
    "RunManifest",
    "RunManifestRow",
    "SelectedRouteRunManifestRow",
    "build_run_manifest",
    "load_run_artifact_refs",
    "read_run_manifest",
    "validate_run_manifest",
    "write_run_manifest",
]
