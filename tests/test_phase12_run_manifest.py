from __future__ import annotations

import copy
import hashlib
import importlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from memcontam.experiment.phase12.contracts import (
    ExploratoryActivationManifest,
    PrefixTemplateSpec,
    RouteSelectionManifest,
    RunTemplateSpec,
    SeedAllocationManifest,
    canonical_json_hash,
)
from memcontam.logging.schema_v3 import parse_log_record_v3


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "phase12"


def _manifest_module() -> Any:
    return importlib.import_module("memcontam.manifests.run_manifest")


def _fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURE_ROOT / name).read_text(encoding="utf-8"))


def _metadata(schema: dict[str, Any], index: int, **updates: Any) -> Any:
    payload = copy.deepcopy(schema["valid_run_metadata"][index])
    payload.update(updates)
    return parse_log_record_v3(payload)


def _template(metadata: Any) -> PrefixTemplateSpec | RunTemplateSpec:
    common = {
        "model_snapshot": "fixture-model",
        "evidence_layer": metadata.evidence_layer,
        "task_family": metadata.task_family,
        "baseline_condition_id": metadata.baseline_condition_id,
        "sensitivity_cell_ref": {
            key: value
            for key, value in metadata.sensitivity_cell_ref.model_dump(mode="json").items()
            if value is not None
        },
        "prompt_version": "fixture-prompt",
        "tool_contract_hash": metadata.tool_contract_hash,
        "artifact_hash": f"template-hash:{metadata.run_template_id}",
    }
    if metadata.execution_key.kind == "branch_free_prefix":
        return PrefixTemplateSpec(
            prefix_template_key=metadata.run_template_id,
            execution_key=metadata.execution_key.model_dump(mode="json"),
            corpus_version="fixture-corpus",
            capacity_contract_id="fixture-capacity",
            **common,
        )
    return RunTemplateSpec(
        run_template_id=metadata.run_template_id,
        layer="extension" if metadata.run_family == "extension" else "core",
        population_layer="extension" if metadata.run_family == "extension" else "core",
        run_family=metadata.run_family,
        analysis_status="primary",
        execution_key=metadata.execution_key.model_dump(mode="json"),
        contamination_type="core",
        horizon=1,
        prefix_template_key_or_none=metadata.prefix_template_key_or_none,
        candidate_and_control_ids=("fixture-candidate",),
        corpus_index_filter_versions={"fixture": "v1"},
        **common,
    )


def _artifact_paths(tmp_path: Path) -> tuple[Path, Path, Path, Path, dict[str, Any]]:
    archive = _fixture("FX-ARCHIVE-001.json")
    config = _fixture("FX-CONFIG-001.json")
    config_path = tmp_path / "config.json"
    raw_log_path = tmp_path / "raw.jsonl"
    output_path = tmp_path / "output.json"
    public_manifest_path = tmp_path / "public_artifact_manifest.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    raw_log_path.write_text(
        "\n".join(json.dumps(record) for record in archive["raw_records"]) + "\n", encoding="utf-8"
    )
    output_path.write_text('{"result":"fixture"}', encoding="utf-8")
    public_manifest_path.write_text('{"status":"completed"}', encoding="utf-8")
    return config_path, raw_log_path, output_path, public_manifest_path, archive


def _artifact_ref(module: Any, metadata: Any, template: Any, paths: tuple[Path, Path, Path, Path]) -> Any:
    config_path, raw_log_path, output_path, public_manifest_path = paths
    return module.RunArtifactRef(
        run_id=f"run:{metadata.run_template_id}",
        metadata=metadata,
        run_template=template,
        git_commit="830b89c8c169ffa9cdea472887fdae134dbae7cf",
        seed_slot=metadata.abstract_seed_slot_or_none,
        run_template_registry_id="fixture-registry",
        run_template_registry_hash="fixture-registry-hash",
        config_path=config_path,
        config_hash=canonical_json_hash(json.loads(config_path.read_text(encoding="utf-8"))),
        raw_log_path=raw_log_path,
        raw_log_hash=hashlib.sha256(raw_log_path.read_bytes()).hexdigest(),
        raw_record_range=(0, 4),
        output_path=output_path,
        output_hash=hashlib.sha256(output_path.read_bytes()).hexdigest(),
        public_artifact_manifest_path=public_manifest_path,
        public_artifact_manifest_hash=canonical_json_hash(
            json.loads(public_manifest_path.read_text(encoding="utf-8"))
        ),
    )


def test_builds_pre_route_selected_route_and_exploratory_manifest_rows(tmp_path: Path) -> None:
    module = _manifest_module()
    schema = _fixture("FX-SCHEMA-001.json")
    route = _fixture("FX-ROUTE-001.json")
    config_path, raw_log_path, output_path, public_manifest_path, archive = _artifact_paths(tmp_path)
    paths = (config_path, raw_log_path, output_path, public_manifest_path)
    selection = RouteSelectionManifest.model_validate(route["valid_external_selection_manifest"])
    allocation = SeedAllocationManifest.model_validate(route["valid_seed_allocation_manifest"])
    activation = ExploratoryActivationManifest.model_validate(route["valid_exploratory_activation_manifest"])

    pre_route = _metadata(schema, 0, run_template_id="template-pre-route")
    selected = _metadata(
        schema,
        2,
        run_template_id="template-selected-route",
        trajectory_seed=allocation.slot_to_seed["game24|core|slot-001"],
        abstract_seed_slot_or_none="game24|core|slot-001",
        route_selection_manifest_id=selection.manifest_id,
        seed_allocation_manifest_id=allocation.manifest_id,
    )
    non_scientific = _metadata(schema, 3, run_template_id="template-exploratory-non-scientific")
    scientific = _metadata(
        schema,
        5,
        run_template_id="template-exploratory-scientific",
        trajectory_seed=activation.exploratory_slot_to_seed["game24|exploratory|exploratory-slot-001"],
        abstract_seed_slot_or_none="game24|exploratory|exploratory-slot-001",
        source_route_selection_manifest_id=selection.manifest_id,
        source_seed_allocation_manifest_id=allocation.manifest_id,
        exploratory_activation_manifest_id=activation.manifest_id,
    )
    refs = tuple(
        _artifact_ref(module, metadata, _template(metadata), paths)
        for metadata in (pre_route, selected, non_scientific, scientific)
    )

    manifest = module.build_run_manifest(refs)
    manifest_path = tmp_path / "run_manifest.jsonl"
    manifest_hash = module.write_run_manifest(manifest, manifest_path)
    loaded = module.read_run_manifest(manifest_path)
    module.validate_run_manifest(
        loaded,
        {selection.manifest_id: selection},
        {allocation.manifest_id: allocation},
        {activation.manifest_id: activation},
    )

    assert [row.metadata_kind for row in loaded.rows] == [
        "pre_route",
        "selected_route",
        "exploratory_code_non_scientific",
        "exploratory_code_scientific",
    ]
    assert isinstance(loaded.rows[0], module.PreRouteRunManifestRow)
    assert isinstance(loaded.rows[1], module.SelectedRouteRunManifestRow)
    assert isinstance(loaded.rows[2], module.ExploratoryRunManifestRow)
    assert isinstance(loaded.rows[3], module.ExploratoryRunManifestRow)
    assert loaded.rows[0].route_selection_manifest_id is None
    assert loaded.rows[1].seed_allocation_manifest_id == allocation.manifest_id
    assert loaded.rows[3].exploratory_activation_manifest_id == activation.manifest_id
    assert loaded.rows[0].raw_record_range == tuple(archive["run_manifest_rows"][0]["raw_record_range"])
    assert loaded.rows[0].config_hash == refs[0].config_hash
    assert loaded.rows[0].output_hash == refs[0].output_hash
    assert manifest_hash == canonical_json_hash([row.to_dict() for row in loaded.rows])


def test_rejects_missing_range_and_orphan_rerun(tmp_path: Path) -> None:
    module = _manifest_module()
    schema = _fixture("FX-SCHEMA-001.json")
    config_path, raw_log_path, output_path, public_manifest_path, _ = _artifact_paths(tmp_path)
    metadata = _metadata(schema, 0, run_template_id="template-pre-route")
    ref = _artifact_ref(
        module,
        metadata,
        _template(metadata),
        (config_path, raw_log_path, output_path, public_manifest_path),
    )

    with pytest.raises(module.ManifestError, match="MISSING_RAW_RANGE"):
        module.build_run_manifest((replace(ref, raw_record_range=None),))
    with pytest.raises(module.ManifestError, match="ORPHAN_RERUN"):
        module.build_run_manifest((replace(ref, rerun_parent_id="missing-run"),))
