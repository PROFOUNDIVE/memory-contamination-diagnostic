from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from memcontam.experiment.phase12.contracts import canonical_json_hash
from memcontam.manifests.aggregate_manifest import (
    AggregateManifest,
    AggregateManifestRow,
    write_aggregate_manifest,
)
from memcontam.manifests.claim_scope import ClaimScopeLedger, ClaimScopeRow, write_claim_scope
from memcontam.manifests.run_manifest import (
    ExploratoryRunManifestRow,
    PreRouteRunManifestRow,
    RunManifest,
    SelectedRouteRunManifestRow,
    write_run_manifest,
)


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "phase12"


def _archive_module() -> Any:
    return importlib.import_module("memcontam.manifests.archive_validation")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _complete_archive(root: Path) -> tuple[Path, PreRouteRunManifestRow]:
    config = json.loads((FIXTURE_ROOT / "FX-CONFIG-001.json").read_text(encoding="utf-8"))
    config_path = root / "config.json"
    raw_log_path = root / "raw.jsonl"
    output_path = root / "output.json"
    public_manifest_path = root / "public_artifact_manifest.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    raw_log_path.write_text('{"trial_id":"trial-1"}\n', encoding="utf-8")
    output_path.write_text('{"result":"fixture"}', encoding="utf-8")
    public_manifest_path.write_text('{"status":"completed"}', encoding="utf-8")

    row = PreRouteRunManifestRow(
        run_id="run-1",
        metadata_kind="pre_route",
        git_commit=config["repository_commit"],
        run_template_id="fixture-prefix-template",
        run_template_hash="fixture-prefix-template-hash",
        run_template_registry_id="fixture-registry",
        run_template_registry_hash="fixture-registry-hash",
        trajectory_seed=1,
        seed_slot=None,
        config_path=str(config_path),
        config_hash=canonical_json_hash(config),
        raw_log_path=str(raw_log_path),
        raw_log_hash=_sha256(raw_log_path),
        raw_record_range=(0, 0),
        output_path=str(output_path),
        output_hash=_sha256(output_path),
        public_artifact_manifest_path=str(public_manifest_path),
        public_artifact_manifest_hash=canonical_json_hash({"status": "completed"}),
        scientific_result=False,
        scientific_admission_hash_or_none=None,
        rerun_parent_id=None,
        route_selection_manifest_id=None,
        route_selection_manifest_hash=None,
        seed_allocation_manifest_id=None,
        seed_allocation_manifest_hash=None,
        exploratory_activation_manifest_id=None,
        exploratory_activation_manifest_hash=None,
    )
    write_run_manifest(RunManifest((row,)), root / "run_manifest.jsonl")
    aggregate = AggregateManifest(
        (
            AggregateManifestRow(
                aggregate_id="aggregate-1",
                estimand="clean_minus_contam",
                population={"task_family": "game24"},
                evidence_layer="build",
                value=0.5,
                status="supported",
                run_ids=(row.run_id,),
                seed_ids=(1,),
                original_weights={"1": 1.0},
                weights={"1": 1.0},
                exclusions=(),
                metadata_kind="pre_route",
                run_template_registry_id=row.run_template_registry_id,
                run_template_registry_hash=row.run_template_registry_hash,
                route_selection_manifest_id=None,
                route_selection_manifest_hash=None,
                seed_allocation_manifest_id=None,
                seed_allocation_manifest_hash=None,
                exploratory_activation_manifest_id=None,
                exploratory_activation_manifest_hash=None,
            ),
        )
    )
    write_aggregate_manifest(aggregate, root / "aggregate_manifest.jsonl")
    write_claim_scope(
        ClaimScopeLedger(
            (
                ClaimScopeRow(
                    claim_id="claim-1",
                    aggregate_ids=("aggregate-1",),
                    estimand="clean_minus_contam",
                    population={"task_family": "game24"},
                    evidence_layer="build",
                    exclusions=(),
                    prohibited_extrapolations=("causal",),
                    status="supported",
                    scope="fixture-only",
                    original_weights={"1": 1.0},
                    weights={"1": 1.0},
                    route_selection_manifest_id=None,
                    route_selection_manifest_hash=None,
                    seed_allocation_manifest_id=None,
                    seed_allocation_manifest_hash=None,
                    exploratory_activation_manifest_id=None,
                    exploratory_activation_manifest_hash=None,
                ),
            )
        ),
        root / "claim_scope_table.md",
    )
    (root / "bfv2_certificate.json").write_text(
        json.dumps({"overall_status": "blocked", "reason": "missing_cached_bge_m3"}),
        encoding="utf-8",
    )
    return root, row


def test_reconstructs_complete_non_scientific_archive(tmp_path: Path) -> None:
    module = _archive_module()
    root, _ = _complete_archive(tmp_path)

    report = module.validate_archive(root)

    assert report.archive_valid is True
    assert report.errors == ()
    assert report.resolved_edges >= 7


def test_reconstructs_selected_and_scientific_exploratory_provenance(tmp_path: Path) -> None:
    module = _archive_module()
    root, pre_route = _complete_archive(tmp_path)
    route = json.loads((FIXTURE_ROOT / "FX-ROUTE-001.json").read_text(encoding="utf-8"))
    selection = route["valid_external_selection_manifest"]
    allocation = route["valid_seed_allocation_manifest"]
    activation = route["valid_exploratory_activation_manifest"]
    plan = route["valid_exploratory_plan"]
    selected_slot, selected_seed = next(iter(allocation["slot_to_seed"].items()))
    exploratory_slot, exploratory_seed = next(iter(activation["exploratory_slot_to_seed"].items()))
    common = {
        name: getattr(pre_route, name)
        for name in (
            "git_commit",
            "config_path",
            "config_hash",
            "raw_log_path",
            "raw_log_hash",
            "raw_record_range",
            "output_path",
            "output_hash",
            "public_artifact_manifest_path",
            "public_artifact_manifest_hash",
        )
    }
    selected = SelectedRouteRunManifestRow(
        run_id="selected-1",
        metadata_kind="selected_route",
        run_template_id="selected-template",
        run_template_hash="selected-template-hash",
        run_template_registry_id=allocation["run_template_registry_id"],
        run_template_registry_hash=allocation["run_template_registry_hash"],
        trajectory_seed=selected_seed,
        seed_slot=selected_slot,
        scientific_result=False,
        scientific_admission_hash_or_none=None,
        rerun_parent_id=None,
        route_selection_manifest_id=selection["manifest_id"],
        route_selection_manifest_hash=selection["artifact_hash"],
        seed_allocation_manifest_id=allocation["manifest_id"],
        seed_allocation_manifest_hash=allocation["artifact_hash"],
        exploratory_activation_manifest_id=None,
        exploratory_activation_manifest_hash=None,
        **common,
    )
    exploratory = ExploratoryRunManifestRow(
        run_id="exploratory-1",
        metadata_kind="exploratory_code_scientific",
        run_template_id="exploratory-template",
        run_template_hash="exploratory-template-hash",
        run_template_registry_id=plan["exploratory_run_template_registry_id"],
        run_template_registry_hash=plan["exploratory_run_template_registry_hash"],
        trajectory_seed=exploratory_seed,
        seed_slot=exploratory_slot,
        scientific_result=True,
        scientific_admission_hash_or_none="admission-hash",
        rerun_parent_id=None,
        route_selection_manifest_id=selection["manifest_id"],
        route_selection_manifest_hash=selection["artifact_hash"],
        seed_allocation_manifest_id=allocation["manifest_id"],
        seed_allocation_manifest_hash=allocation["artifact_hash"],
        exploratory_activation_manifest_id=activation["manifest_id"],
        exploratory_activation_manifest_hash=activation["artifact_hash"],
        **common,
    )
    write_run_manifest(RunManifest((pre_route, selected, exploratory)), root / "run_manifest.jsonl")
    governance = root / "governance"
    governance.mkdir()
    for name, payload in {
        "route_selections.json": [selection],
        "seed_allocations.json": [allocation],
        "exploratory_activations.json": [activation],
        "feasibility_reports.json": route["feasibility_reports"],
        "pilot_b.json": route["valid_pilot_b_manifest"],
        "mft.json": route["valid_mft_manifest"],
        "code_matrix_plans.json": [plan],
        "resource_manifests.json": [route["valid_selected_package_resource_manifest"]],
    }.items():
        (governance / name).write_text(json.dumps(payload), encoding="utf-8")

    report = module.validate_archive(root)

    assert report.archive_valid is True
    assert report.errors == ()


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("missing_manifest", "MANIFEST_DESTINATION_MISSING"),
        ("unsupported_claim", "UNSUPPORTED_CLAIM"),
        ("output_hash", "OUTPUT_HASH_MISMATCH"),
        ("raw_range", "MISSING_RAW_RANGE"),
        ("orphan_rerun", "ORPHAN_RERUN"),
        ("route_on_pre_route", "ROUTE_SELECTION_FORBIDDEN_PRE_ROUTE"),
    ],
)
def test_reports_each_registered_archive_break(
    tmp_path: Path, mutation: str, expected: str
) -> None:
    module = _archive_module()
    root, row = _complete_archive(tmp_path)

    if mutation == "missing_manifest":
        (root / "aggregate_manifest.jsonl").unlink()
    elif mutation == "unsupported_claim":
        claim_path = root / "claim_scope_table.md"
        payload = json.loads(claim_path.read_text(encoding="utf-8"))
        payload["aggregate_ids"] = ["missing"]
        claim_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    else:
        changed = {
            "output_hash": replace(row, output_hash="stale"),
            "raw_range": replace(row, raw_record_range=(1, 1)),
            "orphan_rerun": replace(row, rerun_parent_id="missing"),
            "route_on_pre_route": replace(row, route_selection_manifest_id="route-1"),
        }[mutation]
        (root / "run_manifest.jsonl").write_text(
            json.dumps(changed.to_dict(), sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )

    report = module.validate_archive(root)

    assert report.archive_valid is False
    assert [error.code for error in report.errors] == [expected]
