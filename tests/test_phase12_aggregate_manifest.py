from __future__ import annotations

import json
from pathlib import Path

from memcontam.manifests.run_manifest import PreRouteRunManifestRow, RunManifest


FIXTURES = Path(__file__).parent / "fixtures" / "phase12"


def _row(run_id: str, seed: int) -> PreRouteRunManifestRow:
    return PreRouteRunManifestRow(
        run_id=run_id,
        metadata_kind="pre_route",
        git_commit="fixture-commit",
        run_template_id=f"template-{run_id.rsplit('-', 1)[-1]}",
        run_template_hash="template-hash",
        run_template_registry_id="fixture-registry",
        run_template_registry_hash="fixture-registry-hash",
        trajectory_seed=seed,
        seed_slot=None,
        config_path="/fixture/config.json",
        config_hash="config-hash",
        raw_log_path=f"/fixture/{run_id}.jsonl",
        raw_log_hash="raw-log-hash",
        raw_record_range=(0, 3),
        output_path=f"/fixture/{run_id}.json",
        output_hash="output-hash",
        public_artifact_manifest_path=f"/fixture/{run_id}-manifest.json",
        public_artifact_manifest_hash="public-manifest-hash",
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


def _aggregate_input(value: float) -> dict[str, object]:
    return {
        "aggregate_id": "agg-clean-contam",
        "estimand": "clean_minus_contam",
        "population": {"task_family": "game24", "condition": "rag_frozen"},
        "evidence_layer": "build",
        "value": value,
        "status": "supported",
        "run_ids": (
            "run-s1-clean",
            "run-s1-contam",
            "run-s2-clean",
            "run-s2-contam",
        ),
        "seed_ids": (1, 2),
        "original_weights": {"1": 0.5, "2": 0.5},
        "weights": {"1": 0.5, "2": 0.5},
        "exclusions": (),
    }


def test_links_registered_aggregate_to_runs_and_claim_scope() -> None:
    from memcontam.manifests.aggregate_manifest import build_aggregate_manifest
    from memcontam.manifests.claim_scope import build_claim_scope

    aggregate_fixture = json.loads((FIXTURES / "FX-AGG-001.json").read_text(encoding="utf-8"))
    run_manifest = RunManifest(
        (
            _row("run-s1-clean", 1),
            _row("run-s1-contam", 1),
            _row("run-s2-clean", 2),
            _row("run-s2-contam", 2),
        )
    )
    aggregate_manifest = build_aggregate_manifest(
        (_aggregate_input(aggregate_fixture["expected"]["clean_minus_contam"]),),
        run_manifest,
    )
    ledger = build_claim_scope(
        (
            {
                "claim_id": "claim-clean-contam",
                "aggregate_ids": ("agg-clean-contam",),
                "estimand": "clean_minus_contam",
                "population": {"task_family": "game24", "condition": "rag_frozen"},
                "evidence_layer": "build",
                "exclusions": (),
                "prohibited_extrapolations": ("causal", "universal", "cross-tool pooling"),
                "status": "supported",
            },
        ),
        aggregate_manifest,
    )

    aggregate = aggregate_manifest.rows[0]
    claim = ledger.rows[0]
    assert aggregate.value == aggregate_fixture["expected"]["clean_minus_contam"]
    assert aggregate.seed_ids == (1, 2)
    assert aggregate.run_ids == (
        "run-s1-clean",
        "run-s1-contam",
        "run-s2-clean",
        "run-s2-contam",
    )
    assert claim.aggregate_ids == (aggregate.aggregate_id,)
    assert claim.prohibited_extrapolations == ("causal", "universal", "cross-tool pooling")


def test_preserves_unsupported_cells_as_explicit_records() -> None:
    from memcontam.manifests.aggregate_manifest import build_aggregate_manifest

    manifest = build_aggregate_manifest(
        (
            {
                "aggregate_id": "agg-unavailable",
                "estimand": "clean_minus_contam",
                "population": {"task_family": "game24", "condition": "rag_frozen"},
                "evidence_layer": "build",
                "value": "not_estimable",
                "status": "unsupported",
                "run_ids": (),
                "seed_ids": (),
                "original_weights": None,
                "weights": None,
                "exclusions": ("incomplete_five_arm_pair",),
            },
        ),
        RunManifest(()),
    )

    assert manifest.rows[0].status == "unsupported"
    assert manifest.rows[0].value == "not_estimable"
