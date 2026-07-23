from __future__ import annotations

import pytest
from dataclasses import replace

from memcontam.manifests.aggregate_manifest import AggregateManifest, AggregateManifestRow


def _manifest() -> AggregateManifest:
    return AggregateManifest(
        (
            AggregateManifestRow(
                aggregate_id="agg-clean-contam",
                estimand="clean_minus_contam",
                population={"task_family": "game24", "condition": "rag_frozen"},
                evidence_layer="build",
                value=0.5,
                status="supported",
                run_ids=("run-s1-clean", "run-s1-contam"),
                seed_ids=(1,),
                original_weights={"1": 1.0},
                weights={"1": 1.0},
                exclusions=(),
                metadata_kind="pre_route",
                run_template_registry_id="fixture-registry",
                run_template_registry_hash="fixture-registry-hash",
                route_selection_manifest_id=None,
                route_selection_manifest_hash=None,
                seed_allocation_manifest_id=None,
                seed_allocation_manifest_hash=None,
                exploratory_activation_manifest_id=None,
                exploratory_activation_manifest_hash=None,
            ),
        )
    )


def _claim(**changes: object) -> dict[str, object]:
    claim: dict[str, object] = {
        "claim_id": "claim-clean-contam",
        "aggregate_ids": ("agg-clean-contam",),
        "estimand": "clean_minus_contam",
        "population": {"task_family": "game24", "condition": "rag_frozen"},
        "evidence_layer": "build",
        "exclusions": (),
        "prohibited_extrapolations": ("causal",),
        "status": "supported",
        "original_weights": {"1": 1.0},
        "weights": {"1": 1.0},
    }
    claim.update(changes)
    return claim


def test_rejects_unsupported_or_renormalized_claim() -> None:
    from memcontam.manifests.claim_scope import ClaimScopeError, build_claim_scope

    manifest = _manifest()
    with pytest.raises(ClaimScopeError, match="UNSUPPORTED_CLAIM"):
        build_claim_scope((_claim(aggregate_ids=("missing",)),), manifest)
    with pytest.raises(ClaimScopeError, match="WEIGHT_RENORMALIZATION_FORBIDDEN"):
        build_claim_scope((_claim(weights={"1": 0.5}),), manifest)


def test_requires_applicable_governance_and_keeps_nonclaims() -> None:
    from memcontam.manifests.claim_scope import ClaimScopeError, build_claim_scope

    route_bound = replace(
        _manifest().rows[0],
        metadata_kind="selected_route",
        route_selection_manifest_id="route-001",
        seed_allocation_manifest_id="allocation-001",
    )
    with pytest.raises(ClaimScopeError, match="ROUTE_SELECTION_REQUIRED"):
        build_claim_scope((_claim(),), AggregateManifest((route_bound,)))

    exploratory = replace(
        route_bound,
        metadata_kind="exploratory_code_scientific",
        exploratory_activation_manifest_id="activation-001",
    )
    with pytest.raises(ClaimScopeError, match="EXPLORATORY_ACTIVATION_REQUIRED"):
        build_claim_scope(
            (
                _claim(
                    route_selection_manifest_id="route-001",
                    seed_allocation_manifest_id="allocation-001",
                ),
            ),
            AggregateManifest((exploratory,)),
        )

    ledger = build_claim_scope((_claim(status="nonclaim"),), _manifest())
    assert ledger.rows[0].status == "nonclaim"
