from memcontam.readiness.phase13_core_datasets import load_core_task
from .test_phase13_v3_artifact_builder import builder_source as builder_source
from .test_phase13_seed_zero_shadow import local_embedder as local_embedder


def test_builder_copy_loads_canonical_core(builder_source) -> None:
    root, _commit, _authority = builder_source
    for task in ('mmlu_pro_engineering', 'mmlu_pro_physics'):
        assert len(load_core_task(root / 'data/phase13/core/materialized', task)) == 250


def test_embedder_uses_explicit_cache(local_embedder) -> None:
    assert len(local_embedder.encode_query('offline fixture cache probe')) == 1024
