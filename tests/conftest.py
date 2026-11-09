from pathlib import Path

import pytest

from memcontam.readiness.phase13_core_datasets import load_core_task


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption('--phase13-resource-root', type=Path, default=None)
    parser.addoption('--phase13-embedding-cache', type=Path, default=None)


@pytest.fixture(scope='session')
def offline_resource_root(pytestconfig: pytest.Config) -> Path:
    root = pytestconfig.getoption('phase13_resource_root')
    if root is None or not root.is_absolute() or not root.is_dir():
        pytest.fail('An existing absolute --phase13-resource-root is required')
    for task in ('mmlu_pro_engineering', 'mmlu_pro_physics'):
        assert len(load_core_task(root / 'data/phase13/core/materialized', task)) == 250
    return root


@pytest.fixture(scope='session')
def offline_embedding_cache(pytestconfig: pytest.Config) -> Path:
    cache = pytestconfig.getoption('phase13_embedding_cache')
    if cache is None or not cache.is_absolute() or not cache.is_dir():
        pytest.fail('An existing absolute --phase13-embedding-cache is required')
    return cache
