from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.phase13_fixture_support import provision_readiness


READINESS_SOURCE = Path("data/phase13/authority/pilot_a_readiness_manifest_v1.json")
READINESS_TARGET = Path(".omo/evidence/pilot-a-closeout/pilot_a_readiness_manifest.json")


@pytest.fixture(scope="session", autouse=True)
def provision_legacy_readiness_fixture() -> Iterator[None]:
    with provision_readiness(READINESS_SOURCE, READINESS_TARGET):
        yield
