from __future__ import annotations

import selectors
import subprocess
import sys
from pathlib import Path

import pytest

from .phase13_runner_safety_fixture import FakeProvider, open_run
from .test_phase13_runner_safety import entrypoint_bytes as entrypoint_bytes
from .test_phase13_runner_safety import entrypoint_fixture as entrypoint_fixture
from .test_phase13_runner_safety import local_authority as local_authority
from .test_phase13_runner_safety import provider as provider
from .test_phase13_runner_safety import source_selection as source_selection
from .test_phase13_v3_entrypoint_integration import deny_external as deny_external


@pytest.mark.parametrize("crash", [False, True])
def test_owner_covers_completed_requests_through_parent_commit(entrypoint_fixture, provider: FakeProvider, crash: bool) -> None:
    run = open_run(entrypoint_fixture, create=True)
    run.close()
    command = [sys.executable, "-c", "from pathlib import Path; from tests.phase13_runner_safety_fixture "
               "import launcher; import sys; launcher(Path(sys.argv[1]), execute=True)", str(entrypoint_fixture.repository_root)]
    owner = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert owner.stdout is not None
        with selectors.DefaultSelector() as ready:
            ready.register(owner.stdout, selectors.EVENT_READ)
            assert ready.select(timeout=240), "fake runner failed to reach pre-commit barrier"
        output = owner.stdout.readline().strip()
        assert owner.stderr is not None
        assert output == "COMPLETED 50 50", owner.stderr.read()
        second = subprocess.run(command, input="\n", capture_output=True, text=True, timeout=120)
        assert second.returncode == 0, second.stderr
        assert second.stdout.strip() == "MAIN_RUN_ALREADY_OWNED"
        if crash:
            owner.kill()
        stdout, stderr = owner.communicate(input=None if crash else "\n", timeout=120)
        assert owner.returncode == (-9 if crash else 0), (stdout, stderr)
        recovered = open_run(entrypoint_fixture, create=False)
        try:
            assert recovered.status().completed_count == (0 if crash else 1)
            assert recovered.status().provider_calls_issued == 50
            if crash:
                with pytest.raises(ValueError, match="MAIN_RUN_IN_FLIGHT_RECONCILIATION_REQUIRED"):
                    recovered.execute(Path("unused"), max_units=1, tranche_ceiling_krw=450000,
                                      provider_factory=provider.factory)
            else:
                assert recovered.execute(Path("unused"), max_units=1, tranche_ceiling_krw=450000,
                    provider_factory=provider.factory).provider_calls_issued == 50
            assert (provider.constructors, len(provider.requests)) == (0, 0)
        finally:
            recovered.close()
    finally:
        if owner.poll() is None:
            owner.kill()
        owner.communicate(timeout=10)
