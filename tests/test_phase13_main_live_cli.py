from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import sys

import pytest

from .test_phase13_v3_entrypoint_integration import deny_external as deny_external


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("module", ["phase13_main_live_cli", "phase13_main_runner_cli"])
@pytest.mark.parametrize("version", [1, 2])
def test_historical_packages_validate_without_production_preflight(module, version, monkeypatch, capsys, deny_external):
    package = ROOT / f"data/phase13/main/mr_p5/execution_package_v{version}.json"
    authorization = ROOT / f"data/phase13/main/mr_p6/authorized_execution_v{version}.json"
    cli = importlib.import_module("memcontam.readiness." + module)
    monkeypatch.setattr(sys, "argv", [module, "validate", "--repository-root", str(ROOT),
        "--package", str(package), "--authorization", str(authorization),
        "--expected-authorization-sha256", hashlib.sha256(authorization.read_bytes()).hexdigest()])
    cli.main()
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "VALIDATED_HISTORICAL_ONLY"
    assert report["provider_calls_issued"] == 0


@pytest.mark.parametrize("command", ["run", "resume"])
@pytest.mark.parametrize("version", [1, 2])
def test_historical_packages_never_select_for_active_execution(command, version, deny_external):
    from memcontam.readiness.phase13_v3_authority_models import V3Identity
    from memcontam.readiness.phase13_v3_entrypoint import SelectionRequest, select_execution

    request = SelectionRequest(ROOT, ROOT / f"data/phase13/main/mr_p5/execution_package_v{version}.json",
        ROOT / "absent-authorization", None, None, V3Identity().run_id)
    with pytest.raises(ValueError, match="MAIN_PACKAGE_VERSION_UNSUPPORTED"):
        select_execution(request, command)


@pytest.mark.parametrize("module", ["phase13_main_live_cli", "phase13_main_runner_cli"])
def test_cli_requires_literal_authority_option(module):
    cli = importlib.import_module("memcontam.readiness." + module)
    with pytest.raises(SystemExit) as caught:
        cli._parser().parse_args(["validate", "--repository-root", ".", "--package", "p",
            "--authorization", "a", "--expected-authorization-sha256-file", "s", "--authority", "."])
    assert caught.value.code == 2
