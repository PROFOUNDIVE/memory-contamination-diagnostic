from pathlib import Path
import importlib

import pytest


def test_external_transport_is_denied_when_guard_installed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    import socket
    import subprocess

    from memcontam.clients.openai_responses import OpenAIResponsesClient
    plugin = importlib.import_module("tests.phase13_recovery_offline")

    plugin.install_offline_denial(monkeypatch)

    with pytest.raises(plugin.OfflineTransportDenied):
        socket.create_connection(("example.invalid", 443))
    with socket.socket() as connection:
        with pytest.raises(plugin.OfflineTransportDenied):
            connection.connect(("127.0.0.1", 9))
    client = OpenAIResponsesClient.__new__(OpenAIResponsesClient)
    for method in ("chat", "count_compiled_v3", "send_compiled_v3"):
        with pytest.raises(plugin.OfflineTransportDenied):
            getattr(client, method)()
    with pytest.raises(plugin.OfflineTransportDenied):
        subprocess.run(["opencode", "run", "forbidden"], check=True)
    (tmp_path / "writable-evidence").write_bytes(b"offline evidence")


def test_credentials_are_removed_when_guard_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    plugin = importlib.import_module("tests.phase13_recovery_offline")

    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-not-a-secret")

    removed = plugin.install_offline_denial(monkeypatch)

    assert "OPENAI_API_KEY" in removed
    assert "OPENAI_API_KEY" not in os.environ


def test_guarded_python_child_can_reopen_evidence_when_transport_denied(tmp_path: Path) -> None:
    import subprocess
    import sys

    source = tmp_path / "evidence.json"
    source.write_text('{"offline":true}')
    script = tmp_path / "reopen.py"
    script.write_text(
        "import json, socket\n"
        "from pathlib import Path\n"
        "from tests.phase13_recovery_offline import OfflineTransportDenied\n"
        f"assert json.loads(Path({str(source)!r}).read_bytes()) == {{'offline': True}}\n"
        "try:\n"
        "    socket.create_connection(('example.invalid', 443))\n"
        "except OfflineTransportDenied:\n"
        "    print('REOPENED_OFFLINE')\n"
        "else:\n"
        "    raise AssertionError('transport was not denied')\n"
    )

    completed = subprocess.run(
        [sys.executable, "-B", "-m", "tests.phase13_recovery_offline", str(script)],
        check=True, capture_output=True, text=True,
    )

    assert completed.stdout.strip() == "REOPENED_OFFLINE"


def test_dispatch_output_serializes_existing_ordinary_archive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    import hashlib
    import json

    from pydantic import ValidationError

    from memcontam.experiment import phase13_ordinary_runtime as ordinary
    from memcontam.readiness.phase13_main_live_evidence import MainRuntimeEvidence
    from memcontam.readiness.phase13_main_live_runtime_support import dispatch_output, production_identity
    from memcontam.readiness.phase13_main_production import ProductionObject
    from memcontam.readiness.phase13_production_observability import (
        ProductionObservabilityArchive, validate_production_archive,
    )
    from memcontam.readiness.phase13_production_runtime_join import production_archive_from_ordinary
    from memcontam.evaluation.phase13_observability_registration import ObservabilityRegistrationPacket
    from memcontam.tasks.base import TaskInstance

    from .test_phase13_native_production_archive import NativeResponses

    monkeypatch.setattr(ordinary, "_validated_common_capacity_tokens", lambda: 8192)
    task = TaskInstance(sample_id="task5-offline", task_name="game24",
                        input={"numbers": [1, 3, 4, 6], "target": 24}, verifier_spec={"target": 24})
    raw = Path("data/phase13/observability/registration_packet_v2.json").read_bytes()
    unit = ProductionObject(
        sequence=0, unit_id="5" * 64, kind="NO_MEMORY_SINGLETON", seed=0,
        task="game24", memory_baseline=None, arm="NOT_APPLICABLE", prefix_unit_id=None,
        projected_cost_krw=0, execution_template_id="game24:nomem:task5",
        ordered_sample_ids_sha256=hashlib.sha256(json.dumps((task.sample_id,), separators=(",", ":")).encode()).hexdigest(),
        registration_packet_sha256=hashlib.sha256(raw).hexdigest(), checkpoint_registry_sha256="6" * 64,
    )
    identity = production_identity(unit, scientific_result=False)
    run = ordinary.ProspectiveOrdinaryRun(
        task_name="game24", baseline="nomem", run_id="task5-offline", model="gpt-5.6-luna",
        client=NativeResponses(), allow_test_client=True, verifier=lambda _answer, _task: True,
        decoding={"temperature": 0.0}, tasks=(task,), trajectory_seed=0,
    )
    result = ordinary.execute_prospective_ordinary(run)
    archive = production_archive_from_ordinary(run, result, identity)

    output = dispatch_output(unit, result.trials, identity, archive, realized_cost_krw=0)

    runtime = MainRuntimeEvidence.model_validate_json(json.dumps(output.evidence))
    assert runtime.production_observability_archive == archive
    assert len(archive.records) == 1
    assert validate_production_archive(
        archive, ObservabilityRegistrationPacket.model_validate_json(raw),
        identity.registration_packet_sha256, frozen_tasks=(task,),
    ).status == "PASS"
    (tmp_path / "ordinary-output.json").write_text(output.model_dump_json())
    absent = dispatch_output(unit, result.trials, identity, realized_cost_krw=0)
    assert MainRuntimeEvidence.model_validate_json(json.dumps(absent.evidence)).production_observability_archive is None
    with pytest.raises(ValidationError):
        ProductionObservabilityArchive.model_validate(None)
