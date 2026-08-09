from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from memcontam.experiment.phase12.filter_challenge import bct_live, registry_calibration
from memcontam.experiment.phase12.filter_challenge.bct_live import (
    CalibrationAuthorizationError,
    load_authorization,
)
from memcontam.experiment.phase12.filter_challenge.registry_calibration import (
    BCTAuthorizationV1,
    ScreeningAuthorizationV1,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "phase12" / "filter_v5_bct_calibration.yaml"
FREEZE_A = ROOT / "data" / "phase12" / "filter_v5_bct_v1" / "freeze_a.json"
AUTHORITY = ROOT / "docs" / "evidence" / "phase12-filter-v5-bct-v1" / "authority_transition_manifest.json"
SCREENING_REPORT = ROOT / "docs" / "evidence" / "phase12-filter-v5-bct-v1" / "screening_report.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime_config(path: Path, *, temperature: int = 0) -> str:
    path.write_text(
        CONFIG.read_text(encoding="utf-8")
        + "\nruntime:\n"
        + "  provider: openai_responses\n"
        + "  model_id: gpt-4o-2024-11-20\n"
        + "  decoding:\n"
        + f"    temperature: {temperature}\n"
        + "    top_p: 1\n"
        + "    max_output_tokens: 640\n"
        + "    seed_parameter_sent: false\n"
        + "    requested_seed_metadata: 0\n",
        encoding="utf-8",
    )
    return hashlib.sha256(
        json.dumps(
            {
                "provider": "openai_responses",
                "model_id": "gpt-4o-2024-11-20",
                "decoding": {
                    "temperature": temperature,
                    "top_p": 1,
                    "max_output_tokens": 640,
                    "seed_parameter_sent": False,
                    "requested_seed_metadata": 0,
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _base_request(stage: str, freeze: Path, decoding_sha256: str) -> dict[str, object]:
    calls, inputs, outputs, wall, ceiling = (90, 368_640, 57_600, 3_600, 2) if stage == "screening" else (480, 1_966_080, 307_200, 7_200, 8)
    return {
        "schema_version": "phase12_fv5_authorization_request_v1",
        "stage": stage,
        "provider": "openai_responses",
        "model_id": "gpt-4o-2024-11-20",
        "decoding_sha256": decoding_sha256,
        "freeze_sha256": _sha256(freeze),
        "maximum_calls": calls,
        "maximum_input_tokens": inputs,
        "maximum_output_tokens": outputs,
        "wall_seconds": wall,
        "hard_ceiling_usd": ceiling,
        "ledger_id": "filter-v5-bct-budget-v1",
        "artifact_root": "",
    }


def _authorization_values(request: dict[str, object], artifact_root: Path) -> dict[str, object]:
    head = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    request_path = artifact_root.parents[1] / "request.json"
    request["artifact_root"] = str(artifact_root)
    request_path.write_text(json.dumps(request, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    hard_ceiling_usd = request["hard_ceiling_usd"]
    assert isinstance(hard_ceiling_usd, int)
    return {
        "authorization_id": "authorization-001",
        "issued_at": datetime.now(UTC),
        "expires_at": datetime.now(UTC) + timedelta(hours=1),
        "run_id": "run-001",
        "request_sha256": _sha256(request_path),
        "implementation_commit": head,
        "artifact_root": str(artifact_root),
        "ledger_id": "filter-v5-bct-budget-v1",
        "model_id": "gpt-4o-2024-11-20",
        "approved_plan_sha256": "e8d44600fb3a9177ae691fd8f49ac1c06305b004db7ccd50d391c9876356a230",
        "authority_manifest_sha256": _sha256(AUTHORITY),
        "freeze_sha256": request["freeze_sha256"],
        "provider": "openai_responses",
        "decoding_sha256": request["decoding_sha256"],
        "maximum_calls": request["maximum_calls"],
        "maximum_input_tokens": request["maximum_input_tokens"],
        "maximum_output_tokens": request["maximum_output_tokens"],
        "hard_ceiling_microusd": hard_ceiling_usd * 1_000_000,
        "maximum_wall_seconds": request["wall_seconds"],
    }


def test_authorization_requires_matching_descriptor_digest_and_is_unexpired(tmp_path) -> None:
    authorization = ScreeningAuthorizationV1(
        authorization_id="screening-auth-001",
        issued_at=datetime.now(UTC),
        expires_at=datetime.now(UTC) + timedelta(hours=1),
        run_id="filter-v5-screening-v1-attempt-001",
        request_sha256="a" * 64,
        implementation_commit="b" * 40,
        artifact_root="/home/hyunwoo/git/memory-contamination-diagnostic-filter-v5/runs/phase12-filter-v5-bct-live-v1",
        ledger_id="filter-v5-bct-budget-v1",
        model_id="gpt-4o-2024-11-20",
        approved_plan_sha256="c" * 64,
        authority_manifest_sha256="d" * 64,
        freeze_sha256="e" * 64,
        provider="openai_responses",
        decoding_sha256="f" * 64,
        maximum_calls=90,
        maximum_input_tokens=368640,
        maximum_output_tokens=57600,
        hard_ceiling_microusd=2_000_000,
        maximum_wall_seconds=3600,
    )
    path = tmp_path / "authorization.json"
    path.write_text(authorization.model_dump_json(), encoding="utf-8")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()

    loaded = load_authorization(path, expected, ScreeningAuthorizationV1)

    assert loaded.authorization_id == authorization.authorization_id
    with pytest.raises(CalibrationAuthorizationError, match="AUTHORIZATION_DIGEST_MISMATCH"):
        load_authorization(path, "0" * 64, ScreeningAuthorizationV1)


def test_screening_cost_preview_serializes_a_schedule_digest(tmp_path: Path) -> None:
    output = tmp_path / "screening-request.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "memcontam.cli",
            "phase12",
            "filter-v5",
            "screening-cost-preview",
            "--config",
            str(CONFIG),
            "--freeze-a",
            str(FREEZE_A),
            "--ledger",
            str(registry_calibration.ARTIFACT_ROOT / "budget-ledger.jsonl"),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert len(json.loads(output.read_text(encoding="utf-8"))["schedule_sha256"]) == 64


@pytest.mark.parametrize("drift", ("authorization", "request", "request_ceiling", "config", "freeze"))
def test_screening_authorization_drift_never_creates_root_or_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drift: str
) -> None:
    artifact_root = tmp_path / "runs" / "phase12-filter-v5-bct-live-v1"
    config = tmp_path / "filter_v5_bct_calibration.yaml"
    decoding_sha256 = _runtime_config(config)
    if drift == "config":
        _runtime_config(config, temperature=1)
    freeze = FREEZE_A
    if drift == "freeze":
        freeze = tmp_path / "freeze_a.json"
        freeze.write_bytes(FREEZE_A.read_bytes() + b"\n")
    monkeypatch.setattr(registry_calibration, "ARTIFACT_ROOT", artifact_root)
    request = _base_request("screening", FREEZE_A, decoding_sha256)
    if drift == "request":
        request["decoding_sha256"] = "0" * 64
    values = _authorization_values(request, artifact_root)
    if drift == "request_ceiling":
        request["hard_ceiling_usd"] = "invalid"
        (tmp_path / "request.json").write_text(
            json.dumps(request, sort_keys=True, separators=(",", ":")), encoding="utf-8"
        )
    if drift == "authorization":
        values["decoding_sha256"] = "0" * 64
    authorization = ScreeningAuthorizationV1.model_validate(values)
    authorization_path = tmp_path / "authorization.json"
    authorization_path.write_text(authorization.model_dump_json(), encoding="utf-8")
    calls = 0

    def factory() -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(bct_live, "_build_live_factory", lambda _: factory())
    result = bct_live._run_cli_stage(
        argparse.Namespace(
            config=config,
            freeze_a=freeze,
            artifact_root=artifact_root,
            run_id="run-001",
            stage_result=tmp_path / "stage-result.json",
            authorization=authorization_path,
            expected_authorization_sha256=_sha256(authorization_path),
            authorization_request=tmp_path / "request.json",
        ),
        "screening",
    )

    assert result.disposition == "blocked_before_stage"
    assert result.provider_calls_issued == 0
    assert calls == 0
    assert not artifact_root.exists()


@pytest.mark.parametrize("drift", ("screening_terminal_seal", "ledger_head"))
def test_bct_authorization_evidence_drift_never_creates_root_or_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drift: str
) -> None:
    artifact_root = tmp_path / "runs" / "phase12-filter-v5-bct-live-v1"
    config = tmp_path / "filter_v5_bct_calibration.yaml"
    decoding_sha256 = _runtime_config(config)
    freeze = tmp_path / "freeze_b.json"
    freeze.write_text(
        json.dumps(
            {"provider": "openai_responses", "model_id": "gpt-4o-2024-11-20", "method_call_schedule": []}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(registry_calibration, "ARTIFACT_ROOT", artifact_root)
    request = _base_request("bct", freeze, decoding_sha256)
    screening_seal = json.loads(SCREENING_REPORT.read_text(encoding="utf-8"))["output_seal"]
    request["screening_terminal_seal"] = screening_seal
    request["ledger_head"] = "0" * 64
    values = _authorization_values(request, artifact_root)
    values["screening_terminal_seal"] = screening_seal
    values["ledger_head"] = "0" * 64
    values[drift] = "f" * 64
    authorization = BCTAuthorizationV1.model_validate(values)
    authorization_path = tmp_path / "authorization.json"
    authorization_path.write_text(authorization.model_dump_json(), encoding="utf-8")
    calls = 0

    def factory() -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(bct_live, "_build_live_factory", lambda _: factory())
    monkeypatch.setattr(bct_live, "waiting_screening_stage", lambda *_: None)
    result = bct_live._run_cli_stage(
        argparse.Namespace(
            config=config,
            freeze_b=freeze,
            artifact_root=artifact_root,
            run_id="run-001",
            stage_result=tmp_path / "stage-result.json",
            authorization=authorization_path,
            expected_authorization_sha256=_sha256(authorization_path),
            authorization_request=tmp_path / "request.json",
        ),
        "bct",
    )

    assert result.disposition == "blocked_before_stage"
    assert result.provider_calls_issued == 0
    assert calls == 0
    assert not artifact_root.exists()
