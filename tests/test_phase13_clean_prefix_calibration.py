from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from memcontam.clients.base import LLMResponse
from memcontam.readiness import phase13_clean_prefix
from memcontam.readiness.phase13_clean_prefix import Phase13CalibrationError
from memcontam.readiness.phase13_clean_prefix_authorization import verify_authorization
from memcontam.readiness.phase13_clean_prefix_runtime import execute_clean_prefix_calibration


CONFIG = Path("configs/phase13/clean_prefix_calibration_v1.yaml")


def _authorization_bundle(
    tmp_path: Path,
    monkeypatch,  # noqa: ANN001
    run_id: str = "phase13-calibration-fake",
    config_path: Path = CONFIG,
) -> tuple[Path, Path, str]:
    monkeypatch.setattr(
        phase13_clean_prefix,
        "_git",
        lambda *arguments: "" if arguments[0] == "status" else "test-commit",
    )
    request_path = tmp_path / "request.json"
    request = phase13_clean_prefix.prepare_clean_prefix(config_path, run_id, request_path)
    request_sha256 = hashlib.sha256(request_path.read_bytes()).hexdigest()
    authorization = {
        "schema_version": "phase13_clean_prefix_authorization_v1",
        "run_id": request["run_id"],
        "request_sha256": request_sha256,
        "implementation_commit": request["implementation_commit"],
        "config_sha256": request["config"]["sha256"],
        "schedule_sha256": request["schedule_sha256"],
        "provider_decoding_sha256": request["provider_decoding_sha256"],
        "maximum_semantic_calls": request["budget"]["maximum_semantic_calls"],
        "maximum_transport_attempts": request["budget"]["maximum_transport_attempts"],
        "hard_ceiling_microusd": request["budget"]["hard_ceiling_microusd"],
    }
    authorization_path = tmp_path / "authorization.json"
    authorization_path.write_text(
        json.dumps(authorization, sort_keys=True) + "\n", encoding="utf-8"
    )
    authorization_sha256 = hashlib.sha256(authorization_path.read_bytes()).hexdigest()
    return request_path, authorization_path, authorization_sha256


def _runtime_config(tmp_path: Path, artifact_root: Path) -> Path:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["output"]["artifact_root"] = str(artifact_root)
    path = tmp_path / "runtime-config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


class _UnusedClient:
    def chat(self, messages, model, config) -> LLMResponse:  # noqa: ANN001
        del messages, model, config
        raise AssertionError("provider dispatch is forbidden")


class _UnusedEmbedder:
    def encode_document(self, text: str) -> list[float]:
        del text
        raise AssertionError("embedding is forbidden")

    def encode_query(self, text: str) -> list[float]:
        del text
        raise AssertionError("embedding is forbidden")


def test_prepare_clean_prefix_emits_exact_bounded_authorization_packet(tmp_path: Path) -> None:
    output = tmp_path / "authorization-request.json"

    result = subprocess.run(
        (
            sys.executable,
            "-m",
            "memcontam.cli",
            "phase13",
            "prepare-clean-prefix",
            "--config",
            str(CONFIG),
            "--run-id",
            "phase13-calibration-test",
            "--output",
            str(output),
        ),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    packet = json.loads(output.read_text(encoding="utf-8"))
    assert packet["status"] == "READY_FOR_SEPARATE_PRE_MAIN_CALIBRATION_AUTHORIZATION"
    assert packet["provider_calls_issued"] == 0
    assert packet["filter_calls"] == 0
    assert packet["schedule"]["trajectory_count"] == 12
    assert packet["schedule"]["prefix_position_count"] == 44
    assert packet["budget"] == {
        "nominal_semantic_calls": 264,
        "maximum_semantic_calls": 396,
        "maximum_transport_attempts": 1584,
        "maximum_input_tokens": 6488064,
        "maximum_output_tokens": 3244032,
        "hard_ceiling_microusd": 48660480,
    }
    assert "--allow-live-calls" in packet["execution_command"]


def test_budget_accepts_operator_ceiling_below_full_envelope() -> None:
    config = phase13_clean_prefix.load_clean_prefix_config(CONFIG)
    config["budget"]["hard_ceiling_microusd"] = 15_000_000

    budget = phase13_clean_prefix._budget(config, prefix_positions=44)

    assert budget["hard_ceiling_microusd"] == 15_000_000


def test_run_clean_prefix_denies_missing_authorization_before_provider_dispatch(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        (
            sys.executable,
            "-m",
            "memcontam.cli",
            "phase13",
            "run-clean-prefix",
            "--config",
            str(CONFIG),
            "--run-id",
            "phase13-calibration-denied",
            "--request",
            str(tmp_path / "missing-request.json"),
            "--authorization",
            str(tmp_path / "missing-authorization.json"),
            "--expected-authorization-sha256",
            "0" * 64,
        ),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "CALIBRATION_AUTHORIZATION_REQUIRED" in result.stderr
    assert not Path("runs/phase13-clean-prefix-calibration-v1/phase13-calibration-denied").exists()


def test_clean_prefix_runtime_rejects_direct_execution_without_authorization(
    tmp_path: Path,
) -> None:
    with pytest.raises(Phase13CalibrationError, match="CALIBRATION_AUTHORIZATION_REQUIRED"):
        execute_clean_prefix_calibration(
            CONFIG,
            "phase13-calibration-direct-bypass",
            client=_UnusedClient(),
            embedding_provider=_UnusedEmbedder(),
            artifact_root=tmp_path,
        )


def test_clean_prefix_runtime_rejects_unbound_artifact_root(
    tmp_path: Path, monkeypatch  # noqa: ANN001
) -> None:
    config_path = _runtime_config(tmp_path, tmp_path / "authorized-root")
    request_path, authorization_path, authorization_sha256 = _authorization_bundle(
        tmp_path,
        monkeypatch,
        "phase13-calibration-root-bypass",
        config_path,
    )

    with pytest.raises(Phase13CalibrationError, match="CALIBRATION_OUTPUT_ROOT_MISMATCH"):
        execute_clean_prefix_calibration(
            config_path,
            "phase13-calibration-root-bypass",
            client=_UnusedClient(),
            embedding_provider=_UnusedEmbedder(),
            artifact_root=tmp_path / "unbound-root",
            request_path=request_path,
            authorization_path=authorization_path,
            expected_authorization_sha256=authorization_sha256,
            allow_live_calls=True,
        )


def test_relative_output_root_is_bound_independently_of_cwd(
    tmp_path: Path, monkeypatch  # noqa: ANN001
) -> None:
    run_id = "phase13-calibration-cwd-replay"
    relative_root = Path(os.path.relpath(tmp_path / "authorized-root", phase13_clean_prefix.ROOT))
    config_path = _runtime_config(tmp_path, relative_root)
    request_path, authorization_path, authorization_sha256 = _authorization_bundle(
        tmp_path, monkeypatch, run_id, config_path
    )
    (phase13_clean_prefix.ROOT / relative_root / run_id).resolve().mkdir(parents=True)
    alternate_cwd = tmp_path / "alternate-cwd"
    alternate_cwd.mkdir()
    monkeypatch.chdir(alternate_cwd)

    with pytest.raises(Phase13CalibrationError, match="RUN_ID_ALREADY_EXISTS"):
        execute_clean_prefix_calibration(
            config_path,
            run_id,
            client=_UnusedClient(),
            embedding_provider=_UnusedEmbedder(),
            request_path=request_path,
            authorization_path=authorization_path,
            expected_authorization_sha256=authorization_sha256,
            allow_live_calls=True,
        )


def test_authorization_rejects_dirty_execution_state(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    request_path, authorization_path, authorization_sha256 = _authorization_bundle(
        tmp_path, monkeypatch
    )
    monkeypatch.setattr(
        phase13_clean_prefix,
        "_git",
        lambda *arguments: " M src/memcontam/cli.py"
        if arguments[0] == "status"
        else "test-commit",
    )

    with pytest.raises(Phase13CalibrationError, match="CALIBRATION_WORKTREE_DIRTY"):
        verify_authorization(
            config_path=CONFIG,
            run_id="phase13-calibration-fake",
            request_path=request_path,
            authorization_path=authorization_path,
            expected_authorization_sha256=authorization_sha256,
            allow_live_calls=True,
        )


def test_authorization_recomputes_request_contract(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    request_path, authorization_path, _ = _authorization_bundle(tmp_path, monkeypatch)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["schedule_sha256"] = "0" * 64
    request_path.write_text(json.dumps(request, sort_keys=True) + "\n", encoding="utf-8")
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    authorization["request_sha256"] = hashlib.sha256(request_path.read_bytes()).hexdigest()
    authorization["schedule_sha256"] = request["schedule_sha256"]
    authorization_path.write_text(
        json.dumps(authorization, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(Phase13CalibrationError, match="CALIBRATION_REQUEST_MISMATCH"):
        verify_authorization(
            config_path=CONFIG,
            run_id="phase13-calibration-fake",
            request_path=request_path,
            authorization_path=authorization_path,
            expected_authorization_sha256=hashlib.sha256(
                authorization_path.read_bytes()
            ).hexdigest(),
            allow_live_calls=True,
        )
