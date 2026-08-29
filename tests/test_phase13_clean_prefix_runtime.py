from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from memcontam.clients.base import LLMResponse
from memcontam.readiness import phase13_clean_prefix
from memcontam.readiness.phase13_clean_prefix_runtime import execute_clean_prefix_calibration


CONFIG = Path("configs/phase13/clean_prefix_calibration_v1.yaml")


class _ScriptedProviderError(RuntimeError):
    pass


class _Embedder:
    embedding_contract = {
        "dimension": 2,
        "normalized": True,
        "production_identity": "BAAI/bge-m3@5617a9f61b028005a4858fdac845db406aefb181",
        "provider": "test",
    }

    def encode_document(self, text: str) -> list[float]:
        del text
        return [1.0, 0.0]

    def encode_query(self, text: str) -> list[float]:
        del text
        return [1.0, 0.0]


class _Client:
    def __init__(self) -> None:
        self.reflexion_attempts: dict[str, int] = {}

    def chat(self, messages, model, config) -> LLMResponse:  # noqa: ANN001
        del messages, model
        sample_id = config["sample_id"]
        stage = config["method_stage"]
        answer = _answer(sample_id)
        if stage == "reflexion_generate":
            attempt = self.reflexion_attempts.get(sample_id, 0) + 1
            self.reflexion_attempts[sample_id] = attempt
            content = "final: wrong" if attempt % 2 else answer
        elif stage == "reflexion_reflect":
            content = json.dumps(
                {
                    "mode": "corrective",
                    "failure_class": "incorrect_answer",
                    "reflection_text": "Check the verifier target.",
                    "explicitly_used_memory_ids": [],
                }
            )
        elif stage == "bot_problem_distill":
            content = json.dumps(
                {
                    "key_information": "Use the task input.",
                    "restrictions": "Return a verifier-valid answer.",
                    "distilled_task": "Solve the task.",
                }
            )
        elif stage == "bot_instantiate_solve":
            content = json.dumps(
                {
                    "selected_structure": "registered-template",
                    "solution_trace": "Apply the task verifier contract.",
                    "final_answer": answer,
                }
            )
        elif stage == "bot_thought_distill":
            content = json.dumps(
                {
                    "description": "Verify the final answer.",
                    "template": "Solve and check against the verifier.",
                    "category": "procedure-based",
                    "explicitly_used_memory_ids": [],
                }
            )
        else:
            content = answer
        return LLMResponse(
            content=content,
            raw={"attempts": 1, "cost_usd": 0.0},
            token_usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            latency_ms=0,
        )


class _FailingClient(_Client):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def chat(self, messages, model, config) -> LLMResponse:  # noqa: ANN001
        self.calls += 1
        if self.calls > 1:
            raise _ScriptedProviderError("scripted provider failure")
        return super().chat(messages, model, config)


def _answer(sample_id: str) -> str:
    if sample_id.startswith("game24"):
        return "final: 6 / (1 - 3 / 4)"
    for row in (
        json.loads(line)
        for line in Path("data/tasks/math_equation_balancer_pilot.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ):
        if row["sample_id"] == sample_id:
            return f"final: {row['verifier_spec']['target']}"
    for row in (
        json.loads(line)
        for line in Path("data/tasks/word_sorting_pilot.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ):
        if row["sample_id"] == sample_id:
            return "final: " + " ".join(row["sorted_words"])
    raise AssertionError(sample_id)


def _authorization_bundle(
    tmp_path: Path,
    monkeypatch,  # noqa: ANN001
    run_id: str,
    config_path: Path = CONFIG,
) -> tuple[Path, Path, str]:
    monkeypatch.setattr(
        phase13_clean_prefix,
        "_git",
        lambda *arguments: "" if arguments[0] == "status" else "test-commit",
    )
    request_path = tmp_path / "request.json"
    request = phase13_clean_prefix.prepare_clean_prefix(config_path, run_id, request_path)
    authorization = {
        "schema_version": "phase13_clean_prefix_authorization_v1",
        "run_id": request["run_id"],
        "request_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
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
    return (
        request_path,
        authorization_path,
        hashlib.sha256(authorization_path.read_bytes()).hexdigest(),
    )


def _runtime_config(tmp_path: Path, artifact_root: Path | None = None) -> Path:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["output"]["artifact_root"] = str(artifact_root or tmp_path)
    path = tmp_path / "runtime-config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_clean_prefix_runtime_executes_all_frozen_trajectories(
    tmp_path: Path, monkeypatch  # noqa: ANN001
) -> None:
    config_path = _runtime_config(tmp_path)
    request_path, authorization_path, authorization_sha256 = _authorization_bundle(
        tmp_path, monkeypatch, "phase13-calibration-fake", config_path
    )
    result = execute_clean_prefix_calibration(
        config_path,
        "phase13-calibration-fake",
        client=_Client(),
        embedding_provider=_Embedder(),
        artifact_root=tmp_path,
        request_path=request_path,
        authorization_path=authorization_path,
        expected_authorization_sha256=authorization_sha256,
        allow_live_calls=True,
    )

    assert result["status"] == "completed"
    assert result["trajectory_count"] == 12
    assert result["filter_calls"] == 0
    assert result["rates"] == {
        "game24": {"eligible": 4, "attempted": 4, "rate": "4/4"},
        "math_equation_balancer": {"eligible": 4, "attempted": 4, "rate": "4/4"},
        "word_sorting": {"eligible": 4, "attempted": 4, "rate": "4/4"},
    }
    run_dir = tmp_path / "phase13-calibration-fake"
    assert (run_dir / "artifact_manifest.json").is_file()
    assert (run_dir / "archive_seal.json").is_file()


def test_failed_provider_call_is_retained_as_durable_runtime_evidence(
    tmp_path: Path, monkeypatch  # noqa: ANN001
) -> None:
    client = _FailingClient()
    config_path = _runtime_config(tmp_path)
    request_path, authorization_path, authorization_sha256 = _authorization_bundle(
        tmp_path, monkeypatch, "phase13-calibration-failure", config_path
    )

    result = execute_clean_prefix_calibration(
        config_path,
        "phase13-calibration-failure",
        client=client,
        embedding_provider=_Embedder(),
        artifact_root=tmp_path,
        request_path=request_path,
        authorization_path=authorization_path,
        expected_authorization_sha256=authorization_sha256,
        allow_live_calls=True,
    )

    run_dir = tmp_path / "phase13-calibration-failure"
    assert result["status"] == "completed"
    assert json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["status"] == "completed"
    assert json.loads((run_dir / "accounting.json").read_text(encoding="utf-8"))[
        "semantic_calls"
    ] == 1
    calls = [
        json.loads(line)
        for line in (run_dir / "calls.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(call.get("error_type") == "_ScriptedProviderError" for call in calls)
    assert (run_dir / "failures.jsonl").stat().st_size == 0
    assert (run_dir / "archive_seal.json").is_file()


def test_archive_seal_uses_verified_config_bytes(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    config_path = _runtime_config(tmp_path)
    verified_config_bytes = config_path.read_bytes()
    request_path, authorization_path, authorization_sha256 = _authorization_bundle(
        tmp_path, monkeypatch, "phase13-calibration-config-snapshot", config_path
    )

    class _MutatingClient(_Client):
        def chat(self, messages, model, config) -> LLMResponse:  # noqa: ANN001
            response = super().chat(messages, model, config)
            config_path.write_text("changed after verification\n", encoding="utf-8")
            return response

    execute_clean_prefix_calibration(
        config_path,
        "phase13-calibration-config-snapshot",
        client=_MutatingClient(),
        embedding_provider=_Embedder(),
        artifact_root=tmp_path,
        request_path=request_path,
        authorization_path=authorization_path,
        expected_authorization_sha256=authorization_sha256,
        allow_live_calls=True,
    )

    request = json.loads(request_path.read_text(encoding="utf-8"))
    seal = json.loads(
        (tmp_path / "phase13-calibration-config-snapshot" / "archive_seal.json").read_text(
            encoding="utf-8"
        )
    )
    assert seal["config_sha256"] == request["config"]["sha256"]
    assert (
        tmp_path / "phase13-calibration-config-snapshot" / "authorized_config.yaml"
    ).read_bytes() == verified_config_bytes
