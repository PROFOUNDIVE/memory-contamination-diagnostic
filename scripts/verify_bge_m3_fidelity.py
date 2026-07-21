from __future__ import annotations

import hashlib
import json
import os
import socket
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

from memcontam.clients import openai_compatible as openai_compatible_module
from memcontam.clients.config import ProviderConfig
from memcontam.clients.openai_compatible import OpenAICompatibleClient
from memcontam.clients.provider_profile import normalize_provider_profile, provider_profile_id
from memcontam.cli import load_config, run_config
from memcontam.memory.embeddings import BgeM3EmbeddingProvider


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "baseline_fidelity_v2_bge_smoke.yaml"


class _MockUsage:
    def model_dump(self) -> dict[str, int]:
        return {"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12}


class _MockResponse:
    choices = [SimpleNamespace(message=SimpleNamespace(content="final: 6 / (1 - (3 / 4))"))]
    usage = _MockUsage()

    def model_dump(self) -> dict[str, object]:
        return {"choices": [{"message": {"content": self.choices[0].message.content}}]}


class _MockCompletions:
    def create(self, **_kwargs: Any) -> _MockResponse:
        return _MockResponse()


class _MockOpenAI:
    def __init__(self, **_kwargs: Any) -> None:
        self.chat = SimpleNamespace(completions=_MockCompletions())


@contextmanager
def _network_denied() -> Iterator[None]:
    original_connect = socket.socket.connect
    original_create_connection = socket.create_connection

    def deny(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("network access is forbidden during BGE-M3 fidelity verification")

    socket.socket.connect = deny
    socket.create_connection = deny
    try:
        yield
    finally:
        socket.socket.connect = original_connect
        socket.create_connection = original_create_connection


def _config_hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _validate_run(run_dir: Path) -> dict[str, object]:
    resolved = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    profile = json.loads((run_dir / "provider_profile.json").read_text(encoding="utf-8"))
    trials = [
        json.loads(line)
        for line in (run_dir / "trials.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    calls = [
        json.loads(line)
        for line in (run_dir / "calls.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    provider_identity = f"{BgeM3EmbeddingProvider.MODEL_ID}@{BgeM3EmbeddingProvider.REVISION}"
    if resolved["embedding"]["mode"] != "pinned_semantic":
        raise AssertionError("F1C did not use pinned_semantic embeddings")
    if profile["provider"] != "openai_compatible":
        raise AssertionError("F1C did not use the mocked live provider profile")
    if resolved["run"]["provider_profile_id"] != provider_profile_id(
        normalize_provider_profile(
            ProviderConfig.from_run_config(resolved),
            served_models=resolved["models"],
            model_snapshots=resolved["run"]["model_snapshots"],
        )
    ):
        raise AssertionError("provider profile ID does not match the resolved config")
    expected_hash = _config_hash(resolved)
    if any(trial["metadata"].get("config_hash") != expected_hash for trial in trials):
        raise AssertionError("trial config hash does not match resolved config")
    rag_trials = [trial for trial in trials if trial["baseline"] == "retrieval_rag"]
    bot_trials = [trial for trial in trials if trial["baseline"] == "bot_style"]
    if not rag_trials or any(len(trial["retrieved_memory"]) != 3 for trial in rag_trials):
        raise AssertionError("RAG did not retrieve top-3 from the non-empty corpus")
    if any(
        trial["metadata"]["corpus_identity"]["embedding_provider_identity"] != provider_identity
        for trial in rag_trials
    ):
        raise AssertionError("RAG corpus identity does not join the pinned provider")
    if not any(trial["memory_before"] for trial in bot_trials):
        raise AssertionError("BoT did not exercise retrieval/admission with a non-empty buffer")
    if not any((trial.get("memory_write_event") or {}).get("status") == "accepted" for trial in bot_trials):
        raise AssertionError("BoT did not write an admitted memory entry")
    if not calls or any(call["model"] != "f1c_mocked_live" for call in calls):
        raise AssertionError("mocked live answer dispatch was not recorded")
    return {
        "overall": "pass",
        "provider_identity": provider_identity,
        "rag_retrieval_count": len(rag_trials[0]["retrieved_memory"]),
        "bot_nonempty_buffer": True,
        "calls": len(calls),
    }


def main() -> int:
    config = load_config(CONFIG_PATH)
    config["logging"]["output_dir"] = tempfile.mkdtemp(prefix="f1c-bge-m3-")
    provider_config = ProviderConfig.from_run_config(config)
    os.environ.setdefault(provider_config.api_key_env or "OPENAI_API_KEY", "mocked-transport-only")
    original_openai = openai_compatible_module.OpenAI
    openai_compatible_module.OpenAI = _MockOpenAI
    try:
        client = OpenAICompatibleClient(provider_config)
        with _network_denied():
            run_dir = run_config(config, "f1c-bge-m3", _client_override=client)
    except RuntimeError as exc:
        message = str(exc)
        if BgeM3EmbeddingProvider.MODEL_ID in message and "from cache" in message:
            print(
                json.dumps(
                    {
                        "overall": "blocked",
                        "blocker": "missing_cached_bge_m3",
                        "detail": message,
                    },
                    sort_keys=True,
                )
            )
            return 1
        raise
    finally:
        openai_compatible_module.OpenAI = original_openai
    print(json.dumps(_validate_run(run_dir), sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
