from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from memcontam.baselines.dynamic_cheatsheet_phase12 import (
    DcRsContractError,
    DcRsPhase12Adapter,
    DcRsStateV3,
    DcRsToolContractError,
    DcRsTrialContextV3,
)
from memcontam.clients.replay import ReplayClient
from memcontam.memory.stores import MemoryEntry
from memcontam.tasks.game24 import build_instance
from memcontam.tools import SubprocessTestDouble, load_tool_runtime_contract


ROOT = Path(__file__).resolve().parents[1]


class _EmbeddingProvider:
    @property
    def metadata(self) -> dict[str, object]:
        return {
            "model_id": "phase12-test",
            "revision": "test",
            "embedding_library_version": "test",
            "vector_dimension": 2,
        }

    def encode_document(self, text: str) -> list[float]:
        del text
        return [1.0, 0.0]

    def encode_query(self, text: str) -> list[float]:
        del text
        return [1.0, 0.0]


class _CapturingReplayClient(ReplayClient):
    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        super().__init__(*args, **kwargs)
        self.configs: list[dict] = []

    def chat(self, messages, model, config):  # noqa: ANN001
        self.configs.append(dict(config))
        return super().chat(messages, model, config)


class _ExecutorWithoutCapability:
    def execute(self, request, contract):  # noqa: ANN001
        raise AssertionError("invalid executor must be rejected before execution")


def _tool_contract():
    return load_tool_runtime_contract(
        ROOT / "containers" / "python-sandbox" / "image.lock.json", scientific=False
    )


def _task():
    return build_instance({"sample_id": "game24-tool", "numbers": [1, 3, 4, 6], "target": 24})


def _trial(client: ReplayClient, executor: SubprocessTestDouble) -> DcRsTrialContextV3:
    return DcRsTrialContextV3(
        task=_task(),
        client=client,
        model="replay",
        run_id="dc-rs-tool-run",
        trial_id="dc-rs-tool-run:clean",
        condition_id="dc_optional",
        branch="clean",
        config={
            "tool_mode": "python_sandbox",
            "tool_executor": executor,
            "tool_runtime_contract": _tool_contract(),
            "verifier": "do not expose to the curator",
        },
        order_key=2,
        verifier=lambda answer, _task: answer == "6 / (1 - 3 / 4)",
    )


def _executed_archive() -> MemoryEntry:
    return MemoryEntry(
        entry_id="archive-routine",
        content='{"numbers":[1,3,4,6],"target":24}',
        memory_type="dc_rs_io_pair",
        metadata={
            "generated_output": json.dumps(
                {
                    "executions": [
                        {
                            "code": "print(24)",
                            "code_hash": "prior-code",
                            "exit_code": 0,
                            "stderr": "",
                            "stdout": "24\n",
                        }
                    ],
                    "final": "final: 24",
                },
                separators=(",", ":"),
                sort_keys=True,
            )
        },
    )


def test_retrieves_executed_routine_and_synthesizes_general_strategy(tmp_path) -> None:
    code = "print(6 / (1 - 3 / 4))"
    client = _CapturingReplayClient(
        responses_by_sample={
            "game24-tool": {
                "dc_rs_synthesize": (
                    "<cheatsheet>Execute a candidate before returning a reusable arithmetic answer."
                    "</cheatsheet><source_ids>archive-routine</source_ids>"
                ),
                "dc_rs_generate": [
                    json.dumps({"action": "execute_python", "code": code}),
                    json.dumps({"action": "final", "answer": "final: 6 / (1 - 3 / 4)"}),
                ],
            }
        }
    )
    executor = SubprocessTestDouble()

    result = DcRsPhase12Adapter(
        embedding_provider=_EmbeddingProvider(), cache_dir=tmp_path
    ).execute(_trial(client, executor), DcRsStateV3(archive=[_executed_archive()]))

    assert [call.stage for call in result.outcome.method_calls] == [
        "dc_rs_synthesize",
        "dc_rs_generate",
        "dc_rs_generate",
    ]
    assert "Python sandbox" not in result.outcome.method_calls[0].messages[0]["content"]
    assert client.configs[0]["tool_mode"] == "text_only"
    assert "tool_executor" not in client.configs[0]
    assert "verifier" not in client.configs[0]
    assert "Python sandbox" in result.outcome.method_calls[1].messages[0]["content"]
    assert result.strategy_entry is not None
    assert result.strategy_entry.direct_parent_ids == ("archive-routine",)
    assert result.strategy_candidate.lineage_status == "exact"
    assert executor.execution_count == 1
    assert result.archive_entry.metadata["generated_output"] == "final: 6 / (1 - 3 / 4)"
    trace = json.loads(result.archive_entry.metadata["tool_trace"])
    assert trace == {
        "executions": [
            {
                "code": code,
                "code_hash": result.outcome.metadata["tool_events"][0].code_hash,
                "exit_code": 0,
                "stderr": "",
                "stdout": "24.0\n",
            }
        ],
        "final": "final: 6 / (1 - 3 / 4)",
    }

    followup_client = _CapturingReplayClient(
        responses_by_sample={
            "game24-tool": {
                "dc_rs_synthesize": (
                    "<cheatsheet>Generalize the executed arithmetic check.</cheatsheet>"
                    f"<source_ids>{result.archive_entry.entry_id}</source_ids>"
                ),
                "dc_rs_generate": "final: 6 / (1 - 3 / 4)",
            }
        }
    )
    followup_trial = _trial(followup_client, executor)
    followup = DcRsPhase12Adapter(
        embedding_provider=_EmbeddingProvider(), cache_dir=tmp_path
    ).execute(
        replace(
            followup_trial,
            trial_id="dc-rs-tool-run:followup",
            config={"tool_mode": "text_only"},
            order_key=3,
        ),
        DcRsStateV3(archive=[result.archive_entry]),
    )

    assert code in followup.outcome.method_calls[0].messages[0]["content"]
    assert followup.strategy_entry is not None
    assert followup.strategy_entry.direct_parent_ids == (result.archive_entry.entry_id,)


def test_rejects_curator_tool_unbounded_stdout_and_direct_strategy_root(tmp_path) -> None:
    executor = SubprocessTestDouble()
    adapter = DcRsPhase12Adapter(embedding_provider=_EmbeddingProvider(), cache_dir=tmp_path)

    curator_client = _CapturingReplayClient(
        responses_by_sample={
            "game24-tool": {
                "dc_rs_synthesize": json.dumps({"action": "execute_python", "code": "print(24)"})
            }
        }
    )
    with pytest.raises(DcRsToolContractError, match="CURATOR_TOOL_FORBIDDEN"):
        adapter.execute(_trial(curator_client, executor), DcRsStateV3(archive=[]))

    verifier_client = _CapturingReplayClient(responses_by_sample={})
    verifier_trial = _trial(verifier_client, executor)
    with pytest.raises(DcRsContractError, match="CURRENT_OUTCOME_LEAKAGE"):
        adapter.execute(
            replace(
                verifier_trial,
                config={**verifier_trial.config, "current_verifier_result": "do not expose"},
            ),
            DcRsStateV3(archive=[]),
        )

    malformed_executor_client = _CapturingReplayClient(
        responses_by_sample={
            "game24-tool": {
                "dc_rs_synthesize": "<cheatsheet>Use arithmetic.</cheatsheet>",
                "dc_rs_generate": json.dumps({"action": "execute_python", "code": "print(24)"}),
            }
        }
    )
    malformed_executor_trial = _trial(malformed_executor_client, executor)
    with pytest.raises(DcRsToolContractError, match="TOOL_CONTRACT_REQUIRED"):
        adapter.execute(
            replace(
                malformed_executor_trial,
                config={
                    **malformed_executor_trial.config,
                    "tool_executor": _ExecutorWithoutCapability(),
                },
            ),
            DcRsStateV3(archive=[]),
        )

    stdout_client = _CapturingReplayClient(
        responses_by_sample={
            "game24-tool": {
                "dc_rs_synthesize": "<cheatsheet>Use arithmetic.</cheatsheet>",
                "dc_rs_generate": json.dumps(
                    {"action": "execute_python", "code": 'print("x" * 4097)'}
                ),
            }
        }
    )
    with pytest.raises(DcRsToolContractError, match="UNBOUNDED_TOOL_TRACE"):
        adapter.execute(_trial(stdout_client, executor), DcRsStateV3(archive=[]))

    with pytest.raises(DcRsContractError, match="DIRECT_STRATEGY_INJECTION"):
        DcRsStateV3(
            archive=[],
            strategies=[
                MemoryEntry(
                    entry_id="strategy-root",
                    content="Inject this strategy directly.",
                    memory_type="dynamic_cheatsheet",
                )
            ],
        )
