from __future__ import annotations

from dataclasses import replace
import importlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "containers" / "python-sandbox" / "image.lock.json"
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "phase12" / "FX-TOOL-001.json"


def _tools():
    assert importlib.util.find_spec("memcontam.tools") is not None
    return importlib.import_module("memcontam.tools")


def _contract(tools, *, scientific: bool):
    return tools.load_tool_runtime_contract(LOCK_PATH, scientific=scientific)


def test_executes_deterministic_arithmetic_in_pinned_runtime() -> None:
    tools = _tools()
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    contract = _contract(tools, scientific=False)
    executor = tools.SubprocessTestDouble()

    result = tools.PythonSandbox(contract, executor=executor).execute(
        tools.ToolRequest(fixture["code"])
    )

    assert result.execution_count == fixture["expected"]["execution_count"]
    assert result.exit_code == fixture["expected"]["exit_code"]
    assert result.stderr == fixture["expected"]["stderr"]
    assert result.stdout == fixture["expected"]["stdout"]
    assert result.timed_out is fixture["expected"]["timed_out"]
    assert result.runtime_identity == contract.runtime_identity


def test_fails_closed_for_registered_sandbox_errors() -> None:
    tools = _tools()
    contract = _contract(tools, scientific=False)

    with pytest.raises(tools.ToolPolicyError, match="NETWORK_FORBIDDEN"):
        tools.PythonSandbox(contract, executor=tools.SubprocessTestDouble()).execute(
            tools.ToolRequest("import socket\nsocket.socket()\n")
        )

    with pytest.raises(tools.ToolPolicyError, match="FORBIDDEN_IMPORT"):
        tools.PythonSandbox(contract, executor=tools.SubprocessTestDouble()).execute(
            tools.ToolRequest("import subprocess\n")
        )

    with pytest.raises(tools.ToolExecutionError, match="SANDBOX_TIMEOUT"):
        tools.PythonSandbox(contract, executor=tools.SubprocessTestDouble()).execute(
            tools.ToolRequest("while True:\n    pass\n", timeout_seconds=0.01)
        )

    scientific_executor = tools.SubprocessTestDouble()
    with pytest.raises(
        tools.ToolInfrastructureError, match="SCIENTIFIC_SUBPROCESS_FALLBACK_FORBIDDEN"
    ):
        tools.PythonSandbox(
            _contract(tools, scientific=True), executor=scientific_executor
        ).execute(tools.ToolRequest("print(43)\n"))
    assert scientific_executor.execution_count == 0

    with pytest.raises(tools.ToolPolicyError, match="RUNTIME_IDENTITY_MISMATCH"):
        tools.PythonSandbox(
            replace(contract, runtime_identity="cpython-0.0.0-linux-x86_64"),
            executor=tools.SubprocessTestDouble(),
        ).execute(tools.ToolRequest("print(43)\n"))
