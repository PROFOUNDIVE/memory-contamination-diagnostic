from __future__ import annotations

import subprocess
import sys

from memcontam.tools.base import ToolRequest, ToolResult, ToolRuntimeContract


class SubprocessTestDouble:
    scientific_capable = False
    executor_identity = "subprocess-test-double"

    def __init__(self) -> None:
        self.execution_count = 0

    def execute(self, request: ToolRequest, contract: ToolRuntimeContract) -> ToolResult:
        self.execution_count += 1
        timeout = request.timeout_seconds or contract.timeout_seconds
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-c", request.code],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            return ToolResult(
                stdout=_text(error.stdout),
                stderr=_text(error.stderr),
                exit_code=-1,
                timed_out=True,
                execution_count=self.execution_count,
                runtime_identity=contract.runtime_identity,
                executor_identity=self.executor_identity,
            )
        return ToolResult(
            stdout=completed.stdout,
            stderr=completed.stderr,
            exit_code=completed.returncode,
            timed_out=False,
            execution_count=self.execution_count,
            runtime_identity=contract.runtime_identity,
            executor_identity=self.executor_identity,
        )


def _text(value: str | bytes | None) -> str:
    return value.decode() if isinstance(value, bytes) else value or ""


__all__ = ["SubprocessTestDouble"]
