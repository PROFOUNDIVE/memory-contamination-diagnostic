from __future__ import annotations

from pathlib import Path
import subprocess

from memcontam.tools.base import (
    ToolExecutionError,
    ToolExecutor,
    ToolInfrastructureError,
    ToolRequest,
    ToolResult,
    ToolRuntimeContract,
)
from memcontam.tools.policy import validate_scientific_runtime_contract, validate_tool_request


class PythonSandbox:
    scientific_capable = True

    def __init__(
        self,
        contract: ToolRuntimeContract,
        *,
        executor: ToolExecutor | None = None,
        lock_path: Path | None = None,
    ) -> None:
        self.contract = contract
        self.executor = executor or _OciExecutor()
        self.lock_path = lock_path or Path(__file__).resolve().parents[3] / "containers/python-sandbox/image.lock.json"

    def execute(self, request: ToolRequest) -> ToolResult:
        validate_scientific_runtime_contract(self.contract, self.lock_path)
        if self.contract.scientific and not self.executor.scientific_capable:
            raise ToolInfrastructureError("SCIENTIFIC_SUBPROCESS_FALLBACK_FORBIDDEN")
        validate_tool_request(request, self.contract)
        result = self.executor.execute(request, self.contract)
        if result.timed_out:
            raise ToolExecutionError("SANDBOX_TIMEOUT")
        if result.runtime_identity != self.contract.runtime_identity:
            raise ToolInfrastructureError("RUNTIME_IDENTITY_MISMATCH")
        return result


class _OciExecutor:
    scientific_capable = True
    executor_identity = "oci-python-sandbox"

    def __init__(self) -> None:
        self.execution_count = 0

    def execute(self, request: ToolRequest, contract: ToolRuntimeContract) -> ToolResult:
        self.execution_count += 1
        timeout = request.timeout_seconds or contract.timeout_seconds
        command = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--user",
            "65534:65534",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=16m",
            contract.oci_image,
            "python",
            "-I",
            "-c",
            request.code,
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError as error:
            raise ToolInfrastructureError("SANDBOX_RUNTIME_UNAVAILABLE") from error
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


__all__ = ["PythonSandbox"]
