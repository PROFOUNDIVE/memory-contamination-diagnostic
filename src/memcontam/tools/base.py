from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Protocol


_OCI_DIGEST = re.compile(r"^.+@sha256:[0-9a-f]{64}$")


class _ToolError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class ToolInfrastructureError(_ToolError):
    pass


class ToolExecutionError(_ToolError):
    pass


class ToolPolicyError(_ToolError):
    pass


@dataclass(frozen=True)
class ToolRequest:
    code: str
    timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ToolPolicyError("INVALID_TIMEOUT")


@dataclass(frozen=True)
class ToolResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    execution_count: int
    runtime_identity: str
    executor_identity: str


@dataclass(frozen=True)
class ToolRuntimeContract:
    oci_image: str
    python_version: str
    runtime_identity: str
    network_enabled: bool = False
    forbidden_imports: tuple[str, ...] = (
        "builtins",
        "ctypes",
        "importlib",
        "multiprocessing",
        "os",
        "pathlib",
        "subprocess",
    )
    timeout_seconds: float = 5.0
    scientific: bool = True

    def __post_init__(self) -> None:
        if "@sha256:" not in self.oci_image:
            code = (
                "OCI_IMAGE_TAG_ONLY"
                if ":" in self.oci_image.rsplit("/", 1)[-1]
                else "OCI_DIGEST_REQUIRED"
            )
            raise ToolPolicyError(code)
        if not _OCI_DIGEST.fullmatch(self.oci_image):
            raise ToolPolicyError("OCI_DIGEST_INVALID")
        if self.network_enabled:
            raise ToolPolicyError("NETWORK_FORBIDDEN")
        if self.runtime_identity != f"cpython-{self.python_version}-linux-x86_64":
            raise ToolPolicyError("RUNTIME_IDENTITY_MISMATCH")
        if self.timeout_seconds <= 0:
            raise ToolPolicyError("INVALID_TIMEOUT")


class ToolExecutor(Protocol):
    scientific_capable: bool

    def execute(self, request: ToolRequest, contract: ToolRuntimeContract) -> ToolResult: ...


__all__ = [
    "ToolExecutionError",
    "ToolExecutor",
    "ToolInfrastructureError",
    "ToolPolicyError",
    "ToolRequest",
    "ToolResult",
    "ToolRuntimeContract",
]
