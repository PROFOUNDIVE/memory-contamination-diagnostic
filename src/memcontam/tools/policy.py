from __future__ import annotations

import ast
import json
from pathlib import Path

from memcontam.tools.base import ToolInfrastructureError, ToolPolicyError, ToolRequest, ToolRuntimeContract


_NETWORK_IMPORTS = frozenset(
    {"aiohttp", "asyncore", "ftplib", "http", "httpx", "requests", "smtplib", "socket", "ssl", "telnetlib", "urllib"}
)
_NETWORK_CALLS = frozenset({"connect", "create_connection", "request", "urlopen", "urlretrieve"})


def load_tool_runtime_contract(path: Path, *, scientific: bool) -> ToolRuntimeContract:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ToolInfrastructureError("SANDBOX_LOCK_UNAVAILABLE") from error
    required = {"oci_image", "python_version", "runtime_identity", "network_enabled", "forbidden_imports", "timeout_seconds"}
    if set(payload) - {"schema_version", "recipe_sha256", *required} or required - set(payload):
        raise ToolPolicyError("SANDBOX_LOCK_INVALID")
    try:
        return ToolRuntimeContract(
            oci_image=payload["oci_image"],
            python_version=payload["python_version"],
            runtime_identity=payload["runtime_identity"],
            network_enabled=payload["network_enabled"],
            forbidden_imports=tuple(payload["forbidden_imports"]),
            timeout_seconds=payload["timeout_seconds"],
            scientific=scientific,
        )
    except (TypeError, ValueError) as error:
        raise ToolPolicyError("SANDBOX_LOCK_INVALID") from error


def validate_scientific_runtime_contract(contract: ToolRuntimeContract, lock_path: Path) -> None:
    locked = load_tool_runtime_contract(lock_path, scientific=contract.scientific)
    if contract.oci_image != locked.oci_image:
        raise ToolPolicyError("OCI_IMAGE_MISMATCH")
    if (
        contract.python_version != locked.python_version
        or contract.runtime_identity != locked.runtime_identity
    ):
        raise ToolPolicyError("RUNTIME_IDENTITY_MISMATCH")
    if contract.network_enabled or contract.forbidden_imports != locked.forbidden_imports:
        raise ToolPolicyError("TOOL_POLICY_MISMATCH")
    if contract.timeout_seconds != locked.timeout_seconds:
        raise ToolPolicyError("TOOL_POLICY_MISMATCH")


def validate_tool_request(request: ToolRequest, contract: ToolRuntimeContract) -> None:
    if request.timeout_seconds is not None and request.timeout_seconds > contract.timeout_seconds:
        raise ToolPolicyError("TIMEOUT_EXCEEDS_POLICY")
    try:
        tree = ast.parse(request.code)
    except SyntaxError as error:
        raise ToolPolicyError("INVALID_PYTHON") from error
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                _validate_import(alias.name, contract)
        elif isinstance(node, ast.ImportFrom):
            _validate_import(node.module or "", contract)
        elif isinstance(node, ast.Call):
            _validate_call(node, contract)


def _validate_import(name: str, contract: ToolRuntimeContract) -> None:
    root = name.split(".", 1)[0]
    if root in _NETWORK_IMPORTS:
        raise ToolPolicyError("NETWORK_FORBIDDEN")
    if root in contract.forbidden_imports:
        raise ToolPolicyError("FORBIDDEN_IMPORT")


def _validate_call(node: ast.Call, contract: ToolRuntimeContract) -> None:
    if isinstance(node.func, ast.Name) and node.func.id == "__import__":
        if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            _validate_import(node.args[0].value, contract)
        raise ToolPolicyError("FORBIDDEN_IMPORT")
    if isinstance(node.func, ast.Attribute):
        if node.func.attr == "__import__":
            raise ToolPolicyError("FORBIDDEN_IMPORT")
        if node.func.attr in _NETWORK_CALLS:
            raise ToolPolicyError("NETWORK_FORBIDDEN")


__all__ = [
    "load_tool_runtime_contract",
    "validate_scientific_runtime_contract",
    "validate_tool_request",
]
