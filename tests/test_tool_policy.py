from __future__ import annotations

from dataclasses import replace
import ast
import importlib
import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "containers" / "python-sandbox" / "image.lock.json"


def _tools():
    assert importlib.util.find_spec("memcontam.tools") is not None
    return importlib.import_module("memcontam.tools")


def test_frozen_policy_requires_digest_and_exact_runtime_identity() -> None:
    tools = _tools()
    contract = tools.load_tool_runtime_contract(LOCK_PATH, scientific=True)

    assert contract.oci_image.startswith("docker.io/")
    assert "@sha256:" in contract.oci_image
    assert contract.network_enabled is False
    assert contract.runtime_identity == f"cpython-{contract.python_version}-linux-x86_64"

    with pytest.raises(tools.ToolPolicyError, match="OCI_IMAGE_TAG_ONLY"):
        replace(contract, oci_image="python:3.11")

    with pytest.raises(tools.ToolPolicyError, match="NETWORK_FORBIDDEN"):
        replace(contract, network_enabled=True)

    with pytest.raises(tools.ToolPolicyError, match="RUNTIME_IDENTITY_MISMATCH"):
        tools.validate_scientific_runtime_contract(
            replace(contract, runtime_identity="cpython-0.0.0-linux-x86_64"), LOCK_PATH
        )


def test_primary_text_only_modules_do_not_import_tools() -> None:
    for module_name in (
        "memcontam.cli",
        "memcontam.baselines.execution",
        "memcontam.baselines.bot_phase12",
        "memcontam.baselines.reflexion_phase12",
    ):
        sys.modules.pop(module_name, None)
    sys.modules.pop("memcontam.tools", None)

    for module_name in (
        "memcontam.cli",
        "memcontam.baselines.execution",
        "memcontam.baselines.bot_phase12",
        "memcontam.baselines.reflexion_phase12",
    ):
        source = importlib.util.find_spec(module_name)
        assert source is not None and source.origin is not None
        tree = ast.parse(Path(source.origin).read_text(encoding="utf-8"))
        imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        assert not any(
            name == "memcontam.tools" or name.startswith("memcontam.tools.") for name in imports
        )
