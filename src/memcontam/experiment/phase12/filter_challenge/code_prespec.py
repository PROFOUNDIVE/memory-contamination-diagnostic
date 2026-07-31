from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
from typing import Final, Literal

from pydantic import Field

from memcontam.experiment.phase12.filter_challenge.registry_common import StrictRegistry


_CELL_IDS: Final = (
    "code-v2-nomem-text_only",
    "code-v2-nomem-python_sandbox",
    "code-v2-bot_style-text_only",
    "code-v2-bot_style-python_sandbox",
    "code-v2-dc_rs-text_only",
    "code-v2-dc_rs-python_sandbox",
)
_CALLS: Final = ((1, 1), (2, 2), (2, 2), (3, 3), (2, 2), (2, 3))
_FORBIDDEN: Final = frozenset(
    {
        "activation_manifest",
        "resource_manifest",
        "route_selection_manifest",
        "seed_allocation_manifest",
        "runner",
        "execution_command",
    }
)


class CodePrespecError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class CodeCell(StrictRegistry):
    cell_id: str
    baseline: Literal["nomem", "bot_style", "dc_rs"]
    tool_mode: Literal["text_only", "python_sandbox"]
    expected_calls: int = Field(ge=0)
    maximum_calls: int = Field(ge=0)
    max_tool_rounds: Literal[1]


class CodePrespec(StrictRegistry):
    schema_version: Literal["phase12_exploratory_code_source_fidelity_v2"]
    config_id: Literal["phase12-exploratory-code-source-fidelity-v2"]
    authority_snapshot: dict[str, str]
    reservation: dict[str, str | int]
    evidence_layer: Literal["build"]
    scientific_result: Literal[False]
    activation_status: Literal["inactive"]
    cost_cap_status: Literal["not_authorized_unpriced"]
    provider_calls_issued: Literal[0]
    tool_calls_issued: Literal[0]
    pilot_b_calls_issued: Literal[0]
    main_calls_issued: Literal[0]
    code_calls_issued: Literal[0]
    no_pooling: Literal[True]
    sandbox: dict[str, str | bool]
    cells: list[CodeCell]
    contrasts: list[str]
    metrics: list[str]
    fidelity_labels: dict[str, str]


def validate_code_prespec(path: Path, repository_root: Path) -> CodePrespec:
    try:
        safe_load = getattr(importlib.import_module("yaml"), "safe_load")
        payload = safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise CodePrespecError("CODE_PRESPEC_INVALID")
        if _FORBIDDEN & set(payload):
            raise CodePrespecError("CODE_PRESPEC_EXECUTION_FORBIDDEN")
        if (
            payload.get("evidence_layer") != "build"
            or payload.get("scientific_result") is not False
            or payload.get("activation_status") != "inactive"
            or payload.get("cost_cap_status") != "not_authorized_unpriced"
        ):
            raise CodePrespecError("CODE_PRESPEC_STATUS_INVALID")
        prespec = CodePrespec.model_validate(payload)
    except (AttributeError, ImportError, OSError, UnicodeError, ValueError) as error:
        if isinstance(error, CodePrespecError):
            raise
        raise CodePrespecError("CODE_PRESPEC_INVALID") from error
    _validate(prespec, repository_root)
    return prespec


def _validate(prespec: CodePrespec, repository_root: Path) -> None:
    snapshot = prespec.authority_snapshot
    snapshot_path = snapshot.get("path")
    if not isinstance(snapshot_path, str) or snapshot.get("sha256") != _sha(repository_root / snapshot_path):
        raise CodePrespecError("CODE_PRESPEC_AUTHORITY_DRIFT")
    if prespec.reservation != {
        "anchor_id": "fv5-code-anchor-game24-001",
        "task_family": "game24",
        "position": 10,
        "abstract_slot": "game24|exploratory|slot-001",
    }:
        raise CodePrespecError("CODE_PRESPEC_ANCHOR_DRIFT")
    if tuple(cell.cell_id for cell in prespec.cells) != _CELL_IDS or tuple(
        (cell.expected_calls, cell.maximum_calls) for cell in prespec.cells
    ) != _CALLS:
        raise CodePrespecError("CODE_PRESPEC_CALL_TABLE_DRIFT")
    if sum(cell.expected_calls for cell in prespec.cells) != 12 or sum(
        cell.maximum_calls for cell in prespec.cells
    ) != 13:
        raise CodePrespecError("CODE_PRESPEC_CALL_TABLE_DRIFT")
    sandbox = prespec.sandbox
    image = repository_root / "containers/python-sandbox/image.lock.json"
    oci_digest = sandbox.get("oci_digest")
    recipe_sha256 = sandbox.get("recipe_sha256")
    if (
        sandbox
        != {
            "image_lock_path": "containers/python-sandbox/image.lock.json",
            "oci_digest": "sha256:0d3b86d0b5df1ce0aa7bd7777cdecad541c8da355f02c097209d8da47b0372f3",
            "recipe_sha256": "51ae963c13280337858be40d20b89cde8d67d4a88527b186313612c2783b2c9f",
            "executor_identity": "oci-python-sandbox",
            "provider_native_interpreter": False,
        }
        or not image.is_file()
        or not isinstance(oci_digest, str)
        or not isinstance(recipe_sha256, str)
        or oci_digest not in image.read_text(encoding="utf-8")
        or recipe_sha256 not in image.read_text(encoding="utf-8")
    ):
        raise CodePrespecError("CODE_PRESPEC_SANDBOX_DRIFT")
    if (
        prespec.contrasts != [
            "within_method_text_code_verified_accuracy_delta",
            "nomem_adjusted_memory_tool_interaction",
        ]
        or prespec.metrics != [
            "tool_invocation_rate",
            "successful_execution_rate",
            "repair_rate",
            "accuracy_conditional_on_tool_use",
            "memory_code_uptake_rate",
            "code_reuse_rate",
        ]
        or prespec.fidelity_labels != {
            "nomem": "NoMem-negative-control",
            "bot_style": "BoT-code-enabled-adapted",
            "dc_rs": "DC-RS-source-oriented-adapted",
        }
    ):
        raise CodePrespecError("CODE_PRESPEC_FIDELITY_DRIFT")


def _sha(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise CodePrespecError("CODE_PRESPEC_AUTHORITY_DRIFT") from error
