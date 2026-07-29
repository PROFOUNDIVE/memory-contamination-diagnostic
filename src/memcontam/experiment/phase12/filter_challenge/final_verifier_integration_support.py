from __future__ import annotations

import importlib
from pathlib import Path

from memcontam.experiment.phase12.filter_challenge.final_verifier_types import FinalVerifierError
from memcontam.experiment.phase12.filter_challenge.mft_state_models import JsonValue


def load_yaml_object(path: Path) -> dict[str, JsonValue]:
    value = getattr(importlib.import_module("yaml"), "safe_load")(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FinalVerifierError("INTEGRATION_SEARCH_CONFIG_INVALID")
    return value


def install_execution_guards(scratch_root: Path) -> Path:
    guard_root = scratch_root / "execution-guards"
    guard_root.mkdir()
    (guard_root / "sitecustomize.py").write_text(
        "from memcontam.clients import factory\n"
        "from memcontam.experiment.phase12.filter_challenge import bct\n"
        "def blocked(*args, **kwargs):\n"
        "    raise RuntimeError('FINAL_VERIFIER_EXECUTION_GUARD_REACHED')\n"
        "factory.build_llm_client = blocked\n"
        "bct.authorize_client_construction = blocked\n",
        encoding="utf-8",
    )
    return guard_root
