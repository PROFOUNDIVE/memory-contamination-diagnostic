from __future__ import annotations

import importlib
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from memcontam.experiment.phase12.filter_challenge.final_verifier_types import FinalVerifierError
from memcontam.experiment.phase12.filter_challenge.mft_state_models import JsonValue


GUARD_SENTINEL_CONTENT = "FINAL_VERIFIER_EXECUTION_GUARD_READY\n"
_GUARD_SENTINEL_ENV = "MEMCONTAM_FINAL_VERIFIER_GUARD_SENTINEL"


@dataclass(frozen=True, slots=True)
class GuardedCommand:
    result: subprocess.CompletedProcess[str]
    sentinel_content: str


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
        "from memcontam.clients import openai_compatible, openai_responses\n"
        "from memcontam.experiment.phase12.filter_challenge import bct\n"
        "import os\n"
        "from pathlib import Path\n"
        "def blocked(*args, **kwargs):\n"
        "    raise RuntimeError('FINAL_VERIFIER_EXECUTION_GUARD_REACHED')\n"
        "factory.build_llm_client = blocked\n"
        "factory.OpenAICompatibleClient = blocked\n"
        "factory.OpenAIResponsesClient = blocked\n"
        "openai_compatible.OpenAICompatibleClient = blocked\n"
        "openai_responses.OpenAIResponsesClient = blocked\n"
        "bct.authorize_client_construction = blocked\n"
        "sentinel = os.environ.get('MEMCONTAM_FINAL_VERIFIER_GUARD_SENTINEL')\n"
        "if sentinel:\n"
        "    Path(sentinel).write_text('FINAL_VERIFIER_EXECUTION_GUARD_READY\\n', encoding='utf-8')\n",
        encoding="utf-8",
    )
    return guard_root


def run_guarded_command(
    root: Path, command_id: str, arguments: tuple[str, ...], guard_root: Path
) -> GuardedCommand:
    sentinel = guard_root / f"{command_id}-{uuid4().hex}.sentinel"
    environment = os.environ | {
        _GUARD_SENTINEL_ENV: str(sentinel),
        "PYTHONPATH": os.pathsep.join((str(guard_root), str(Path(__file__).parents[4]))),
    }
    result = subprocess.run(
        (sys.executable, "-m", "memcontam.cli", "phase12", "filter-v5", command_id, *arguments),
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if "FINAL_VERIFIER_EXECUTION_GUARD_REACHED" in result.stderr:
        raise FinalVerifierError("FINAL_VERIFIER_EXECUTION_GUARD_REACHED")
    if not sentinel.is_file():
        raise FinalVerifierError("FINAL_VERIFIER_EXECUTION_GUARD_REACHED")
    sentinel_content = sentinel.read_text(encoding="utf-8")
    if sentinel_content != GUARD_SENTINEL_CONTENT:
        raise FinalVerifierError("FINAL_VERIFIER_EXECUTION_GUARD_REACHED")
    return GuardedCommand(result=result, sentinel_content=sentinel_content)
