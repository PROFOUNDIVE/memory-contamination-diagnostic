from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from memcontam.experiment.phase12.filter_challenge.final_verifier_integration_support import (
    install_execution_guards,
)


ROOT = Path(__file__).resolve().parents[1]


def test_execution_guard_startup_writes_sentinel_after_installation(tmp_path: Path) -> None:
    guard_root = install_execution_guards(tmp_path)
    sentinel = tmp_path / "guard-startup.txt"
    environment = os.environ | {
        "MEMCONTAM_FINAL_VERIFIER_GUARD_SENTINEL": str(sentinel),
        "PYTHONPATH": os.pathsep.join((str(guard_root), str(ROOT / "src"))),
    }

    result = subprocess.run(
        (sys.executable, "-c", "pass"), capture_output=True, text=True, check=False, env=environment
    )

    assert result.returncode == 0, result.stderr
    assert sentinel.read_text(encoding="utf-8") == "FINAL_VERIFIER_EXECUTION_GUARD_READY\n"


@pytest.mark.parametrize(
    "statement",
    (
        "from memcontam.clients import factory; factory.build_llm_client(None, stage='main', execution_class='live')",
        "from memcontam.clients.openai_compatible import OpenAICompatibleClient; OpenAICompatibleClient(None)",
        "from memcontam.clients.openai_responses import OpenAIResponsesClient; OpenAIResponsesClient(None, allow_live_calls=True)",
        "from memcontam.experiment.phase12.filter_challenge import bct; bct.authorize_client_construction(None, None, lambda stage: None)",
    ),
)
def test_execution_guards_block_direct_provider_and_bct_seams(tmp_path: Path, statement: str) -> None:
    guard_root = install_execution_guards(tmp_path)
    environment = os.environ | {"PYTHONPATH": os.pathsep.join((str(guard_root), str(ROOT / "src")))}

    result = subprocess.run(
        (sys.executable, "-c", statement), capture_output=True, text=True, check=False, env=environment
    )

    assert result.returncode != 0
    assert "FINAL_VERIFIER_EXECUTION_GUARD_REACHED" in result.stderr
