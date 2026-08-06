from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path(
    os.environ.get(
        "ROOTLESS_T1_IGNORED_INPUT_ROOT",
        str(ROOT.parent / "memory-contamination-diagnostic-filter-v5"),
    )
)
IGNORED_INPUTS = (
    ".omo/plans/phase12-filter-v5-screening-bct-execution.md",
    ".omo/approvals/phase12-filter-v5-screening-bct-execution.plan.sha256",
    ".omo/approvals/phase12-post-filter-v5-calibration-readiness.plan.sha256",
    ".omo/evidence/phase12-post-filter-v5-calibration-readiness/task-3-screening-stage-result.json",
    ".omo/evidence/phase12-post-filter-v5-calibration-readiness/task-5-bct-stage-result.json",
    ".omo/evidence/phase12-post-filter-v5-calibration-readiness/task-6-pilot-b-readiness-stage-result.json",
)
EXTERNAL_INPUTS_AVAILABLE = False


def _provision_ignored_inputs() -> bool:
    if not SOURCE_ROOT.is_dir() or any(not (SOURCE_ROOT / relative).is_file() for relative in IGNORED_INPUTS):
        return False
    for relative in IGNORED_INPUTS:
        source = SOURCE_ROOT / relative
        destination = ROOT / relative
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(destination.parent, 0o700)
        if destination.exists():
            if destination.read_bytes() != source.read_bytes():
                raise pytest.UsageError(f"T1 ignored input drift: {relative}")
            continue
        shutil.copyfile(source, destination)
        os.chmod(destination, 0o600)
    return True


def pytest_sessionstart(session: pytest.Session) -> None:
    global EXTERNAL_INPUTS_AVAILABLE
    EXTERNAL_INPUTS_AVAILABLE = _provision_ignored_inputs()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if EXTERNAL_INPUTS_AVAILABLE:
        return
    marker = pytest.mark.skip(reason="detached T1 ignored inputs were not supplied to this clone")
    for item in items:
        if item.path.name in {
            "test_phase12_filter_v5_plan_digest.py",
            "test_phase12_filter_v5_evidence_security.py",
        }:
            item.add_marker(marker)
