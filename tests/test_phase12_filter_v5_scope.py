from __future__ import annotations

from pathlib import Path

import pytest
from tests.test_phase12_filter_v5_final_verifier_modes import _fixture, _request

from memcontam.experiment.phase12.filter_challenge.final_verifier import (
    FinalVerifierError,
    verify_final_report,
)


@pytest.mark.parametrize(
    "forbidden_path",
    (
        "src/memcontam/memory/admission.py",
        "src/memcontam/experiment/phase12/filter_v4.py",
        "tests/test_pilot_a_preflight.py",
        "tests/test_phase12_pilot_a_invariants.py",
        "tests/test_phase12_pilot_a_launch.py",
        "src/memcontam/readiness/pilot_a_preflight.py",
        "scripts/inspect_phase12_pilot_a.py",
        "configs/phase12/pilot_a.yaml",
        "docs/phase12-pilot-a-operator-checklist.md",
        ".sisyphus/evidence/pilot-a-closeout/pilot_a_execution_manifest.json",
        ".sisyphus/evidence/pilot-a-clean-audit/pilot_a_frozen_evidence_manifest.json",
        "runs/runs/pilot-a-game24-example/public_artifact_manifest.json",
        "runs/runs/pilot-a-game24-example/archive_seal.json",
        "Pilot-A 관련 기록.md",
    ),
)
def test_scope_rejects_actual_pilot_a_and_core_path_families(
    tmp_path: Path, forbidden_path: str
) -> None:
    fixture = _fixture(tmp_path, forbidden_path=forbidden_path)

    with pytest.raises(FinalVerifierError, match="SCOPE_FORBIDDEN_DIFF"):
        verify_final_report(_request(fixture, "scope", tmp_path / "f4.json"))


def test_scope_payload_binds_changed_commit_metadata(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    report = verify_final_report(_request(fixture, "scope", tmp_path / "f4.json"))

    assert report["base_commit"] == fixture.base_commit
    assert report["implementation_commit"] == fixture.evidence.implementation_commit
    assert report["changed_paths"] == ["src/filter_v5_marker.py"]
