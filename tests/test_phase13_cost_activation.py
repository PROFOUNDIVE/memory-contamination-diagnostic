from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from memcontam.readiness.phase13_cost_activation import (
    Phase13CostActivationError,
    validate_activated_cost_policy,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "data/phase13/main/cost_envelope_v2/activated_policy_corrected_v2.json"


def test_current_authority_activates_the_deterministic_cost_bound() -> None:
    report = validate_activated_cost_policy(ROOT)

    assert report.status == "PASS"
    assert report.cmax_main_krw == 444_256
    assert report.margin_to_core_gate_krw == 5_744
    assert report.main_execution_authorized is False
    assert report.main_a_measured_scientific_execution_count == 0


def test_cost_activation_rejects_stale_current_authority_hash(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    shutil.copytree(ROOT / "data", root / "data")
    payload = json.loads((root / ARTIFACT.relative_to(ROOT)).read_text(encoding="utf-8"))
    payload["authority"]["post_cutoff_addendum_sha256"] = "0" * 64
    (root / ARTIFACT.relative_to(ROOT)).write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(Phase13CostActivationError, match="COST_ACTIVATION_AUTHORITY_MISMATCH"):
        validate_activated_cost_policy(root)
