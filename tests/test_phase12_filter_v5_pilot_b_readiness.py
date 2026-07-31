from __future__ import annotations

from pathlib import Path

import pytest

from memcontam.experiment.phase12.filter_challenge.code_prespec import (
    CodePrespecError,
    validate_code_prespec,
)
from memcontam.experiment.phase12.filter_challenge.pilot_b_readiness import (
    ReadinessEvidence,
    derive_terminal,
)


ROOT = Path(__file__).resolve().parents[1]
PRESPEC = ROOT / "configs" / "phase12" / "exploratory_code_source_fidelity_v2.yaml"


@pytest.mark.parametrize(
    ("evidence", "expected"),
    (
        (
            ReadinessEvidence(screening="missing"),
            "AWAITING_SCREENING_AUTHORIZATION",
        ),
        (
            ReadinessEvidence(screening="invalid"),
            "FILTER_V5_PILOT_B_BLOCKED_BY_INVALID_CALIBRATION_EVIDENCE",
        ),
        (
            ReadinessEvidence(screening="valid", common_strict_probes=1),
            "FILTER_V5_PILOT_B_NOT_ESTIMABLE",
        ),
        (
            ReadinessEvidence(screening="valid", common_strict_probes=2, freeze_b="valid"),
            "AWAITING_BCT_AUTHORIZATION",
        ),
        (
            ReadinessEvidence(
                screening="valid",
                common_strict_probes=2,
                freeze_b="valid",
                bct_authorization="valid",
                bct_archive="invalid",
            ),
            "FILTER_V5_PILOT_B_BLOCKED_BY_INVALID_BCT_EVIDENCE",
        ),
        (
            ReadinessEvidence(
                screening="valid",
                common_strict_probes=2,
                freeze_b="valid",
                bct_authorization="valid",
                bct_archive="completed",
                completed_bct_families=("BCT-FV5-01", "BCT-FV5-02", "BCT-FV5-03", "BCT-FV5-04"),
                behavioral_false_negative=True,
            ),
            "READY_FOR_SEPARATE_FILTER_V5_PILOT_B_AUTHORIZATION",
        ),
    ),
)
def test_readiness_uses_the_first_reachable_terminal(
    evidence: ReadinessEvidence, expected: str
) -> None:
    # Given: one branch of the frozen readiness evidence.
    # When: Pilot-B readiness derives its terminal.
    result = derive_terminal(evidence)

    # Then: behavioral false negatives do not become infrastructure blockers.
    assert result.terminal_status == expected
    assert result.provider_calls_issued == 0


def test_code_v2_rejects_activation_drift(tmp_path: Path) -> None:
    # Given: a copy whose build-only status was changed to active.
    mutated = tmp_path / "code-v2.yaml"
    mutated.write_text(
        PRESPEC.read_text(encoding="utf-8").replace("activation_status: inactive", "activation_status: active"),
        encoding="utf-8",
    )

    # When: the prespec validator reads the mutation.
    # Then: active execution is refused before any provider or tool path exists.
    with pytest.raises(CodePrespecError, match="CODE_PRESPEC_STATUS_INVALID"):
        validate_code_prespec(mutated, ROOT)
