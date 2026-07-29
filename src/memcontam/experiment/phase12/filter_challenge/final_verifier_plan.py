from __future__ import annotations

from pathlib import Path

from memcontam.experiment.phase12.filter_challenge.final_verifier_plan_checks import (
    LEDGER_CHECKS,
)
from memcontam.experiment.phase12.filter_challenge.final_verifier_types import FinalVerifierError
from memcontam.experiment.phase12.filter_challenge.mft_state_models import JsonValue


def verify_plan_compliance(evidence_root: Path, summary: JsonValue) -> dict[str, JsonValue]:
    for index, check in enumerate(LEDGER_CHECKS, start=1):
        if not check(evidence_root, summary):
            raise FinalVerifierError(f"LEDGER_CLAUSE_{index:02d}_REJECTED")
    return {
        "checklist": [
            {"clause_id": clause_id, "description": description, "status": "pass"}
            for clause_id, description in LEDGER_CHECKS.descriptions
        ]
    }
