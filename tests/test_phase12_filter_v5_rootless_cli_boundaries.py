from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from memcontam.experiment.phase12 import cli as phase12_cli
from memcontam.experiment.phase12.filter_challenge import cli as filter_cli
from memcontam.experiment.phase12.filter_challenge.registry_common import RegistryValidationError
from memcontam.readiness.scientific_admission import AdmissionDenied


ROOTLESS_FORBIDDEN = "ROOTLESS_PROFILE_FORBIDDEN"


def test_duplicate_profiles_are_rejected_before_cli_deserialization(tmp_path: Path) -> None:
    duplicate_json = (
        '{"profile":"local_rootless_non_authoritative","profile":"authoritative"}'
    )
    admission = tmp_path / "admission.json"
    admission.write_text(duplicate_json, encoding="utf-8")

    with pytest.raises(AdmissionDenied, match=ROOTLESS_FORBIDDEN):
        phase12_cli._load_admission_evidence(
            argparse.Namespace(
                admission_bundle=admission,
                run_family="pilot_a",
                candidate="3w",
                mode="text_only",
            ),
            True,
        )

    policy = tmp_path / "policy.yaml"
    policy.write_text(
        "profile: local_rootless_non_authoritative\nprofile: authoritative\n", encoding="utf-8"
    )
    with pytest.raises(RegistryValidationError, match=ROOTLESS_FORBIDDEN):
        filter_cli._load_policy(policy)
