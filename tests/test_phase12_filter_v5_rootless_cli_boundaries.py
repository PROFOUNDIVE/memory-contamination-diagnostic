from __future__ import annotations

import argparse
from pathlib import Path

import pytest
from pydantic import ValidationError

from memcontam.experiment.phase12 import cli as phase12_cli
from memcontam.experiment.phase12.filter_challenge import cli as filter_cli
from memcontam.experiment.phase12.filter_challenge.registry_common import RegistryValidationError
from memcontam.experiment.phase12.filter_challenge.rootless_local_firewall import (
    has_forbidden_rootless_profile,
)
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


@pytest.mark.parametrize(
    "raw",
    (
        b'{"metadata":{"profile":"local_rootless_non_authoritative"}}',
        b'[{"profile":"local_rootless_non_authoritative"}]',
        b'{"note":"profile: local_rootless_non_authoritative"}',
        b"metadata:\n  profile: local_rootless_non_authoritative\n",
        b"note: |\n  profile: local_rootless_non_authoritative\n",
        b"# profile: local_rootless_non_authoritative\n",
        b'note: "profile: local_rootless_non_authoritative"\n',
    ),
)
def test_nested_and_text_profile_decoys_are_not_rootless_authority(raw: bytes) -> None:
    assert not has_forbidden_rootless_profile(raw)


@pytest.mark.parametrize(
    "raw",
    (
        "metadata:\n  profile: local_rootless_non_authoritative\n",
        "note: |\n  profile: local_rootless_non_authoritative\n",
        "# profile: local_rootless_non_authoritative\n",
        'note: "profile: local_rootless_non_authoritative"\n',
    ),
)
def test_yaml_decoys_keep_legacy_validation_reason(tmp_path: Path, raw: str) -> None:
    policy = tmp_path / "policy.yaml"
    policy.write_text(raw, encoding="utf-8")

    with pytest.raises((RegistryValidationError, ValidationError)) as error:
        filter_cli._load_policy(policy)

    assert ROOTLESS_FORBIDDEN not in str(error.value)


def test_nested_json_admission_decoy_keeps_legacy_validation_reason(tmp_path: Path) -> None:
    admission = tmp_path / "admission.json"
    admission.write_text(
        '{"metadata":{"profile":"local_rootless_non_authoritative"}}', encoding="utf-8"
    )

    with pytest.raises(SystemExit, match="ADMISSION_EVIDENCE_INVALID"):
        phase12_cli._load_admission_evidence(
            argparse.Namespace(
                admission_bundle=admission,
                run_family="pilot_a",
                candidate="3w",
                mode="text_only",
            ),
            True,
        )
