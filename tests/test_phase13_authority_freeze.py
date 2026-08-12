from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable

import pytest
from pydantic import ValidationError

from memcontam.readiness.phase13_authority import (
    Phase13AuthorityError,
    Phase13AuthorityFreeze,
    parse_phase13_authority_freeze,
)


AUTHORITY_HASHES = (
    ("theory", "34f63f37a49e92607c78ced038c4c70b4c9d5e3fa8fc57d6e97de1ee79db59a8"),
    ("baseline", "c28f0e2b00db6a2731f64933ccc67c5ea5a163d6233c526e6b473e540f988204"),
    ("protocol", "06d23e29dff6c607bc2035c5641fbb696fb5c09dd86f2ce190a99c6baa57eefc"),
    ("experiment_design", "6b8ab4e414c86dbcb4afc9c2781b13f9312e8ba2834d20473d261f264e6e1acf"),
)
REGISTRY_KINDS = ("calibration_v2", "execution", "analysis")


def _canonical_hash(payload: dict[str, object]) -> str:
    content = copy.deepcopy(payload)
    content.pop("closure_hash", None)
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _fixture() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "phase13-authority-freeze-v1",
        "closure_id": "phase13-calibration-v2-prospective",
        "authorities": [
            {
                "kind": "authority",
                "authority_role": role,
                "artifact": {
                    "kind": "artifact",
                    "artifact_id": f"phase13-{role}-authority",
                    "path": f"external-authority/{role}.md",
                    "sha256": digest,
                },
            }
            for role, digest in AUTHORITY_HASHES
        ],
        "parameter_classifications": [
            {
                "kind": "scientific_design",
                "class_code": "A",
                "H_primary": 5,
                "primary_analysis_window_id": "accuracy-h5-primary",
            },
            {"kind": "execution", "class_code": "B", "H_run": 10},
            {
                "kind": "inference",
                "class_code": "C",
                "estimator_id": "paired-seed-risk-difference-v1",
            },
            {
                "kind": "planning",
                "class_code": "D",
                "calibration_seed_count_per_task": 12,
            },
            {
                "kind": "reproducibility",
                "class_code": "E",
                "bootstrap_replicates": 20_000,
                "bootstrap_rng_seed": 13,
                "serialization_version": "canonical-json-v1",
            },
        ],
        "registries": [
            {
                "kind": "registry",
                "registry_kind": registry_kind,
                "registry_id": f"phase13-{registry_kind}-registry-v1",
                "artifact": {
                    "kind": "artifact",
                    "artifact_id": f"phase13-{registry_kind}-registry-v1",
                    "path": f"fixture/{registry_kind}.json",
                    "sha256": hashlib.sha256(registry_kind.encode()).hexdigest(),
                },
            }
            for registry_kind in REGISTRY_KINDS
        ],
    }
    payload["closure_hash"] = _canonical_hash(payload)
    return payload


def _parse(payload: dict[str, object]) -> Phase13AuthorityFreeze:
    return parse_phase13_authority_freeze(json.dumps(payload).encode())


def _resign(payload: dict[str, object]) -> None:
    payload["closure_hash"] = _canonical_hash(payload)


def _rows(payload: dict[str, object], key: str) -> list[dict[str, object]]:
    rows = payload[key]
    assert isinstance(rows, list)
    assert all(isinstance(row, dict) for row in rows)
    return rows


def _artifact(row: dict[str, object]) -> dict[str, object]:
    artifact = row["artifact"]
    assert isinstance(artifact, dict)
    return artifact


def test_complete_fixture_is_frozen_and_matches_independent_canonical_hash() -> None:
    payload = _fixture()

    closure = _parse(payload)

    assert closure.closure_hash == _canonical_hash(payload)
    with pytest.raises(ValidationError):
        closure.closure_id = "mutated"


Mutation = Callable[[dict[str, object]], None]


def _drop_closure_hash(payload: dict[str, object]) -> None:
    payload.pop("closure_hash")


def _drift_authority(payload: dict[str, object]) -> None:
    _artifact(_rows(payload, "authorities")[0])["sha256"] = "0" * 64
    _resign(payload)


def _unknown_registry(payload: dict[str, object]) -> None:
    _rows(payload, "registries")[0]["registry_kind"] = "unknown"
    _resign(payload)


def _duplicate_registry(payload: dict[str, object]) -> None:
    _rows(payload, "registries")[1]["registry_kind"] = "calibration_v2"
    _resign(payload)


def _wrong_class(payload: dict[str, object]) -> None:
    _rows(payload, "parameter_classifications")[0]["class_code"] = "B"
    _resign(payload)


def _missing_e_setting(payload: dict[str, object]) -> None:
    _rows(payload, "parameter_classifications")[4].pop("bootstrap_rng_seed")
    _resign(payload)


def _bare_h(payload: dict[str, object]) -> None:
    _artifact(_rows(payload, "registries")[0])["H"] = 5
    _resign(payload)


def _missing_h_primary(payload: dict[str, object]) -> None:
    _rows(payload, "parameter_classifications")[0].pop("H_primary")
    _resign(payload)


def _malformed_reference(payload: dict[str, object]) -> None:
    _artifact(_rows(payload, "registries")[0]).pop("sha256")
    _resign(payload)


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (_drop_closure_hash, "FULL_CLOSURE_REQUIRED"),
        (_drift_authority, "AUTHORITY_HASH_DRIFT"),
        (_unknown_registry, "UNKNOWN_REGISTRY"),
        (_duplicate_registry, "DUPLICATE_REGISTRY"),
        (_wrong_class, "WRONG_PARAMETER_CLASS"),
        (_missing_e_setting, "MISSING_E_SETTINGS"),
        (_bare_h, "BARE_H_PROHIBITED"),
        (_missing_h_primary, "MISSING_H_PRIMARY"),
        (_malformed_reference, "MALFORMED_REFERENCE"),
    ],
)
def test_single_field_mutations_raise_distinct_codes(mutate: Mutation, code: str) -> None:
    payload = _fixture()
    mutate(payload)

    with pytest.raises(Phase13AuthorityError) as caught:
        _parse(payload)

    assert caught.value.code == code


def test_closure_hash_drift_and_malformed_json_have_typed_codes() -> None:
    payload = _fixture()
    payload["closure_id"] = "changed-after-signing"

    with pytest.raises(Phase13AuthorityError) as drift:
        _parse(payload)
    with pytest.raises(Phase13AuthorityError) as malformed:
        parse_phase13_authority_freeze(b"{")

    assert drift.value.code == "CLOSURE_HASH_MISMATCH"
    assert malformed.value.code == "MALFORMED_CLOSURE"
