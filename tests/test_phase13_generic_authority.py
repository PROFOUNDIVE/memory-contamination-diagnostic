from __future__ import annotations

import hashlib
import json

import pytest

from memcontam.experiment.phase12.filter_challenge.mft_state_models import JsonValue
from memcontam.readiness.phase13_authority import (
    Phase13AuthorityError,
    parse_authority_freeze,
    parse_authority_requirements,
)


def _hash(payload: dict[str, JsonValue], field: str) -> str:
    unsigned = dict(payload)
    unsigned.pop(field, None)
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _freeze() -> dict[str, JsonValue]:
    payload: dict[str, JsonValue] = {
        "schema_version": "phase13_authority_freeze_v1",
        "freeze_id": "prospective-freeze",
        "authorities": [
            {
                "role": "protocol",
                "artifact": {"path": "authority/protocol.json", "sha256": "1" * 64},
            },
            {
                "role": "experiment_design",
                "artifact": {"path": "authority/design.json", "sha256": "2" * 64},
            },
        ],
        "registries": [
            {
                "kind": "execution",
                "registry_id": "execution-future",
                "artifact": {"path": "registry/execution.json", "sha256": "3" * 64},
            }
        ],
        "parameters": {
            "backbone": "prospective-backbone",
            "H_run": 7,
            "tasks": ["future-task"],
            "baselines": ["future-baseline"],
            "rag_corpus": "future-corpus",
        },
    }
    payload["closure_hash"] = _hash(payload, "closure_hash")
    return payload


def test_authority_freeze_accepts_prospective_scientific_values_from_requirements() -> None:
    requirements = {
        "schema_version": "phase13_authority_requirements_v1",
        "authority_hashes": {"protocol": "1" * 64, "experiment_design": "2" * 64},
        "registry_kinds": ["execution"],
        "parameter_names": ["H_run", "backbone", "baselines", "rag_corpus", "tasks"],
    }

    freeze = parse_authority_freeze(
        json.dumps(_freeze()).encode(),
        parse_authority_requirements(json.dumps(requirements).encode()),
    )

    assert freeze.parameters["backbone"] == "prospective-backbone"
    assert freeze.parameters["H_run"] == 7


def test_authority_freeze_rejects_values_not_bound_by_requirements() -> None:
    requirements = {
        "schema_version": "phase13_authority_requirements_v1",
        "authority_hashes": {"protocol": "1" * 64},
        "registry_kinds": ["execution"],
        "parameter_names": ["H_run", "backbone", "baselines", "rag_corpus", "tasks"],
    }

    with pytest.raises(Phase13AuthorityError, match="AUTHORITY_SET_MISMATCH"):
        parse_authority_freeze(
            json.dumps(_freeze()).encode(),
            parse_authority_requirements(json.dumps(requirements).encode()),
        )
