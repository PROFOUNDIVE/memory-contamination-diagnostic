from __future__ import annotations

import hashlib
import json

import pytest

from memcontam.experiment.phase12.filter_challenge.mft_state_models import JsonValue
from memcontam.readiness.phase13_execution_contract import (
    Phase13ExecutionError,
    parse_execution_registry,
)
from memcontam.readiness.phase13_route_capacity import recompute_capacity


def _registry() -> dict[str, JsonValue]:
    payload: dict[str, JsonValue] = {
        "schema_version": "phase13_execution_registry_v1",
        "registry_id": "prospective-execution",
        "authority_freeze_sha256": "a" * 64,
        "backbone_id": "prospective-backbone",
        "H_run": 7,
        "tasks": ["task-a", "task-b"],
        "baselines": ["memory-a"],
        "arms": ["clean", "contam"],
        "rag_corpus": {"path": "corpora/prospective.jsonl", "sha256": "b" * 64},
        "execution_owner_id": "future-owner",
        "templates": [
            {
                "template_id": "task-a-template",
                "task": "task-a",
                "baseline": "memory-a",
                "arm": "clean",
                "nominal_semantic_calls_per_trial": 2,
                "maximum_semantic_calls_per_trial": 3,
            },
            {
                "template_id": "task-b-template",
                "task": "task-b",
                "baseline": "memory-a",
                "arm": "contam",
                "nominal_semantic_calls_per_trial": 4,
                "maximum_semantic_calls_per_trial": 5,
            },
        ],
        "capacity": {
            "prefix_nominal_calls_per_seed": 1,
            "prefix_maximum_calls_per_seed": 2,
            "reserve_percent": 10,
            "maximum_transport_attempts_per_semantic_call": 3,
            "maximum_input_tokens_per_transport_attempt": 100,
            "maximum_output_tokens_per_transport_attempt": 50,
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["registry_hash"] = hashlib.sha256(canonical).hexdigest()
    return payload


def test_execution_registry_and_capacity_are_driven_by_prospective_inputs() -> None:
    registry = parse_execution_registry(json.dumps(_registry()).encode())

    plan = recompute_capacity(registry, {"task-a": 3, "task-b": 2})

    assert registry.backbone_id == "prospective-backbone"
    assert registry.H_run == 7
    assert plan.nominal_semantic_calls == 103
    assert plan.raw_maximum_semantic_calls == 143
    assert plan.reserved_semantic_calls == 158
    assert plan.reserved_transport_attempts == 474


def test_execution_registry_rejects_templates_outside_declared_dimensions() -> None:
    payload = _registry()
    templates = payload["templates"]
    assert isinstance(templates, list)
    first = templates[0]
    assert isinstance(first, dict)
    first["task"] = "undeclared"
    payload["registry_hash"] = "0" * 64

    with pytest.raises(Phase13ExecutionError, match="TEMPLATE_DIMENSION_UNDECLARED"):
        parse_execution_registry(json.dumps(payload).encode())
