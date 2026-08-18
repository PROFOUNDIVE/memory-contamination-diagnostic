from __future__ import annotations

import hashlib
import json

import pytest

from memcontam.experiment.phase12.filter_challenge.mft_state_models import JsonValue
from memcontam.readiness.phase13_execution_contract import (
    CORE_MAIN_REGISTRY,
    Phase13ExecutionError,
    parse_execution_registry,
    validate_execution_closure,
)
from memcontam.readiness.phase13_route_capacity import recompute_capacity


def _registry() -> dict[str, JsonValue]:
    payload: dict[str, JsonValue] = {
        "schema_version": "phase13_execution_registry_v1",
        "registry_id": "prospective-execution",
        "authority_freeze_id": "prospective-freeze",
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


def test_core_main_registry_matches_final_phase13_authority() -> None:
    registry = CORE_MAIN_REGISTRY

    assert registry.tasks == (
        "game24",
        "math_equation_balancer",
        "word_sorting",
        "mmlu_pro_engineering",
        "mmlu_pro_physics",
        "gpqa_diamond",
    )
    assert registry.memory_baselines == (
        "fh_bounded",
        "rag_frozen",
        "bot_style",
        "reflexion_style",
        "dc_rs",
    )
    assert registry.arms == ("clean", "correct", "irrelevant", "contam")
    assert registry.nomem_policy == "singleton_per_task_seed"
    assert registry.backbone_id == "gpt-5.6-luna"
    assert (registry.H_run, registry.H_primary, registry.primary_analysis_window_id) == (
        50,
        50,
        "core_prefix_50",
    )
    assert registry.capacity_unit == "registered_serialized_tokens"
    assert registry.capacity_law_id == "luna_common_visible_memory_capacity_v1"
    assert registry.capacity_formula == "min(B_FH_feasible,B_DC_feasible)"
    assert registry.dc_rs_capacity_binding == "L_DC_tokens=B_mem_tokens"
    assert registry.writer_max_output_tokens == 8192
    assert (registry.preferred_seed_count, registry.fallback_seed_count) == (12, 10)
    assert registry.rag_deadline == "2026-08-22T18:00:00+09:00"
    assert registry.rag_deadline_policy == "all_three_or_prospective_extension"
    assert registry.authority_sha256 == (
        "34f63f37a49e92607c78ced038c4c70b4c9d5e3fa8fc57d6e97de1ee79db59a8",
        "0bacce62718a93c14ce4da0c1b426e3823b75cf70b362f8f9a0632e83f4166c1",
        "eac32c3eb0de5d48cb73a2dbd6cc943d01001650e6262d99aef49e1131071ed1",
        "880ba261285758b8c5fea697a105690ffd1c0e4b0b6ab8409673f8408d457b11",
        "624f3e9a198b7bdd14aa5fdfb3883eb141b5a5def8ef5ff4fff59667ca233280",
    )


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


def test_execution_registry_rejects_prefix_nominal_calls_above_maximum() -> None:
    payload = _registry()
    capacity = payload["capacity"]
    assert isinstance(capacity, dict)
    capacity["prefix_nominal_calls_per_seed"] = 3
    capacity["prefix_maximum_calls_per_seed"] = 2
    unsigned = dict(payload)
    unsigned.pop("registry_hash")
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    payload["registry_hash"] = hashlib.sha256(canonical).hexdigest()

    with pytest.raises(Phase13ExecutionError, match="PREFIX_CALL_LIMIT_INVALID"):
        parse_execution_registry(json.dumps(payload).encode())


def test_execution_closure_binds_freeze_id_registry_id_path_and_hash(tmp_path) -> None:
    registry_payload = _registry()
    registry_raw = json.dumps(registry_payload, sort_keys=True).encode()
    registry_path = tmp_path / "registries" / "execution.json"
    registry_path.parent.mkdir()
    registry_path.write_bytes(registry_raw)
    freeze_payload: dict[str, JsonValue] = {
        "schema_version": "phase13_authority_freeze_v1",
        "freeze_id": "prospective-freeze",
        "authorities": [
            {
                "role": "protocol",
                "artifact": {"path": "authority/protocol.json", "sha256": "1" * 64},
            }
        ],
        "registries": [
            {
                "kind": "execution",
                "registry_id": "prospective-execution",
                "artifact": {
                    "path": "registries/execution.json",
                    "sha256": hashlib.sha256(registry_raw).hexdigest(),
                },
            }
        ],
        "parameters": {"backbone": "prospective-backbone"},
    }
    freeze_unsigned = json.dumps(
        freeze_payload, sort_keys=True, separators=(",", ":")
    ).encode()
    freeze_payload["closure_hash"] = hashlib.sha256(freeze_unsigned).hexdigest()
    requirements = {
        "schema_version": "phase13_authority_requirements_v1",
        "authority_hashes": {"protocol": "1" * 64},
        "registry_kinds": ["execution"],
        "parameter_names": ["backbone"],
    }

    registry = validate_execution_closure(
        json.dumps(freeze_payload).encode(),
        json.dumps(requirements).encode(),
        tmp_path,
    )

    assert registry.registry_id == "prospective-execution"
