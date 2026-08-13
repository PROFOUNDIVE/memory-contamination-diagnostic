from __future__ import annotations

import copy
import hashlib
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from memcontam.readiness.phase13_execution_contract import (
    Phase13ExecutionError,
    load_execution_registry,
    parse_execution_registry,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data/phase13/authority/execution_registry_v1.json"
PARTITIONS = ROOT / "data/phase13/calibration_v2/seed_partition_registry_v1.json"
PARTITION_HASH = "a31b731244f5c56b4aafa5ed83bbe720c8623563cfbd800e7a478a0025aff4ba"
STREAM_HASHES = {
    "game24": "6f474aa24f1d33e62ccfdfafddb6e6656c45a66099729668bbfb1bc6e58356ea",
    "math_equation_balancer": "ea4702092a9964104b1852dc9794b6d84c8aa4faf4df7f490826b2c0b6ba19cb",
    "word_sorting": "f032f49cf712757266d4d23cd00753fc7d986cd27627ba218c225ba04e660394",
}


def _payload() -> dict[str, Any]:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def _resign(payload: dict[str, Any]) -> bytes:
    unsigned = copy.deepcopy(payload)
    unsigned.pop("registry_hash", None)
    payload["registry_hash"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return json.dumps(payload, sort_keys=True).encode()


def test_committed_execution_registry_binds_todo3_streams_and_execution_contract() -> None:
    registry = load_execution_registry(REGISTRY, ROOT)

    assert hashlib.sha256(PARTITIONS.read_bytes()).hexdigest() == PARTITION_HASH
    assert registry.source_partition.sha256 == PARTITION_HASH
    assert {stream.task: stream.calibration_sha256 for stream in registry.task_streams} == STREAM_HASHES
    assert (registry.timing.L_min, registry.timing.tau_star, registry.timing.H_run) == (1, 2, 10)
    assert (registry.timing.absolute_trial_start, registry.timing.absolute_trial_end) == (2, 11)
    assert (registry.timing.event_time_start, registry.timing.event_time_end) == (0, 9)
    assert all(len(stream.suffixes) == 12 for stream in registry.task_streams)
    assert all(suffix.suffix_length == 10 for stream in registry.task_streams for suffix in stream.suffixes)
    assert {arm.arm_key for arm in registry.memory_arms} == {
        "Clean", "Correct", "Irrelevant", "Contam"
    }
    assert registry.nomem.arm_key == "star_NoMem"
    assert {capacity.baseline for capacity in registry.native_capacities} == {
        "fh_bounded", "rag_frozen", "bot_style", "reflexion_style"
    }


Mutation = Callable[[dict[str, Any]], None]


def _suffix_order(payload: dict[str, Any]) -> None:
    suffixes = payload["task_streams"][0]["suffixes"]
    suffixes[0], suffixes[1] = suffixes[1], suffixes[0]


def _horizon(payload: dict[str, Any]) -> None:
    payload["timing"]["H_run"] = 9


def _event_range(payload: dict[str, Any]) -> None:
    payload["timing"]["event_time_end"] = 8


def _coordinated_trial_shift(payload: dict[str, Any]) -> None:
    payload["timing"]["absolute_trial_start"] = 3
    payload["timing"]["absolute_trial_end"] = 12


def _coordinated_event_shift(payload: dict[str, Any]) -> None:
    payload["timing"]["event_time_start"] = 1
    payload["timing"]["event_time_end"] = 10


def _tau_rule_drift(payload: dict[str, Any]) -> None:
    payload["timing"]["tau_star_rule"] = "maximum_static_feasible_position"


def _minimum_stream_length_drift(payload: dict[str, Any]) -> None:
    payload["timing"]["minimum_stream_length"] = 12


def _timing_scalar_drift(payload: dict[str, Any]) -> None:
    payload["timing"]["L_min"] = 2


def _primary_id(payload: dict[str, Any]) -> None:
    payload["primary_analysis_window_id"] = "accuracy-h2-sensitivity"


def _owner(payload: dict[str, Any]) -> None:
    payload["call_components"][0]["owner_id"] = "execution-owner-v1"


def _capacity(payload: dict[str, Any]) -> None:
    payload["operator_capacity"]["maximum_cost_microusd"]["value"] = 1


def _source_hash(payload: dict[str, Any]) -> None:
    payload["source_partition"]["sha256"] = "0" * 64


def _bare_h(payload: dict[str, Any]) -> None:
    payload["timing"]["H"] = 10


def _duplicate_owner(payload: dict[str, Any]) -> None:
    payload["execution_owner_id"] = payload["prefix_owner_id"]
    payload["call_components"][1]["owner_id"] = payload["prefix_owner_id"]


def _component_drift(payload: dict[str, Any]) -> None:
    payload["call_components"][0]["nominal_calls_per_activation"] = 7


def _native_capacity_drift(payload: dict[str, Any]) -> None:
    payload["native_capacities"][0]["configured_limit"] = 9999


def _model_drift(payload: dict[str, Any]) -> None:
    payload["identities"]["model_snapshot_id"] = "gpt-4o-mutated"


def _arm_drift(payload: dict[str, Any]) -> None:
    payload["memory_arms"][0]["candidate_or_control_id"] = "mutated-clean"


def _duplicate_component(payload: dict[str, Any]) -> None:
    payload["call_components"].append(copy.deepcopy(payload["call_components"][0]))


def _native_policy_drift(payload: dict[str, Any]) -> None:
    payload["native_capacities"][0]["transition_policy"] = "bounded_template_eviction"


def _template_drift(payload: dict[str, Any]) -> None:
    payload["execution_templates"][0]["main_seed_multiplicity"] = 11


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (_suffix_order, "SUFFIX_ORDER_INVALID"),
        (_horizon, "HORIZON_INVALID"),
        (_event_range, "EVENT_RANGE_INVALID"),
        (_coordinated_trial_shift, "EVENT_RANGE_INVALID"),
        (_coordinated_event_shift, "EVENT_RANGE_INVALID"),
        (_tau_rule_drift, "MALFORMED_REGISTRY"),
        (_minimum_stream_length_drift, "HORIZON_INVALID"),
        (_timing_scalar_drift, "HORIZON_INVALID"),
        (_primary_id, "PRIMARY_WINDOW_INVALID"),
        (_owner, "OWNER_BINDING_INVALID"),
        (_capacity, "CAPACITY_CONTRACT_INVALID"),
        (_source_hash, "SOURCE_AUTHORITY_HASH_MISMATCH"),
        (_bare_h, "BARE_H_PROHIBITED"),
        (_duplicate_owner, "OWNER_IDENTITY_INVALID"),
        (_component_drift, "CALL_COMPONENT_INVALID"),
        (_native_capacity_drift, "NATIVE_CAPACITY_INVALID"),
        (_model_drift, "EXECUTION_IDENTITY_INVALID"),
        (_arm_drift, "ARM_REGISTRY_INVALID"),
        (_duplicate_component, "CALL_COMPONENT_INVALID"),
        (_native_policy_drift, "NATIVE_CAPACITY_INVALID"),
        (_template_drift, "MALFORMED_REGISTRY"),
    ],
)
def test_resigned_execution_mutations_fail_with_bounded_codes(
    mutate: Mutation, code: str
) -> None:
    payload = _payload()
    mutate(payload)

    with pytest.raises(Phase13ExecutionError) as caught:
        parse_execution_registry(_resign(payload), ROOT)

    assert caught.value.code == code


def test_coordinated_resigned_partition_rewrite_is_rejected(tmp_path: Path) -> None:
    authority_root = tmp_path / "authority"
    source = ROOT / "data/phase13/calibration_v2/seed_partition_registry_v1.json"
    target = authority_root / "data/phase13/calibration_v2/seed_partition_registry_v1.json"
    target.parent.mkdir(parents=True)
    shutil.copy2(source, target)
    partition = json.loads(target.read_text())
    trajectory = partition["tasks"]["game24"]["trajectories"][0]
    trajectory["ordered_sample_ids"].reverse()
    trajectory["ordered_stream_sha256"] = hashlib.sha256(
        (json.dumps({"ordered_sample_ids": trajectory["ordered_sample_ids"]}, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()
    target.write_text(json.dumps(partition, sort_keys=True, indent=2) + "\n")
    payload = _payload()
    payload["source_partition"]["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    payload["task_streams"][0]["suffixes"][0]["source_ordered_stream_sha256"] = trajectory[
        "ordered_stream_sha256"
    ]

    with pytest.raises(Phase13ExecutionError) as caught:
        parse_execution_registry(_resign(payload), authority_root)

    assert caught.value.code == "SOURCE_AUTHORITY_HASH_MISMATCH"
