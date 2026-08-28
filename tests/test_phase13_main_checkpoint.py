from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from memcontam.readiness.phase13_main_checkpoint import (
    Phase13MainCheckpointError,
    _tau_star,
    production_identity_from_checkpoint,
    validate_main_checkpoint_package,
)


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "data/phase13/main/mr_p4"
TASKS = {
    "game24",
    "math_equation_balancer",
    "word_sorting",
    "mmlu_pro_engineering",
    "mmlu_pro_physics",
}


def test_checkpoint_package_freezes_complete_task_local_orders_and_suffixes() -> None:
    report = validate_main_checkpoint_package(PACKAGE, ROOT)

    assert set(report.tasks) == TASKS
    assert report.seed_ids == tuple(range(10))
    assert report.tau_star_by_task == {task: 2 for task in TASKS}
    assert report.suffix_lengths == {task: 50 for task in TASKS}


def test_tau_star_is_derived_from_explicit_feasibility_inputs() -> None:
    assert _tau_star(
        task_stream_length=53,
        minimum_clean_prefix_length=2,
        execution_horizon=50,
        static_route_constraints_satisfied=True,
    ) == 3

    with pytest.raises(Phase13MainCheckpointError, match="MAIN_CHECKPOINT_ROUTE_INFEASIBLE"):
        _tau_star(
            task_stream_length=51,
            minimum_clean_prefix_length=2,
            execution_horizon=50,
            static_route_constraints_satisfied=True,
        )


def test_checkpoint_orders_follow_the_registered_raw_digest_law() -> None:
    payload = json.loads((PACKAGE / "task_seed_orders_v1.json").read_text(encoding="utf-8"))

    for task, task_row in payload["tasks"].items():
        for seed_row in task_row["seeds"]:
            seed = seed_row["seed"]
            expected = sorted(
                task_row["sample_ids"],
                key=lambda sample_id: hashlib.sha256(
                    f"sha256_task_seed_v1\0{task}\0{seed}\0{sample_id}".encode()
                ).digest(),
            )
            assert seed_row["ordered_sample_ids"] == expected


def test_production_identity_consumes_the_frozen_50_item_suffix() -> None:
    registry = json.loads(
        (PACKAGE / "main_a_common_checkpoint_registry_v1.json").read_text(encoding="utf-8")
    )
    expected = registry["tasks"]["game24"]["seeds"][0]

    identity = production_identity_from_checkpoint(
        PACKAGE,
        ROOT,
        task="game24",
        trajectory_seed=0,
        execution_template_id="game24|fh_bounded|clean",
        registration_packet_sha256="0" * 64,
    )

    assert identity.ordered_sample_ids_sha256 == expected["suffix_sample_ids_sha256"]
    assert identity.checkpoint_registry_sha256 == hashlib.sha256(
        (PACKAGE / "main_a_common_checkpoint_registry_v1.json").read_bytes()
    ).hexdigest()


def test_checkpoint_validator_rejects_self_consistent_noncanonical_order(tmp_path: Path) -> None:
    package = tmp_path / "mr_p4"
    shutil.copytree(PACKAGE, package)
    orders_path = package / "task_seed_orders_v1.json"
    orders = json.loads(orders_path.read_text(encoding="utf-8"))
    ordered = orders["tasks"]["game24"]["seeds"][0]["ordered_sample_ids"]
    ordered[0], ordered[1] = ordered[1], ordered[0]
    orders["orders_hash"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in orders.items() if key != "orders_hash"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    orders_path.write_text(json.dumps(orders), encoding="utf-8")

    with pytest.raises(Phase13MainCheckpointError, match="MAIN_CHECKPOINT_ORDER_INVALID"):
        validate_main_checkpoint_package(package, ROOT)
