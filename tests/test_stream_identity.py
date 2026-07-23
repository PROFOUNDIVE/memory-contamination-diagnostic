from __future__ import annotations

from dataclasses import fields
import importlib
import importlib.util


def test_stream_identity_isolated_by_arm_but_cross_arm_pairs_share_pair_key() -> None:
    assert importlib.util.find_spec("memcontam.baselines.contracts"), (
        "StreamIdentity and StreamPairKey require shared contracts"
    )
    contracts = importlib.import_module("memcontam.baselines.contracts")

    identity = getattr(contracts, "StreamIdentity", None)
    pair_key = getattr(contracts, "StreamPairKey", None)
    to_pair_key = getattr(contracts, "stream_pair_key", None)
    assert identity is not None
    assert pair_key is not None
    assert callable(to_pair_key)
    assert [field.name for field in fields(identity)] == [
        "run_id",
        "task_family",
        "baseline",
        "arm",
        "backbone",
    ]
    assert [field.name for field in fields(pair_key)] == [
        "run_id",
        "task_family",
        "baseline",
        "backbone",
    ]
    clean = identity("run", "game24", "full_history", "clean", "replay")
    contaminated = identity("run", "game24", "full_history", "contaminated", "replay")
    assert clean != contaminated
    assert (
        to_pair_key(clean)
        == to_pair_key(contaminated)
        == pair_key("run", "game24", "full_history", "replay")
    )
