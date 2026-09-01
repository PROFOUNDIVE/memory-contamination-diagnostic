from __future__ import annotations

from pathlib import Path

import pytest

from memcontam.baselines.full_history_phase12 import FullHistoryStateV3
from memcontam.experiment.phase12.runtime_registry import PHASE13_CORE_BASELINE_REGISTRY
from memcontam.memory.checkpoint_v3 import NativeState, serialize_checkpoint
from memcontam.readiness.phase13_main_live_dispatch import MainUnitDispatchOutput
from memcontam.readiness.phase13_main_production import ProductionObject
from memcontam.readiness.phase13_main_production_backend import (
    MainProductionBackend,
    MainProductionBackendError,
    OrdinaryRuntimeRequest,
    PrefixRuntimeOutput,
)


def test_orphaned_prefix_backend_output_cannot_enable_memory_consumer(tmp_path: Path) -> None:
    prefix = ProductionObject(
        0, "1" * 64, "CLEAN_PREFIX", 0, "game24", "fh_bounded", "NOT_APPLICABLE", None, 1
    )
    consumer = ProductionObject(
        1, "2" * 64, "MEMORY_BEARING", 0, "game24", "fh_bounded", "clean", prefix.unit_id, 1
    )
    entry = PHASE13_CORE_BASELINE_REGISTRY["fh_bounded"]
    state = entry.serialize_state(FullHistoryStateV3(records=[]))
    assert isinstance(state, NativeState)
    checkpoint = serialize_checkpoint(state)
    ordinary_calls = 0

    def execute_prefix(_unit: ProductionObject) -> PrefixRuntimeOutput:
        return PrefixRuntimeOutput(
            checkpoint,
            MainUnitDispatchOutput(evidence={}, provider_calls=(), realized_cost_krw=0),
        )

    def execute_ordinary(_request: OrdinaryRuntimeRequest) -> MainUnitDispatchOutput:
        nonlocal ordinary_calls
        ordinary_calls += 1
        return MainUnitDispatchOutput(evidence={}, provider_calls=(), realized_cost_krw=0)

    backend = MainProductionBackend(tmp_path, execute_prefix, execute_ordinary)
    backend(prefix)

    with pytest.raises(MainProductionBackendError, match="MAIN_PREFIX_CHECKPOINT_INVALID"):
        backend(consumer)

    assert ordinary_calls == 0
