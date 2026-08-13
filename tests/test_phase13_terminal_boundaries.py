from __future__ import annotations

from memcontam.readiness.phase13_terminal import (
    CalibrationV2Completed,
    CalibrationV2ExternalBlock,
    CalibrationV2Invalidated,
    DeterministicAuthoritySyncComplete,
    MainExecutionForbidden,
    render_terminal,
)


def test_terminal_union_renders_every_boundary() -> None:
    terminals = (
        DeterministicAuthoritySyncComplete(),
        CalibrationV2ExternalBlock(),
        CalibrationV2Invalidated(),
        CalibrationV2Completed(),
        MainExecutionForbidden(),
    )

    assert tuple(render_terminal(terminal) for terminal in terminals) == (
        "DETERMINISTIC_AUTHORITY_SYNC_COMPLETE",
        "CALIBRATION_V2_EXTERNAL_BLOCK",
        "CALIBRATION_V2_INVALIDATED",
        "CALIBRATION_V2_COMPLETED",
        "MAIN_A_EXECUTION_FORBIDDEN",
    )
