from __future__ import annotations

from dataclasses import dataclass
from typing import assert_never


@dataclass(frozen=True, slots=True)
class DeterministicAuthoritySyncComplete:
    pass


@dataclass(frozen=True, slots=True)
class CalibrationV2ExternalBlock:
    pass


@dataclass(frozen=True, slots=True)
class CalibrationV2Invalidated:
    pass


@dataclass(frozen=True, slots=True)
class CalibrationV2Completed:
    pass


@dataclass(frozen=True, slots=True)
class MainExecutionForbidden:
    pass


Terminal = (
    DeterministicAuthoritySyncComplete
    | CalibrationV2ExternalBlock
    | CalibrationV2Invalidated
    | CalibrationV2Completed
    | MainExecutionForbidden
)


def render_terminal(terminal: Terminal) -> str:
    match terminal:
        case DeterministicAuthoritySyncComplete():
            return "DETERMINISTIC_AUTHORITY_SYNC_COMPLETE"
        case CalibrationV2ExternalBlock():
            return "CALIBRATION_V2_EXTERNAL_BLOCK"
        case CalibrationV2Invalidated():
            return "CALIBRATION_V2_INVALIDATED"
        case CalibrationV2Completed():
            return "CALIBRATION_V2_COMPLETED"
        case MainExecutionForbidden():
            return "MAIN_A_EXECUTION_FORBIDDEN"
        case unreachable:
            assert_never(unreachable)
