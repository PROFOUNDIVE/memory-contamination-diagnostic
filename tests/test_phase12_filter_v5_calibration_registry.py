from __future__ import annotations

from memcontam.experiment.phase12.filter_challenge.registry_calibration import (
    bct_schedule,
    screening_schedule,
)


def test_calibration_schedules_are_deterministic_and_cover_native_stages() -> None:
    probes = {
        "game24": ("fv5-cal-game24-001", "fv5-cal-game24-002"),
        "math_equation_balancer": ("fv5-cal-meb-001", "fv5-cal-meb-002"),
        "word_sorting": ("fv5-cal-words-001", "fv5-cal-words-002"),
    }

    screening = screening_schedule(
        {task: tuple(f"{probe}-x{index}" for probe in values for index in range(1, 4)) for task, values in probes.items()}
    )
    bct = bct_schedule(probes)

    assert len(screening) == 90
    assert all(item.side == "control" for item in screening)
    assert len(bct) == 480
    assert tuple(item.task for item in bct[:8]) == ("game24",) * 8
