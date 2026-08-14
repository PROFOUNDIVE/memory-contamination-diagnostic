from __future__ import annotations

import subprocess
import sys


def test_phase13_help_exposes_only_prospective_generic_validators_for_new_surface() -> None:
    result = subprocess.run(
        (sys.executable, "-m", "memcontam.cli", "phase13", "--help"),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "validate-authority-freeze" in result.stdout
    assert "validate-execution-registry" in result.stdout
    assert "validate-provenance" in result.stdout
    assert "calibration-v2" not in result.stdout
