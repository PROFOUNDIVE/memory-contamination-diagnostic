from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "generate_phase12_contract_fixtures.py"
FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "phase12"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_check_verifies_committed_task_zero_fixture_bytes() -> None:
    result = _run("--check")

    assert result.returncode == 0, result.stderr


def test_check_reports_the_exact_mutated_fixture_without_changing_committed_bytes(
    tmp_path: Path,
) -> None:
    fixture_root = tmp_path / "phase12"
    shutil.copytree(FIXTURE_ROOT, fixture_root)
    committed = (FIXTURE_ROOT / "FX-SCHEMA-001.json").read_bytes()
    mutated = fixture_root / "FX-SCHEMA-001.json"
    mutated.write_bytes(mutated.read_bytes().replace(b"phase12", b"phase13", 1))

    result = _run("--check", "--fixture-root", str(fixture_root))

    assert result.returncode == 1
    assert "FIXTURE_HASH_MISMATCH:FX-SCHEMA-001.json" in result.stderr
    assert (FIXTURE_ROOT / "FX-SCHEMA-001.json").read_bytes() == committed


def test_fixture_writes_require_an_explicit_plan_revision_flag(tmp_path: Path) -> None:
    fixture_root = tmp_path / "phase12"
    shutil.copytree(FIXTURE_ROOT, fixture_root)
    mutated = fixture_root / "FX-CONFIG-001.json"
    mutated.write_bytes(mutated.read_bytes().replace(b"phase12", b"phase13", 1))

    refused = _run("--write", "--fixture-root", str(fixture_root))

    assert refused.returncode == 2
    assert "PLAN_REVISION_ONLY_REQUIRED" in refused.stderr
    assert mutated.read_bytes() != (FIXTURE_ROOT / "FX-CONFIG-001.json").read_bytes()

    revised = _run("--write", "--plan-revision-only", "--fixture-root", str(fixture_root))

    assert revised.returncode == 0, revised.stderr
    assert mutated.read_bytes() == (FIXTURE_ROOT / "FX-CONFIG-001.json").read_bytes()
    assert _run("--check", "--fixture-root", str(fixture_root)).returncode == 0


def test_fixture_writes_are_forbidden_at_the_committed_root() -> None:
    result = _run("--write", "--plan-revision-only", "--fixture-root", str(FIXTURE_ROOT))

    assert result.returncode == 2
    assert "UNAUTHORIZED_FIXTURE_WRITE" in result.stderr


def test_check_rejects_fixture_without_lf_trailing_newline(tmp_path: Path) -> None:
    fixture_root = tmp_path / "phase12"
    shutil.copytree(FIXTURE_ROOT, fixture_root)
    path = fixture_root / "FX-SCHEMA-001.json"
    path.write_bytes(path.read_bytes().rstrip(b"\n"))

    result = _run("--check", "--fixture-root", str(fixture_root))

    assert result.returncode == 1
    assert "FIXTURE_TRAILING_NEWLINE_MISSING:FX-SCHEMA-001.json" in result.stderr


def test_check_forbids_the_plan_revision_flag() -> None:
    result = _run("--check", "--plan-revision-only")

    assert result.returncode == 2
    assert "PLAN_REVISION_ONLY_FLAG_FORBIDDEN" in result.stderr
