from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_phase12_aggregate_manifest.py"


def test_nested_duplicate_profile_is_rejected_before_manifest_parsing(tmp_path: Path) -> None:
    records = tmp_path / "records.json"
    run_manifest = tmp_path / "run-manifest.jsonl"
    output = tmp_path / "aggregate.jsonl"
    records.write_text(
        '[{"profile":"local_rootless_non_authoritative","profile":"authoritative"}]',
        encoding="utf-8",
    )
    run_manifest.write_text("", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input",
            str(records),
            "--run-manifest",
            str(run_manifest),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "ROOTLESS_PROFILE_FORBIDDEN" in result.stderr
    assert not output.exists()


def test_nested_aggregate_profile_decoy_keeps_legacy_validation_reason(tmp_path: Path) -> None:
    records = tmp_path / "records.json"
    run_manifest = tmp_path / "run-manifest.jsonl"
    output = tmp_path / "aggregate.jsonl"
    records.write_text(
        '[{"metadata":{"profile":"local_rootless_non_authoritative"}}]', encoding="utf-8"
    )
    run_manifest.write_text("", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input",
            str(records),
            "--run-manifest",
            str(run_manifest),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "ROOTLESS_PROFILE_FORBIDDEN" not in result.stderr
    assert not output.exists()
