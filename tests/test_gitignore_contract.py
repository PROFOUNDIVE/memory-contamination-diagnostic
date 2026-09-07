import json
import os
from pathlib import Path
import subprocess


def test_no_ignored_files_are_tracked() -> None:
    result = subprocess.run(
        ["git", "ls-files", "-ci", "--exclude-standard"],
        check=True,
        capture_output=True,
        env={**os.environ, "GIT_MASTER": "1", "LC_ALL": "C"},
    )

    baseline = json.loads(
        (Path(__file__).parent / "fixtures/tracked_ignored_provenance_baseline_v1.json").read_bytes()
    )
    assert baseline["schema_version"] == "tracked_ignored_provenance_baseline_v1"
    assert baseline["command"] == "git ls-files -ci --exclude-standard"
    assert result.stdout == "".join(f"{path}\n" for path in baseline["paths"]).encode("utf-8")
