from __future__ import annotations

from memcontam.experiment.phase12.filter_challenge.bct_archive import (
    append_archive_record,
    validate_live_archive,
)


def test_partial_archive_separates_public_and_audit_records(tmp_path) -> None:
    root = tmp_path / "run"
    append_archive_record(root, "public", {"run_id": "run-001", "status": "planned"})
    append_archive_record(root, "audit", {"candidate_role": "certified_false"})

    report = validate_live_archive(root)

    assert report.valid is True
    assert (root / "public.jsonl").read_text(encoding="utf-8").find("candidate_role") == -1
