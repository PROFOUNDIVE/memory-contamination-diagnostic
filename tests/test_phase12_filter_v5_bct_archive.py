from __future__ import annotations

import hashlib
import json

import pytest

from memcontam.experiment.phase12.filter_challenge.bct_archive import (
    LedgerError,
    append_archive_record,
    validate_live_archive,
)


def test_partial_archive_separates_public_and_audit_records(tmp_path) -> None:
    root = tmp_path / "run"
    append_archive_record(root, "public", {"run_id": "run-001", "status": "planned"})
    append_archive_record(
        root,
        "audit",
        {"run_id": "run-001", "status": "planned", "candidate_role": "certified_false"},
    )

    report = validate_live_archive(root)

    assert report.valid is True
    assert (root / "public.jsonl").read_text(encoding="utf-8").find("candidate_role") == -1


def test_archive_records_are_chained_and_reconcile_each_run(tmp_path) -> None:
    root = tmp_path / "run"
    append_archive_record(root, "public", {"run_id": "run-001", "status": "completed"})
    append_archive_record(root, "audit", {"run_id": "run-001", "status": "completed"})
    public = json.loads((root / "public.jsonl").read_text(encoding="utf-8"))

    assert public["sequence"] == 1
    assert public["previous_hash"] == "0" * 64
    assert public["raw_byte_start"] == 0
    assert public["raw_byte_end"] == len((root / "public.jsonl").read_bytes())
    assert validate_live_archive(root).valid is True


def test_archive_validator_rejects_tamper_stale_ranges_and_unreconciled_run_ids(tmp_path) -> None:
    root = tmp_path / "run"
    append_archive_record(root, "public", {"run_id": "run-001", "status": "completed"})
    append_archive_record(root, "audit", {"run_id": "run-001", "status": "completed"})
    path = root / "public.jsonl"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["raw_byte_end"] += 1
    signed = {key: value for key, value in payload.items() if key != "record_hash"}
    payload["record_hash"] = hashlib.sha256(
        json.dumps(signed, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")

    assert validate_live_archive(root).valid is False


def test_archive_rejects_hidden_public_labels_and_invalid_or_reused_runs(tmp_path) -> None:
    root = tmp_path / "run"
    with pytest.raises(LedgerError, match="AUDIT_FIELD_IN_PUBLIC_STREAM"):
        append_archive_record(
            root,
            "public",
            {"run_id": "run-001", "status": "planned", "nested": {"candidate_role": "false"}},
    )
    append_archive_record(root, "public", {"run_id": "run-001", "status": "completed", "valid": True})
    append_archive_record(root, "public", {"run_id": "run-001", "status": "completed"})
    with pytest.raises(LedgerError, match="LIVE_ARCHIVE_STATUS_INVALID"):
        append_archive_record(root, "audit", {"run_id": "run-001", "status": "not-a-status"})
    append_archive_record(root, "audit", {"run_id": "run-001", "status": "completed"})

    assert validate_live_archive(root).valid is False


def test_archive_validator_rejects_rehashed_invalid_status(tmp_path) -> None:
    root = tmp_path / "run"
    for stream in ("public", "audit"):
        append_archive_record(root, stream, {"run_id": "run-001", "status": "completed"})
        path = root / f"{stream}.jsonl"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["status"] = "notvalid!"
        unsigned = {key: value for key, value in payload.items() if key != "record_hash"}
        payload["record_hash"] = hashlib.sha256(
            json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")

    assert validate_live_archive(root).valid is False
