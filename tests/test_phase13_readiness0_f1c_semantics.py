from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from memcontam.readiness.phase13_readiness0_f1c_report import (
    F1CReportError,
    validate_f1c_report,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "data/phase13/main/mr_p4/readiness0_f1c_report_v1.json"


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _tampered(mutator: Callable[[dict[str, Any]], None]) -> bytes:
    report = json.loads(REPORT.read_bytes())
    mutator(report)
    report["runtime"]["runtime_hash"] = _hash(
        {key: value for key, value in report["runtime"].items() if key != "runtime_hash"}
    )
    report["report_hash"] = _hash(
        {key: value for key, value in report.items() if key != "report_hash"}
    )
    return json.dumps(report, separators=(",", ":")).encode()


def _row(index: int, field: str, value: Any) -> Callable[[dict[str, Any]], None]:
    def mutate(report: dict[str, Any]) -> None:
        report["rows"][index][field] = deepcopy(value)

    return mutate


@pytest.mark.parametrize(
    "mutator",
    (
        _row(0, "row_id", "0" * 64),
        _row(0, "query_source", "data/phase13/main/word_sorting_main_v1.jsonl"),
        _row(0, "query_sha256", "0" * 64),
        _row(0, "candidate_ids", ["wrong", *json.loads(REPORT.read_bytes())["rows"][0]["candidate_ids"][1:]]),
        _row(0, "scores", [-0.99, *json.loads(REPORT.read_bytes())["rows"][0]["scores"][1:]]),
        _row(0, "ranks", list(reversed(json.loads(REPORT.read_bytes())["rows"][0]["ranks"]))),
        _row(0, "selected_ids", ["wrong"]),
        _row(0, "tie_policy", "input_order"),
        _row(0, "corpus_identity_sha256", "0" * 64),
        _row(12, "state_identity_sha256", "0" * 64),
        _row(12, "index_identity_sha256", "0" * 64),
        _row(0, "top_k", 2),
        _row(12, "threshold", None),
        _row(0, "source_span_ids", ["wrong"]),
        _row(0, "source_span_join_sha256", "0" * 64),
        lambda report: report["runtime"].update({"device": "cuda"}),
        lambda report: report["runtime"].update({"network_attempts": 1}),
    ),
)
def test_f1c_report_rejects_internally_rehashed_semantic_tampering(
    mutator: Callable[[dict[str, Any]], None],
) -> None:
    with pytest.raises(F1CReportError):
        validate_f1c_report(_tampered(mutator), ROOT)


def test_f1c_report_accepts_committed_semantic_evidence() -> None:
    report = validate_f1c_report(REPORT.read_bytes(), ROOT)

    assert len(report.rows) == 52
