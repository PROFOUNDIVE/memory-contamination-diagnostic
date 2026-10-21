from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import pytest

from memcontam.evaluation.phase13_historical_exposure import (
    HistoricalIntegrityError,
    checked,
    reconstruct_stages,
    validate_population,
    validate_output_path,
    summarize,
)
from memcontam.evaluation.phase13_observability_models import Phase13TrialEvidence
from memcontam.logging.schema import MethodCall

helpers = importlib.import_module("tests.phase13_observability_helpers")


def fixture(*, included: bool = False) -> tuple[Phase13TrialEvidence, tuple[MethodCall, ...]]:
    given = helpers.evidence(helpers.module(), retrieved=True, included=included, verified=1)
    spans = [helpers.span("root-b").model_copy(update={"rendered_hash": hashlib.sha256(b"x").hexdigest()})] if included else []
    given = given.model_copy(update={"target_set": given.target_set.model_copy(update={"answer_call_spans": tuple(spans)})})
    messages = [{"role": "user", "content": "x"}]
    request_hash = hashlib.sha256(json.dumps(messages, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    call = MethodCall(call_id="answer-1", stage="bot_instantiate_solve", messages=messages,
                      raw_response="no target text", model="fixture", source_spans=spans,
                      provider_request_contract={"input_sha256": request_hash})
    return given, (call,)


def test_context_when_answer_does_not_repeat_target() -> None:
    given, calls = fixture(included=True)
    result = reconstruct_stages(given, calls, ("root-b",))
    assert result["actual_answer_context"]["value"] is True


def test_missing_answer_when_retrieval_is_known() -> None:
    given, _calls = fixture()
    result = reconstruct_stages(given, (), ("root-b",))
    assert result["retrieved"]["value"] is True
    assert result["actual_answer_context"] == {
        "status": "unknown", "value": None, "reason": "MISSING_ACTUAL_ANSWER_CALL",
        "missing_artifact": "answer-1", "eligible": True,
    }


@pytest.mark.parametrize("mutation", ["request", "span", "duplicate"])
def test_reject_integrity_when_call_is_corrupt(mutation: str) -> None:
    given, calls = fixture(included=True)
    call = calls[0]
    if mutation == "request":
        calls = (call.model_copy(update={"provider_request_contract": {"input_sha256": "0" * 64}}),)
    elif mutation == "span":
        calls = (call.model_copy(update={"source_spans": [call.source_spans[0].model_copy(update={"rendered_hash": "0" * 64})]}),)
    else:
        calls = (call, call)
    with pytest.raises(HistoricalIntegrityError):
        reconstruct_stages(given, calls, ("root-b",))


@pytest.mark.parametrize("status,expected", [("unavailable", None), ("exact", False), ("approximate", None)])
@pytest.mark.parametrize("parent_call", [None, "curator-1"])
def test_parentage_when_strategy_is_visible(status: str, expected: bool | None, parent_call: str | None) -> None:
    given, calls = fixture(included=True)
    raw = given.model_dump(mode="json")
    raw["baseline"] = "dc_rs"
    raw["context"]["final_entry_ids"] = ["strategy"]
    raw["lineage"].append({"entry_id": "strategy", "lineage_status": status})
    strategy_span = calls[0].source_spans[0].model_copy(update={
        "entry_id": "strategy", "injected_root_ids": [], "direct_parent_ids": [],
        "is_target_contamination": False, "lineage_status": status,
        "parent_call_id": parent_call,
    })
    raw["target_set"]["answer_call_spans"] = []
    given = Phase13TrialEvidence.model_validate_json(json.dumps(raw))
    calls = (calls[0].model_copy(update={"source_spans": [strategy_span]}),)
    result = reconstruct_stages(given, calls, ("root-b",))
    assert result["strategy_visible"]["value"] is True
    assert result["derivation"]["value"] is expected
    assert result["actual_answer_context"]["value"] is expected
    assert result["registered_z"]["value"] is expected


def test_known_zero_when_retrieved_but_absent_from_request() -> None:
    given, calls = fixture()
    result = reconstruct_stages(given, calls, ("root-b",))
    assert result["retrieved"]["value"] is True
    assert result["actual_answer_context"]["value"] is False
    assert result["registered_z"]["value"] is False


def test_reject_join_when_context_trial_is_wrong() -> None:
    given, calls = fixture()
    assert given.context is not None
    given = given.model_copy(update={"context": given.context.model_copy(update={"trial_id": "wrong"})})
    with pytest.raises(ValueError):
        reconstruct_stages(given, calls, ("root-b",))


@pytest.mark.parametrize("duplicate", [False, True])
def test_reject_population_when_registered_suffix_is_truncated_or_duplicated(duplicate: bool) -> None:
    rows = [{"population_key": "cohort:trial", "cell_key": "cell", "sample_id": "a", "position": 1}]
    cells = [{"cell_key": "cell", "expected": 2, "registered_suffix_order": ["a", "b"]}]
    if duplicate:
        rows.append(dict(rows[0]))
    with pytest.raises(HistoricalIntegrityError):
        validate_population({"rows": rows, "cells": cells, "expected_scientific_slots": 2})


def test_reject_hash_when_source_bytes_drift(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"changed")
    with pytest.raises(HistoricalIntegrityError):
        checked({"path": str(source), "sha256": hashlib.sha256(b"original").hexdigest()})


@pytest.mark.parametrize("kind", ["input", "collision", "alias"])
def test_reject_output_when_aliased_or_colliding(tmp_path: Path, kind: str) -> None:
    source = tmp_path / "historical"
    source.mkdir()
    output = tmp_path / "report"
    if kind == "input":
        output = source / "report"
    elif kind == "collision":
        output.mkdir()
    else:
        output.symlink_to(source, target_is_directory=True)
    with pytest.raises((HistoricalIntegrityError, FileExistsError)):
        validate_output_path(output, (source,), False)


def test_denominators_when_zero_eligibility_and_unknown_are_separate() -> None:
    rows = [{"task": "task", "baseline": "baseline", "arm": "correct", "cohort": cohort,
             "segment": "c003", "observation_status": "unattempted", "stages": {
                 "registered_z": {"eligible": False, "status": "unknown", "value": None},
                 "retrieved": {"eligible": True, "status": "unknown", "value": None},
             }} for cohort in ("original", "changed-envelope")]
    result = summarize(rows)
    assert len(result) == 4
    assert all(row["expected"] == row["unknown"] == 1 and row["known"] == row["completed"] == 0 for row in result)
    assert all(row["true_over_known"] is None for row in result)
    assert all(row["known_over_expected"] is None for row in result if row["stage"] == "registered_z")
    assert all(row["known_over_expected"] == 0 for row in result if row["stage"] == "retrieved")


def test_attribution_when_intermediate_parent_is_unavailable() -> None:
    given, calls = fixture(included=True)
    raw = given.model_dump(mode="json")
    raw["baseline"] = "dc_rs"
    raw["context"]["final_entry_ids"] = ["strategy"]
    raw["target_set"]["answer_call_spans"] = []
    raw["lineage"].extend([
        {"entry_id": "strategy", "lineage_status": "exact", "direct_parent_ids": ["bridge"]},
        {"entry_id": "bridge", "lineage_status": "unavailable", "direct_parent_ids": ["root-b"]},
    ])
    given = Phase13TrialEvidence.model_validate_json(json.dumps(raw))
    strategy = calls[0].source_spans[0].model_copy(update={
        "entry_id": "strategy", "parent_call_id": "curator-1", "injected_root_ids": [],
        "direct_parent_ids": ["bridge"], "is_target_contamination": False,
    })
    calls = (calls[0].model_copy(update={"source_spans": [strategy]}),)
    result = reconstruct_stages(given, calls, ("root-b",))
    assert result["actual_answer_context"]["value"] is None
    assert result["derivation"]["value"] is None
