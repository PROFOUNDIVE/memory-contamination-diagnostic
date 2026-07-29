from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal

import pytest
from tests.test_phase12_filter_v5_final_verifier_modes import (
    VerifierFixture,
    _fixture,
    _request,
    _terminal_request,
)

from memcontam.experiment.phase12.filter_challenge.evidence_contract import (
    canonical_json_bytes,
    json_value_from_bytes,
)
from memcontam.experiment.phase12.filter_challenge.final_verifier import (
    FinalVerifierError,
    verify_final_report,
)
from memcontam.experiment.phase12.filter_challenge.mft_state_models import JsonValue


JsonObject = dict[str, JsonValue]
ApprovalMode = Literal["plan-compliance", "code-quality", "integration", "scope"]
ApprovalMutation = Callable[[JsonObject], None]
_APPROVAL_MODES: tuple[ApprovalMode, ...] = (
    "plan-compliance", "code-quality", "integration", "scope",
)


def _approvals(tmp_path: Path) -> tuple[VerifierFixture, tuple[Path, Path, Path, Path]]:
    fixture = _fixture(tmp_path)
    paths: tuple[Path, Path, Path, Path] = (
        tmp_path / "f1.json", tmp_path / "f2.json", tmp_path / "f3.json", tmp_path / "f4.json",
    )
    for mode, path in zip(_APPROVAL_MODES, paths, strict=True):
        verify_final_report(_request(fixture, mode, path))
    return fixture, paths


def _replace(path: Path, mutation: ApprovalMutation) -> None:
    payload = _json_object(json_value_from_bytes(path.read_bytes(), "TEST_APPROVAL_INVALID"))
    mutation(payload)
    path.write_bytes(canonical_json_bytes(payload))


def _json_object(value: JsonValue) -> JsonObject:
    assert isinstance(value, dict)
    return value


def _json_array(value: JsonValue) -> list[JsonValue]:
    assert isinstance(value, list)
    return value


def _json_field(value: JsonObject, name: str) -> JsonValue:
    assert name in value
    return value[name]


def _command(report: JsonObject) -> JsonObject:
    return _json_object(_json_array(_json_field(report, "commands"))[0])


def _reconciled_output(report: JsonObject, name: str) -> JsonObject:
    outputs = _json_object(_json_field(report, "reconciled_outputs"))
    return _json_object(_json_field(outputs, name))


def test_terminal_rejects_semantic_approval_tampering(tmp_path: Path) -> None:
    fixture, paths = _approvals(tmp_path)
    f1, f2, f3, f4 = paths
    mutations: tuple[tuple[Path, ApprovalMutation], ...] = (
        (f2, lambda report: _command(report).__setitem__("stdout_sha256", "f" * 64)),
        (f2, lambda report: report.__setitem__("base_commit", "b" * 40)),
        (f4, lambda report: report.__setitem__("base_commit", "b" * 40)),
        (f2, lambda report: report.__setitem__("changed_paths", ["scripts/tampered.py"])),
        (f3, lambda report: _command(report).__setitem__("stdout_sha256", "f" * 64)),
        (f3, lambda report: _json_array(_json_field(report, "mft_pass_ids")).__setitem__(0, "MFT-FV5-TAMPERED")),
        (f3, lambda report: _reconciled_output(report, "validate-selected-policy").__setitem__("execution_authorized", True)),
        (f3, lambda report: _reconciled_output(report, "build-archive").__setitem__("implementation_commit", "b" * 40)),
        (f3, lambda report: _reconciled_output(report, "bct-readiness").__setitem__("provider_calls_issued", 1)),
        (f3, lambda report: _json_object(_json_field(report, "bct_family_statuses")).__setitem__("BCT-FV5-01-CERTIFIED", "executed")),
        (f4, lambda report: report.__setitem__("authority_status", "tampered")),
        (f4, lambda report: report.__setitem__("source_dirty_allowlist", [])),
        (f4, lambda report: report.__setitem__("task_worktree_clean", False)),
    )
    originals = {path: path.read_text(encoding="utf-8") for path in paths}

    for path, mutation in mutations:
        _replace(path, mutation)
        with pytest.raises(FinalVerifierError, match="FINAL_APPROVAL_MISMATCH"):
            verify_final_report(_terminal_request(fixture, tmp_path / "terminal.json", paths))
        path.write_text(originals[path], encoding="utf-8")


def test_terminal_lists_every_unresolved_scientific_choice(tmp_path: Path) -> None:
    fixture, paths = _approvals(tmp_path)

    report = verify_final_report(_terminal_request(fixture, tmp_path / "terminal.json", paths))

    choices = _json_object(report["remaining_scientific_choices"])
    assert list(choices) == [
        "inventory",
        "operational_suite",
        "decision_rule",
        "kappa",
        "coverage_contract",
        "probe_count",
        "replicate_count",
        "retry_count",
        "canonicalizer",
        "tolerance",
        "evaluability_rate",
        "inclusion_rate",
        "ordinary_route_rate",
        "price_registry",
        "monetary_cost_cap",
        "latency_cap",
        "ci_procedure",
        "constraint_order",
        "tie_break",
        "provider_authorization",
    ]
    assert choices["inventory"] == "pending_freeze"
    assert choices["provider_authorization"] == "absent"
    assert all(value == "unresolved" for key, value in choices.items() if key not in {"inventory", "provider_authorization"})


def test_terminal_rejects_coordinated_f2_f4_commit_and_path_tampering(tmp_path: Path) -> None:
    fixture, paths = _approvals(tmp_path)
    _, f2, _, f4 = paths
    for path in (f2, f4):
        _replace(path, lambda report: report.__setitem__("base_commit", "b" * 40))
        _replace(path, lambda report: report.__setitem__("changed_paths", ["scripts/tampered.py"]))

    with pytest.raises(FinalVerifierError, match="FINAL_APPROVAL_MISMATCH"):
        verify_final_report(_terminal_request(fixture, tmp_path / "terminal.json", paths))
