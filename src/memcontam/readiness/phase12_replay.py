from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from memcontam.baselines.contracts import BaselineExecutionOutcome
from memcontam.contamination.phase12.registry import load_candidate_registry
from memcontam.contamination.phase12.renderers import RendererRegistry
from memcontam.experiment.phase12 import cli as phase12_cli
from memcontam.experiment.phase12.branching import BranchSet, NoMemAliasRecord, build_matched_branches
from memcontam.experiment.phase12.contracts import canonical_json_hash
from memcontam.experiment.phase12.eligibility import compute_joint_eligibility
from memcontam.experiment.phase12.maturity import MaturityDecision
from memcontam.experiment.phase12.outcomes import classify_baseline_outcome
from memcontam.experiment.phase12.timing import select_timing_checkpoint
from memcontam.logging.schema_v3 import parse_log_record_v3
from memcontam.logging.writer_v3 import Phase12RunWriter
from memcontam.memory.checkpoint_v3 import (
    NativeEntry,
    NativeState,
    Phase12Checkpoint,
    Phase12CheckpointIdentity,
    deserialize_checkpoint,
    serialize_checkpoint,
)


GateId = Literal[
    "prefix_checkpoint",
    "five_arm_branch",
    "nomem_alias",
    "filter_information_boundary",
    "logging_v3_join",
    "model_behavior_denominator",
    "eligibility_recomputation",
    "run_archive_reconstruction",
]
GateStatus = Literal["pass", "fail"]
ReplayStatus = Literal["pass", "fail"]

_GATE_IDS: tuple[GateId, ...] = (
    "prefix_checkpoint",
    "five_arm_branch",
    "nomem_alias",
    "filter_information_boundary",
    "logging_v3_join",
    "model_behavior_denominator",
    "eligibility_recomputation",
    "run_archive_reconstruction",
)
_GATE_REASON_CODES: dict[GateId, str] = {
    "prefix_checkpoint": "P12I_PREFIX_CHECKPOINT_GATE_FAILED",
    "five_arm_branch": "P12I_FIVE_ARM_BRANCH_GATE_FAILED",
    "nomem_alias": "P12I_NOMEM_ALIAS_GATE_FAILED",
    "filter_information_boundary": "P12I_FILTER_INFORMATION_BOUNDARY_GATE_FAILED",
    "logging_v3_join": "P12I_LOGGING_V3_JOIN_GATE_FAILED",
    "model_behavior_denominator": "P12I_MODEL_BEHAVIOR_DENOMINATOR_GATE_FAILED",
    "eligibility_recomputation": "P12I_ELIGIBILITY_RECOMPUTATION_GATE_FAILED",
    "run_archive_reconstruction": "P12I_RUN_ARCHIVE_RECONSTRUCTION_GATE_FAILED",
}
_PUBLIC_STREAMS = (
    "trials.jsonl",
    "calls.jsonl",
    "tool_events.jsonl",
    "retrieval_events.jsonl",
    "context_events.jsonl",
    "failures.jsonl",
    "memory_events.jsonl",
    "admission_events.jsonl",
    "intervention_events.jsonl",
    "checkpoint_events.jsonl",
    "eligibility_events.jsonl",
)


class P12IReplayError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class P12IReplaySpec:
    fixture_root: Path
    e2e_fixture_id: str = "FX-E2E-001"
    p12i_fixture_id: str = "FX-P12I-001"
    branch_fixture_id: str = "FX-BRANCH-001"
    run_id: str = "p12i-contract-replay-run"


@dataclass(frozen=True)
class P12ISubgateEvidence:
    gate_id: GateId
    status: GateStatus
    reason_code: str | None
    evidence_path: Path
    evidence_hash: str | None


@dataclass(frozen=True)
class P12IReplayResult:
    replay_id: str
    scientific_result: Literal[False]
    subgates: tuple[P12ISubgateEvidence, ...]
    overall_status: ReplayStatus
    reason_code: str | None
    archive_run_dir: Path


def run_p12i_replay(spec: P12IReplaySpec, archive_root: Path) -> P12IReplayResult:
    fixture_root = Path(spec.fixture_root)
    archive_root = Path(archive_root)
    fixtures = _load_fixtures(fixture_root)
    e2e = _fixture(fixtures, spec.e2e_fixture_id)
    p12i = _fixture(fixtures, spec.p12i_fixture_id)
    branch_fixture = _fixture(fixtures, spec.branch_fixture_id)
    _validate_fixture_contract(e2e, p12i, spec)

    run_dir = _ensure_branch_replay(spec, fixture_root, archive_root)
    prefix = phase12_cli._build_prefix(branch_fixture)
    _validate_prefix(prefix, branch_fixture)
    branches = _build_branches(prefix, branch_fixture)
    payloads = {
        "prefix_checkpoint": _prefix_payload(prefix),
        "five_arm_branch": _branch_payload(branches),
        "nomem_alias": _nomem_payload(branches, branch_fixture),
        "filter_information_boundary": _filter_payload(branches),
        "logging_v3_join": _logging_payload(run_dir),
        "model_behavior_denominator": _denominator_payload(fixtures),
        "eligibility_recomputation": _eligibility_payload(fixtures),
        "run_archive_reconstruction": _archive_payload(run_dir, fixtures),
    }
    evidence_root = archive_root / "p12i-replay-evidence"
    subgates = tuple(
        _write_or_validate_evidence(gate_id, payloads[gate_id], evidence_root)
        for gate_id in _GATE_IDS
    )
    failed = next((gate for gate in subgates if gate.status == "fail"), None)
    return P12IReplayResult(
        replay_id=spec.p12i_fixture_id,
        scientific_result=False,
        subgates=subgates,
        overall_status="fail" if failed else "pass",
        reason_code=None if failed is None else failed.reason_code,
        archive_run_dir=run_dir,
    )


def _load_fixtures(root: Path) -> dict[str, dict[str, Any]]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    fixtures: dict[str, dict[str, Any]] = {}
    for filename in manifest["files"]:
        payload = json.loads((root / filename).read_text(encoding="utf-8"))
        fixture_id = payload.get("fixture_id")
        if isinstance(fixture_id, str):
            fixtures[fixture_id] = payload
    return fixtures


def _fixture(fixtures: dict[str, dict[str, Any]], fixture_id: str) -> dict[str, Any]:
    try:
        return fixtures[fixture_id]
    except KeyError as error:
        raise P12IReplayError("P12I_FIXTURE_MISSING") from error


def _validate_fixture_contract(
    e2e: dict[str, Any], p12i: dict[str, Any], spec: P12IReplaySpec
) -> None:
    if {spec.p12i_fixture_id, spec.branch_fixture_id} - set(e2e.get("compose", ())):
        raise P12IReplayError("P12I_COMPOSE_REFERENCE_MISSING")
    bfv2 = p12i.get("bfv2", {})
    if bfv2.get("f1a") != "pass" or bfv2.get("f1b") != "pass" or bfv2.get("f1c") not in {
        "pass",
        "blocked",
    }:
        raise P12IReplayError("P12I_BFV2_INPUT_INVALID")
    if p12i.get("expected", {}).get("p12i_overall") != "pass":
        raise P12IReplayError("P12I_FIXTURE_EXPECTATION_INVALID")
    if any(p12i.get("gates", {}).get(gate_id) != "pass" for gate_id in _GATE_IDS):
        raise P12IReplayError("P12I_GATE_FIXTURE_INVALID")


def _ensure_branch_replay(spec: P12IReplaySpec, fixture_root: Path, archive_root: Path) -> Path:
    run_dir = archive_root / spec.run_id
    if not run_dir.exists():
        phase12_cli._run_branch(
            Namespace(
                fixture_root=fixture_root,
                replay=spec.branch_fixture_id,
                run_root=archive_root,
                run_id=spec.run_id,
            )
        )
    return run_dir


def _validate_prefix(prefix: Any, fixture: dict[str, Any]) -> None:
    expected = fixture["baseline_prefixes"]["fh_bounded"]
    checkpoint = prefix.checkpoint
    if (
        checkpoint.identity.checkpoint_id != expected["expected_checkpoint_id"]
        or checkpoint.identity.sha256 != expected["expected_checkpoint_sha256"]
        or serialize_checkpoint(deserialize_checkpoint(checkpoint)).canonical_bytes != checkpoint.canonical_bytes
    ):
        raise P12IReplayError(_GATE_REASON_CODES["prefix_checkpoint"])


def _build_branches(prefix: Any, fixture: dict[str, Any]) -> BranchSet:
    registry_path = Path(__file__).resolve().parents[3] / "data" / "phase12" / "registries" / "candidate_registry_v1.json"
    branches = build_matched_branches(
        prefix.checkpoint,
        load_candidate_registry(registry_path).triplets[0],
        RendererRegistry.native(),
        phase12_cli._admission_context(prefix.checkpoint.state.baseline, prefix.checkpoint.state.entries),
    )
    if not isinstance(branches, BranchSet):
        raise P12IReplayError(_GATE_REASON_CODES["five_arm_branch"])
    expected_entries = tuple(fixture["baseline_prefixes"]["fh_bounded"]["checkpoint"]["entries"])
    if _entry_ids(branches.clean.checkpoint.state.entries) != expected_entries:
        raise P12IReplayError(_GATE_REASON_CODES["five_arm_branch"])
    return branches


def _prefix_payload(prefix: Any) -> dict[str, Any]:
    return {
        "checkpoint_hash": prefix.checkpoint.identity.sha256,
        "checkpoint_id": prefix.checkpoint.identity.checkpoint_id,
        "checkpoint_indices": [event.checkpoint_index for event in prefix.checkpoint_events],
        "prefix_run_id": prefix.prefix_run_id,
        "trial_count": len(prefix.trials),
    }


def _branch_payload(branches: BranchSet) -> dict[str, Any]:
    branch_ids = {
        branch.arm: branch.checkpoint.identity.sha256 for branch in branches.materialized
    }
    if set(branch_ids) != {"clean", "correct", "irrelevant", "contam"}:
        raise P12IReplayError(_GATE_REASON_CODES["five_arm_branch"])
    return {
        "audit_label_count": len(branches.audit_labels),
        "branch_hashes": branch_ids,
        "filter_source_hash": branches.filter.source_checkpoint.identity.sha256,
        "source_checkpoint_hash": branches.source_checkpoint.identity.sha256,
    }


def _nomem_payload(branches: BranchSet, fixture: dict[str, Any]) -> dict[str, Any]:
    checkpoint = Phase12Checkpoint(
        identity=Phase12CheckpointIdentity("nomem", "no_memory", "unused"),
        state=NativeState("no_memory", (), {}),
        canonical_bytes=b"",
        canonical_sha256="",
    )
    alias = build_matched_branches(
        checkpoint,
        load_candidate_registry(
            Path(__file__).resolve().parents[3]
            / "data"
            / "phase12"
            / "registries"
            / "candidate_registry_v1.json"
        ).triplets[0],
        RendererRegistry.native(),
        phase12_cli._admission_context("fh_bounded", branches.source_checkpoint.state.entries),
    )
    if not isinstance(alias, NoMemAliasRecord) or alias.materialized_branches:
        raise P12IReplayError(_GATE_REASON_CODES["nomem_alias"])
    return {
        "display_alias_count": alias.display_alias_count,
        "fixture_baseline_count": len(fixture["baseline_prefixes"]),
        "underlying_execution_count": alias.underlying_execution_count,
    }


def _filter_payload(branches: BranchSet) -> dict[str, Any]:
    root_id = branches.contam.inserted_entry_id
    active_ids = _entry_ids(branches.filter.active.state.entries)
    quarantined_ids = _entry_ids(branches.filter.quarantine.state.entries)
    if (
        root_id is None
        or branches.filter.source_checkpoint != branches.contam.checkpoint
        or root_id in active_ids
        or quarantined_ids != (root_id,)
    ):
        raise P12IReplayError(_GATE_REASON_CODES["filter_information_boundary"])
    return {
        "active_entry_ids": list(active_ids),
        "quarantined_entry_ids": list(quarantined_ids),
        "source_checkpoint_hash": branches.filter.source_checkpoint.identity.sha256,
    }


def _logging_payload(run_dir: Path) -> dict[str, Any]:
    manifest = Phase12RunWriter.read_manifest(run_dir)
    parse_log_record_v3(manifest["run_metadata"])
    trials = Phase12RunWriter.read_jsonl(run_dir, "trials.jsonl")
    trial_ids = {row["trial_id"] for row in trials}
    if not trial_ids:
        raise P12IReplayError(_GATE_REASON_CODES["logging_v3_join"])
    stream_counts: dict[str, int] = {}
    for filename in _PUBLIC_STREAMS:
        rows = Phase12RunWriter.read_jsonl(run_dir, filename)
        stream_counts[filename] = len(rows)
        for row in rows:
            if filename == "trials.jsonl":
                parse_log_record_v3({key: value for key, value in row.items() if key != "trial_id"})
            elif filename not in {"calls.jsonl", "memory_events.jsonl"}:
                if row.get("trial_id") not in trial_ids:
                    raise P12IReplayError(_GATE_REASON_CODES["logging_v3_join"])
                parse_log_record_v3({key: value for key, value in row.items() if key != "trial_id"})
    return {"stream_counts": stream_counts, "trial_count": len(trial_ids)}


def _denominator_payload(fixtures: dict[str, dict[str, Any]]) -> dict[str, Any]:
    expected = {case["id"]: case["expected"] for case in _fixture(fixtures, "FX-OUTCOME-001")["cases"]}
    malformed = classify_baseline_outcome(
        BaselineExecutionOutcome(
            status="failed",
            final_response="invalid",
            error_type="BaselineOutputError",
            failure_disposition="no_memory_invalid_final_answer",
            scientific_ineligibility_reason="invalid_final_answer",
        ),
        None,
    )
    provider = classify_baseline_outcome(
        BaselineExecutionOutcome(
            status="failed",
            error_type="ProviderCallFailure",
            failure_disposition="provider_call_failed",
            scientific_ineligibility_reason="provider_call_failed",
        ),
        None,
    )
    observed = {
        "malformed_final": [
            malformed.execution_status,
            malformed.failure_class,
            malformed.analysis_inclusion,
            malformed.verified_score,
        ],
        "provider_loss": [
            provider.execution_status,
            provider.failure_class,
            provider.analysis_inclusion,
            provider.verified_score,
        ],
    }
    if observed != {case_id: expected[case_id] for case_id in observed}:
        raise P12IReplayError(_GATE_REASON_CODES["model_behavior_denominator"])
    return observed


def _eligibility_payload(fixtures: dict[str, dict[str, Any]]) -> dict[str, Any]:
    fixture = _fixture(fixtures, "FX-ELIGIBILITY-001")
    horizon = fixture["horizon"]
    decisions = [
        MaturityDecision(
            condition_id=f"{family}-{index}",
            baseline_family=family,
            checkpoint_id=f"{family}-t{index}",
            checkpoint_index=index,
            horizon=horizon,
            eligible=True,
        )
        for family, indices in fixture["baseline_eligible"].items()
        for index in indices
    ]
    decisions.append(
        MaturityDecision("nomem-3", "no_memory", "nomem-t3", 3, horizon, True)
    )
    result = compute_joint_eligibility(decisions, horizon)
    base_checkpoint = select_timing_checkpoint(result.joint_eligible_indices, "base")
    if (
        list(result.joint_eligible_indices) != fixture["joint"]
        or result.not_estimable is not fixture["not_estimable"]
        or base_checkpoint != fixture["quantiles"]["0.5"]
        or "no_memory" in result.baseline_eligible
    ):
        raise P12IReplayError(_GATE_REASON_CODES["eligibility_recomputation"])
    return {
        "base_checkpoint": base_checkpoint,
        "joint_eligible_indices": list(result.joint_eligible_indices),
        "primary_baseline_count": result.estimability_counts["primary_baselines"],
    }


def _archive_payload(run_dir: Path, fixtures: dict[str, dict[str, Any]]) -> dict[str, Any]:
    expected = _fixture(fixtures, "FX-ARCHIVE-001")["expected"]
    archive = phase12_cli._validate_archive(Namespace(replay=None, run_dir=run_dir))
    if archive["archive_valid"] is not expected["archive_valid"]:
        raise P12IReplayError(_GATE_REASON_CODES["run_archive_reconstruction"])
    manifest_path = run_dir / "public_artifact_manifest.json"
    return {
        "archive_valid": archive["archive_valid"],
        "public_artifact_manifest_hash": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "stream_count": len(_PUBLIC_STREAMS),
    }


def _write_or_validate_evidence(
    gate_id: GateId, payload: dict[str, Any], evidence_root: Path
) -> P12ISubgateEvidence:
    evidence_root.mkdir(parents=True, exist_ok=True)
    path = evidence_root / f"{gate_id}.json"
    value = {"gate_id": gate_id, "payload": payload}
    expected_hash = canonical_json_hash(value)
    expected_bytes = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if not path.exists():
        path.write_bytes(expected_bytes)
    observed_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed_hash != expected_hash:
        return P12ISubgateEvidence(
            gate_id,
            "fail",
            _GATE_REASON_CODES[gate_id],
            path,
            observed_hash,
        )
    return P12ISubgateEvidence(gate_id, "pass", None, path, observed_hash)


def _entry_ids(entries: tuple[str | NativeEntry, ...]) -> tuple[str, ...]:
    return tuple(entry.entry_id if isinstance(entry, NativeEntry) else entry for entry in entries)


__all__ = [
    "P12IReplayError",
    "P12IReplayResult",
    "P12IReplaySpec",
    "P12ISubgateEvidence",
    "run_p12i_replay",
]
