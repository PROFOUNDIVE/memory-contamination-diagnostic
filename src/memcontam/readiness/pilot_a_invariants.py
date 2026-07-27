from __future__ import annotations

import ast
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Callable

from memcontam.logging.schema_v3 import parse_log_record_v3
from memcontam.readiness.pilot_a_preflight import load_preflight_config


INVARIANT_NAMES = (
    "verifier",
    "nomem_singleton",
    "prefix_clean_checkpoint",
    "branch_contam_filter_source",
    "observability",
    "updater_information",
    "filter_no_hidden_labels",
    "reflexion_verifier_ordering",
    "baseline_pre_generation_ordering",
    "rag_branch_identity",
    "python_candidate_execution",
    "split_no_main_ids",
    "intervention_resolution",
    "fh_label_distinction",
    "filter_result_reason_route",
    "archive_reconstruction",
    "operations_reconciliation",
)
REQUIRED_ARTIFACTS = (
    "run.json",
    "resolved_config.json",
    "provider_profile.json",
    "trials.jsonl",
    "calls.jsonl",
    "retrieval_events.jsonl",
    "context_events.jsonl",
    "failures.jsonl",
    "memory_events.jsonl",
    "admission_events.jsonl",
    "intervention_events.jsonl",
    "checkpoint_events.jsonl",
    "eligibility_events.jsonl",
    "audit/audit_labels.jsonl",
    "public_artifact_manifest.json",
    "decision_ledger.jsonl",
)
_SEALED_ARTIFACT = "archive_seal.json"
_OPTIONAL_EMPTY_STREAMS = ("failures.jsonl",)
_SECRET_PATTERN = re.compile(r"(?i)(api[_-]?key|authorization|bearer\s|sk-[a-z0-9]{8,}|password|secret)")
_SCIENTIFIC_MANIFEST_STATUSES = frozenset({"completed", "blocked", "invalidated", "interrupted"})


class PilotAInvariantError(ValueError):
    def __init__(self, code: str, *, hash_mismatches: int = 0) -> None:
        self.code = code
        self.hash_mismatches = hash_mismatches
        super().__init__(code)


def run_replay(config_path: Path, run_id: str, *, artifact_root: Path | None = None, client: object | None = None) -> Path:
    """Write a fixed offline three-arm Pilot-A audit run without constructing a provider client."""
    if client is not None:
        raise PilotAInvariantError("LIVE_PROVIDER_CLIENT_FORBIDDEN")
    load_preflight_config(Path(config_path))
    _validate_run_id(run_id)
    root = artifact_root or _artifact_root()
    run_dir = root / "runs" / run_id
    if run_dir.exists():
        raise PilotAInvariantError("RUN_ID_ALREADY_EXISTS")
    run_dir.mkdir(parents=True)
    (run_dir / "audit").mkdir()

    files = _replay_files(Path(config_path), run_id)
    for filename, payload in files.items():
        _write(run_dir / filename, payload)
    artifacts = {
        filename: {"count": _count(run_dir / filename), "sha256": _sha256(run_dir / filename)}
        for filename in REQUIRED_ARTIFACTS
        if filename != "public_artifact_manifest.json"
    }
    _write(run_dir / "public_artifact_manifest.json", {"artifacts": artifacts, "status": "completed"})
    _write(
        run_dir / _SEALED_ARTIFACT,
        {"public_artifact_manifest_sha256": _sha256(run_dir / "public_artifact_manifest.json")},
    )
    return run_dir


def inspect_run(run_dir: Path) -> dict[str, Any]:
    try:
        data, artifacts = _load_run(Path(run_dir))
        results = _run_invariants(data)
        failed = next((result for result in results if result["status"] == "fail"), None)
        return {
            "artifacts": artifacts,
            "live_provider_calls": data["ledger"]["live_provider_calls"],
            "manifest_hash": _sha256(Path(run_dir) / "public_artifact_manifest.json"),
            "overall": "fail" if failed else "pass",
            "reason_code": None if failed is None else failed["reason_code"],
            "results": results,
            "run_dir": str(Path(run_dir)),
            "scientific_result": data["ledger"]["scientific_result"],
        }
    except PilotAInvariantError as error:
        return _failed_report(Path(run_dir), error)


def validate_archive(run_dir: Path) -> dict[str, Any]:
    report = inspect_run(run_dir)
    if report["overall"] == "pass":
        report["artifacts"] = {
            **report["artifacts"],
            "archive_seal.json": {
                "count": 1,
                "sha256": _sha256(Path(run_dir) / "archive_seal.json"),
            },
            "public_artifact_manifest.json": {
                "count": 1,
                "sha256": _sha256(Path(run_dir) / "public_artifact_manifest.json"),
            },
        }
    report["unresolved_references"] = 0 if report["overall"] == "pass" else 1
    report["hash_mismatches"] = 0 if report["overall"] == "pass" else int(
        report.get("reason_code") == "ARCHIVE_HASH_MISMATCH"
    )
    if report["overall"] == "pass":
        report["retry_total"] = sum(call["retry_count"] for call in _load_run(Path(run_dir))[0]["calls"])
        report["cost_total"] = sum(call["cost_usd"] for call in _load_run(Path(run_dir))[0]["calls"])
    return report


def _replay_files(config_path: Path, run_id: str) -> dict[str, Any]:
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    prefix_trial = _trial_prefix(run_id)
    clean_trial = _trial_branch(run_id, "clean")
    contam_trial = _trial_branch(run_id, "contam")
    filter_trial = _trial_branch(run_id, "filter")
    trials = [prefix_trial, clean_trial, contam_trial, filter_trial]
    clean_id = clean_trial["trial_id"]
    contam_id = contam_trial["trial_id"]
    filter_id = filter_trial["trial_id"]
    def event(event_id: str, trial_id: str, record_type: str, **fields: Any) -> dict[str, Any]:
        return {
            "schema_version": "logging_v3",
            "contract_level": "phase12",
            "event_id": event_id,
            "run_id": run_id,
            "trial_id": trial_id,
            "event_seq": 0,
            "record_type": record_type,
            **fields,
        }
    retrieval = event(
        "retrieval:clean",
        clean_id,
        "retrieval_event",
        retrieval_id="retrieval:clean",
        query_hash="sha256:clean-query",
        retrieved_entry_ids=["corpus:clean"],
        retrieved_scores=[1.0],
    )
    context = event(
        "context:clean",
        clean_id,
        "context_event",
        context_id="context:clean",
        final_entry_ids=["context:clean-entry"],
    )
    admissions = [
        event(f"admission:{arm}", trial_id, "admission_event", admission_id=f"admission:{arm}", decision="admit")
        for arm, trial_id in (("clean", clean_id), ("contam", contam_id), ("filter", filter_id))
    ]
    interventions = [
        event(
            f"intervention:{arm}",
            trial_id,
            "intervention_event",
            intervention_id=f"intervention:{arm}",
            arm=arm,
            candidate_triplet_id="game24-fraction-intermediate-v1",
            native_render_id="render-game24-false-v1",
        )
        for arm, trial_id in (("contam", contam_id), ("filter", filter_id))
    ]
    checkpoints = [
        event(
            "checkpoint:prefix",
            prefix_trial["trial_id"],
            "checkpoint_event",
            checkpoint_id="checkpoint:clean",
            checkpoint_index=0,
            memory_hash="sha256:clean",
        ),
        *[
            event(
                f"checkpoint:{arm}",
                trial_id,
                "checkpoint_event",
                checkpoint_id=f"checkpoint:{arm}",
                checkpoint_index=1,
                memory_hash=f"sha256:{arm}",
            )
            for arm, trial_id in (("clean", clean_id), ("contam", contam_id), ("filter", filter_id))
        ],
    ]
    eligibility = [
        event(
            f"eligibility:{arm}",
            trial_id,
            "eligibility_event",
            eligibility_id=f"eligibility:{arm}",
            eligible=True,
        )
        for arm, trial_id in (("clean", clean_id), ("contam", contam_id), ("filter", filter_id))
    ]
    ledger = {
        "baseline_pre_generation_registered": ["dynamic_cheatsheet_rs_optional"],
        "branch": {"contam_source": "checkpoint:clean", "filter_source": "checkpoint:contam"},
        "fh_labels": {"bounded": "bounded", "exact": "exact"},
        "filter_result": {"injected_root": "candidate:game24-false", "reason_route": "quarantine"},
        "live_provider_calls": 0,
        "nomem": {"aliases": 5, "underlying_executions": 1},
        "observability": {
            "context_ids": ["context:clean-entry"],
            "retrieval_ids": ["corpus:clean"],
            "storage_ids": ["memory:clean"],
        },
        "operations": {"cost_total": 0.0, "retry_total": 0},
        "python_candidate": {
            "code": "def is_integer_intermediate(numerator, denominator):\n    return denominator == 1\n",
            "parser_status": "parsed",
            "runtime_status": "not_executed",
            "semantic_result": "semantic_invalid",
            "termination_status": "within_limit",
            "timeout_seconds": 0.1,
            "validation_mode": "static_metadata",
        },
        "rag": {"branch_corpus_id": "corpus:clean", "branch_index_id": "index:clean"},
        "scientific_result": False,
        "split": {"build_calibration_ids": ["build-game24-1"], "main_ids": []},
        "verifier": {"recomputed_outcome": False, "stored_outcome": False},
    }
    return {
        "run.json": {
            "run_id": run_id,
            "run_metadata": _metadata(),
            "schema_version": "phase12_pilot_a_replay_v1",
            "status": "completed",
        },
        "resolved_config.json": {
            "config_sha256": config_hash,
            "execution_class": "offline_contract_replay",
            "scientific_result": False,
        },
        "provider_profile.json": {
            "live_provider_calls": 0,
            "provider": "replay",
            "transport": "offline",
        },
        "trials.jsonl": trials,
        "calls.jsonl": [
            {
                "baseline": "dynamic_cheatsheet_rs_optional",
                "call_id": "call:dc-pre",
                "cost_usd": 0.0,
                "pre_generation": True,
                "retry_count": 0,
                "sequence": 1,
                "stage": "dc_rs_synthesize",
                "trial_id": clean_id,
            },
            {
                "baseline": "reflexion_style",
                "call_id": "call:reflexion-actor",
                "cost_usd": 0.0,
                "pre_generation": False,
                "retry_count": 0,
                "sequence": 2,
                "stage": "reflexion_actor",
                "trial_id": clean_id,
            },
            {
                "baseline": "reflexion_style",
                "call_id": "call:reflexion-verifier",
                "cost_usd": 0.0,
                "pre_generation": False,
                "retry_count": 0,
                "sequence": 3,
                "stage": "reflexion_verifier",
                "trial_id": clean_id,
            },
            {
                "baseline": "rag_frozen",
                "call_id": "call:rag-generate",
                "cost_usd": 0.0,
                "pre_generation": False,
                "retry_count": 0,
                "sequence": 4,
                "stage": "rag_generate",
                "trial_id": clean_id,
            },
        ],
        "retrieval_events.jsonl": [retrieval],
        "context_events.jsonl": [context],
        "failures.jsonl": [],
        "memory_events.jsonl": [
            {
                "available_information_ids": ["context:clean-entry"],
                "event_id": "memory:clean",
                "parent_ids": ["context:clean-entry"],
                "trial_id": clean_id,
            }
        ],
        "admission_events.jsonl": admissions,
        "intervention_events.jsonl": interventions,
        "checkpoint_events.jsonl": checkpoints,
        "eligibility_events.jsonl": eligibility,
        "audit/audit_labels.jsonl": [
            {"candidate_id": "candidate:game24-false", "origin_class": "protocol_injected"}
        ],
        "decision_ledger.jsonl": ledger,
    }


def _metadata() -> dict[str, Any]:
    return {
        "abstract_seed_slot_or_none": None,
        "baseline_condition_id": "fh_bounded",
        "behavior_registry_version": "replay",
        "candidate_registry_version": "replay",
        "contract_level": "phase12",
        "embedding_contract_hash": "replay",
        "evidence_layer": "build",
        "execution_key": {"kind": "branch_free_prefix"},
        "metadata_kind": "pre_route",
        "metric_registry_version": "replay",
        "prefix_template_key_or_none": "pilot-a-clean-prefix",
        "protocol_index_or_none": None,
        "protocol_version": "phase12_primary_v1",
        "rerun_policy_version": "replay",
        "run_family": "readiness",
        "run_template_id": "pilot-a-offline-replay",
        "run_template_registry_version": "replay",
        "schema_version": "logging_v3",
        "scientific_admission_ref_or_none": None,
        "scientific_result": False,
        "sensitivity_cell_ref": {"cell_id": "base", "kind": "base"},
        "split_manifest_version": "replay",
        "task_family": "game24",
        "tool_contract_hash": "replay",
        "trajectory_seed": 0,
    }


def _trial_prefix(run_id: str) -> dict[str, Any]:
    return {
        "absolute_trial_index": 0,
        "admission_event_ids": [],
        "analysis_inclusion": "included",
        "auxiliary_context_inclusion_or_none": None,
        "checkpoint_event_ids": ["checkpoint:prefix"],
        "context_event_id_or_none": None,
        "contract_level": "phase12",
        "event_time": 0,
        "execution_key": {"kind": "branch_free_prefix"},
        "execution_status": "completed",
        "failure_class": None,
        "inclusion_reason": "offline_replay",
        "memory_event_ids": [],
        "operational_attribution_or_none": None,
        "parse_status": "parsed",
        "prefix_run_id": "prefix:pilot-a",
        "retrieval_event_ids": [],
        "schema_version": "logging_v3",
        "tool_event_ids": [],
        "trial_id": f"{run_id}:prefix",
        "trial_kind": "branch_free_prefix",
    }


def _trial_branch(run_id: str, arm: str) -> dict[str, Any]:
    clean = arm == "clean"
    return {
        "absolute_trial_index": 1,
        "admission_event_ids": [f"admission:{arm}"],
        "analysis_inclusion": "included",
        "auxiliary_context_inclusion_or_none": None,
        "branch_id": arm,
        "candidate_triplet_id_or_none": None if clean else "game24-fraction-intermediate-v1",
        "checkpoint_id": "checkpoint:clean",
        "checkpoint_index": 0,
        "context_event_id_or_none": "context:clean" if clean else None,
        "contract_level": "phase12",
        "event_time": 1,
        "execution_key": {"arm": arm, "kind": "memory_arm"},
        "execution_status": "completed",
        "failure_class": None,
        "inclusion_reason": "offline_replay",
        "intervention_event_id_or_none": None if clean else f"intervention:{arm}",
        "memory_event_ids": ["memory:clean"] if clean else [],
        "native_render_id_or_none": None if clean else "render-game24-false-v1",
        "operational_attribution_or_none": None,
        "parse_status": "parsed",
        "prefix_run_id": "prefix:pilot-a",
        "retrieval_event_ids": ["retrieval:clean"] if clean else [],
        "schema_version": "logging_v3",
        "tool_event_ids": [],
        "trial_id": f"{run_id}:{arm}",
        "trial_kind": "memory_branch",
    }


def _load_run(run_dir: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if not run_dir.is_dir():
        raise PilotAInvariantError("REQUIRED_ARTIFACT_MISSING")
    required = (*REQUIRED_ARTIFACTS, _SEALED_ARTIFACT)
    if any(not (run_dir / filename).is_file() for filename in required):
        raise PilotAInvariantError("REQUIRED_ARTIFACT_MISSING")
    manifest = _json(run_dir / "public_artifact_manifest.json")
    if manifest.get("status") not in _SCIENTIFIC_MANIFEST_STATUSES or not isinstance(manifest.get("artifacts"), dict):
        raise PilotAInvariantError("ARCHIVE_MANIFEST_INVALID")
    expected_files = {str(filename) for filename in REQUIRED_ARTIFACTS} - {
        "public_artifact_manifest.json"
    }
    if set(manifest["artifacts"]) != expected_files:
        raise PilotAInvariantError("ARCHIVE_MANIFEST_INVALID")
    artifacts = manifest["artifacts"]
    mismatches = sum(
        not isinstance(artifact, dict)
        or artifact.get("sha256") != _sha256(run_dir / filename)
        or artifact.get("count") != _count(run_dir / filename)
        for filename, artifact in artifacts.items()
    )
    if mismatches:
        raise PilotAInvariantError("ARCHIVE_HASH_MISMATCH", hash_mismatches=mismatches)
    seal = _json(run_dir / _SEALED_ARTIFACT)
    if seal.get("public_artifact_manifest_sha256") != _sha256(
        run_dir / "public_artifact_manifest.json"
    ):
        raise PilotAInvariantError("ARCHIVE_SEAL_MISMATCH")
    raw = {filename: _read_artifact(run_dir / filename) for filename in REQUIRED_ARTIFACTS}
    _reject_secrets(raw)
    if any(raw[filename] for filename in _OPTIONAL_EMPTY_STREAMS):
        raise PilotAInvariantError("OPTIONAL_STREAM_UNEXPECTED_NONEMPTY")
    try:
        parse_log_record_v3(raw["run.json"]["run_metadata"])
        for filename in (
            "trials.jsonl",
            "retrieval_events.jsonl",
            "context_events.jsonl",
            "admission_events.jsonl",
            "intervention_events.jsonl",
            "checkpoint_events.jsonl",
            "eligibility_events.jsonl",
        ):
            for row in raw[filename]:
                parse_log_record_v3({key: value for key, value in row.items() if key != "trial_id"})
    except (TypeError, ValueError) as error:
        raise PilotAInvariantError("LOGGING_V3_PARSE_FAILURE") from error
    _validate_references(raw)
    if raw["provider_profile.json"].get("provider") != "replay":
        raise PilotAInvariantError("LIVE_PROVIDER_CLIENT_FORBIDDEN")
    ledger_rows = raw["decision_ledger.jsonl"]
    if len(ledger_rows) != 1 or not isinstance(ledger_rows[0], dict):
        raise PilotAInvariantError("DECISION_LEDGER_INVALID")
    return {
        "calls": raw["calls.jsonl"],
        "ledger": ledger_rows[0],
        "memory": raw["memory_events.jsonl"],
        "raw": raw,
    }, artifacts


def _validate_references(raw: dict[str, Any]) -> None:
    trial_ids = {row.get("trial_id") for row in raw["trials.jsonl"]}
    if None in trial_ids or len(trial_ids) != len(raw["trials.jsonl"]):
        raise PilotAInvariantError("UNRESOLVED_REFERENCE")
    event_files = {
        "retrieval_events.jsonl": "retrieval_event_ids",
        "context_events.jsonl": None,
        "admission_events.jsonl": "admission_event_ids",
        "intervention_events.jsonl": "intervention_event_id_or_none",
        "checkpoint_events.jsonl": "checkpoint_event_ids",
        "eligibility_events.jsonl": None,
    }
    ids: dict[str, set[str]] = {}
    for filename in event_files:
        rows = raw[filename]
        if any(row.get("trial_id") not in trial_ids for row in rows):
            raise PilotAInvariantError("UNRESOLVED_REFERENCE")
        ids[filename] = {str(row.get("event_id")) for row in rows}
    memory_ids = {str(row.get("event_id")) for row in raw["memory_events.jsonl"]}
    for trial in raw["trials.jsonl"]:
        checks = (
            (trial.get("retrieval_event_ids", []), ids["retrieval_events.jsonl"]),
            (trial.get("admission_event_ids", []), ids["admission_events.jsonl"]),
            (trial.get("checkpoint_event_ids", []), ids["checkpoint_events.jsonl"]),
            (trial.get("memory_event_ids", []), memory_ids),
        )
        if any(not set(values) <= known for values, known in checks):
            raise PilotAInvariantError("UNRESOLVED_REFERENCE")
        intervention = trial.get("intervention_event_id_or_none")
        if intervention is not None and intervention not in ids["intervention_events.jsonl"]:
            raise PilotAInvariantError("UNRESOLVED_REFERENCE")
        context = trial.get("context_event_id_or_none")
        if context is not None and context not in ids["context_events.jsonl"]:
            raise PilotAInvariantError("UNRESOLVED_REFERENCE")


def _run_invariants(data: dict[str, Any]) -> list[dict[str, str | None]]:
    checks: dict[str, Callable[[dict[str, Any]], str | None]] = {
        "verifier": _check_verifier,
        "nomem_singleton": _check_nomem,
        "prefix_clean_checkpoint": _check_prefix,
        "branch_contam_filter_source": _check_branch,
        "observability": _check_observability,
        "updater_information": _check_updater,
        "filter_no_hidden_labels": _check_filter_labels,
        "reflexion_verifier_ordering": _check_reflexion,
        "baseline_pre_generation_ordering": _check_baseline_order,
        "rag_branch_identity": _check_rag,
        "python_candidate_execution": _check_python_candidate,
        "split_no_main_ids": _check_split,
        "intervention_resolution": _check_interventions,
        "fh_label_distinction": _check_fh,
        "filter_result_reason_route": _check_filter_route,
        "archive_reconstruction": lambda _: None,
        "operations_reconciliation": _check_operations,
    }
    results = []
    for name in INVARIANT_NAMES:
        code = checks[name](data)
        results.append({"name": name, "reason_code": code, "status": "fail" if code else "pass"})
    return results


def _check_verifier(data: dict[str, Any]) -> str | None:
    verifier = data["ledger"]["verifier"]
    return None if verifier["stored_outcome"] == verifier["recomputed_outcome"] else "VERIFIER_RECOMPUTATION_FAILED"


def _check_nomem(data: dict[str, Any]) -> str | None:
    nomem = data["ledger"]["nomem"]
    return None if nomem == {"aliases": 5, "underlying_executions": 1} else "NOMEM_SINGLETON_FAILED"


def _check_prefix(data: dict[str, Any]) -> str | None:
    checkpoints = data["raw"]["checkpoint_events.jsonl"]
    prefix = next(row for row in checkpoints if row["event_id"] == "checkpoint:prefix")
    clean = next(row for row in checkpoints if row["event_id"] == "checkpoint:clean")
    return None if prefix["memory_hash"] == "sha256:clean" and clean["checkpoint_id"] == "checkpoint:clean" else "PREFIX_CLEAN_CHECKPOINT_FAILED"


def _check_branch(data: dict[str, Any]) -> str | None:
    branch = data["ledger"]["branch"]
    return None if branch == {"contam_source": "checkpoint:clean", "filter_source": "checkpoint:contam"} else "BRANCH_CONTAM_FILTER_SOURCE_FAILED"


def _check_observability(data: dict[str, Any]) -> str | None:
    values = data["ledger"]["observability"].values()
    ids = [item for value in values for item in value]
    return None if len(ids) == len(set(ids)) else "OBSERVABILITY_SEPARATION_FAILED"


def _check_updater(data: dict[str, Any]) -> str | None:
    return None if all(row["available_information_ids"] and row["parent_ids"] for row in data["memory"]) else "UPDATER_INFORMATION_FAILED"


def _check_filter_labels(data: dict[str, Any]) -> str | None:
    public = {
        name: value
        for name, value in data["raw"].items()
        if not name.startswith("audit/") and name != "public_artifact_manifest.json"
    }
    return None if "hidden_label" not in _canonical(public) and "audit_label" not in _canonical(public) else "FILTER_HIDDEN_LABEL_FAILED"


def _check_reflexion(data: dict[str, Any]) -> str | None:
    calls = {row["stage"]: row["sequence"] for row in data["calls"]}
    return None if calls["reflexion_verifier"] > calls["reflexion_actor"] else "REFLEXION_VERIFIER_ORDERING_FAILED"


def _check_baseline_order(data: dict[str, Any]) -> str | None:
    registered = set(data["ledger"]["baseline_pre_generation_registered"])
    actual = {row["baseline"] for row in data["calls"] if row["pre_generation"]}
    return None if actual <= registered else "BASELINE_PRE_GENERATION_ORDERING_FAILED"


def _check_rag(data: dict[str, Any]) -> str | None:
    rag = data["ledger"]["rag"]
    return None if rag["branch_corpus_id"] and rag["branch_index_id"] else "RAG_BRANCH_IDENTITY_FAILED"


def _check_python_candidate(data: dict[str, Any]) -> str | None:
    candidate = data["ledger"]["python_candidate"]
    code = candidate.get("code")
    if not isinstance(code, str):
        return "PYTHON_CANDIDATE_PARSER_FAILURE"
    try:
        ast.parse(code)
    except SyntaxError:
        return "PYTHON_CANDIDATE_PARSER_FAILURE"
    if candidate.get("validation_mode") != "static_metadata" or candidate.get("parser_status") != "parsed":
        return "PYTHON_CANDIDATE_PARSER_FAILURE"
    if candidate.get("runtime_status") != "not_executed":
        return "PYTHON_CANDIDATE_RUNTIME_FAILURE"
    if candidate.get("termination_status") != "within_limit":
        return "PYTHON_CANDIDATE_TIMEOUT"
    return None if candidate["semantic_result"] in {"semantic_invalid", "semantic_valid"} else "PYTHON_CANDIDATE_SEMANTIC_STATUS_INVALID"


def _check_split(data: dict[str, Any]) -> str | None:
    split = data["ledger"]["split"]
    return None if not set(split["build_calibration_ids"]) & set(split["main_ids"]) else "SPLIT_MAIN_ID_LEAKAGE"


def _check_interventions(data: dict[str, Any]) -> str | None:
    interventions = data["raw"]["intervention_events.jsonl"]
    return None if all(row["candidate_triplet_id"] and row["native_render_id"] for row in interventions) else "INTERVENTION_RESOLUTION_FAILED"


def _check_fh(data: dict[str, Any]) -> str | None:
    labels = data["ledger"]["fh_labels"]
    return None if labels["bounded"] != labels["exact"] else "FH_LABEL_DISTINCTION_FAILED"


def _check_filter_route(data: dict[str, Any]) -> str | None:
    result = data["ledger"]["filter_result"]
    return None if result["injected_root"] and result["reason_route"] == "quarantine" else "FILTER_RESULT_REASON_ROUTE_FAILED"


def _check_operations(data: dict[str, Any]) -> str | None:
    operations = data["ledger"]["operations"]
    retry_total = sum(row["retry_count"] for row in data["calls"])
    cost_total = sum(row["cost_usd"] for row in data["calls"])
    if data["ledger"]["live_provider_calls"] != 0 or data["ledger"]["scientific_result"] is not False:
        return "OPERATIONS_RECONCILIATION_FAILED"
    return None if (retry_total, cost_total) == (operations["retry_total"], operations["cost_total"]) else "OPERATIONS_RECONCILIATION_FAILED"


def _failed_report(run_dir: Path, error: PilotAInvariantError) -> dict[str, Any]:
    return {
        "artifacts": {},
        "hash_mismatches": error.hash_mismatches,
        "live_provider_calls": 0,
        "manifest_hash": None,
        "overall": "fail",
        "reason_code": error.code,
        "results": [],
        "run_dir": str(run_dir),
        "scientific_result": False,
    }


def _artifact_root() -> Path:
    value = os.environ.get("MEMCONTAM_ARTIFACT_ROOT")
    if not value:
        raise PilotAInvariantError("MEMCONTAM_ARTIFACT_ROOT_REQUIRED")
    return Path(value)


def _validate_run_id(run_id: str) -> None:
    candidate = Path(run_id)
    if candidate.is_absolute() or ".." in candidate.parts or len(candidate.parts) != 1:
        raise PilotAInvariantError("INVALID_RUN_ID")


def _read_artifact(path: Path) -> Any:
    if path.suffix == ".jsonl":
        return [_json_line(line, path) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return _json(path)


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PilotAInvariantError("ARCHIVE_JSON_INVALID") from error


def _json_line(line: str, path: Path) -> Any:
    try:
        return json.loads(line)
    except json.JSONDecodeError as error:
        raise PilotAInvariantError("ARCHIVE_JSON_INVALID") from error


def _reject_secrets(value: Any) -> None:
    if _SECRET_PATTERN.search(_canonical(value)):
        raise PilotAInvariantError("SECRET_LIKE_VALUE_FORBIDDEN")


def _write(path: Path, value: Any) -> None:
    path.write_text(_canonical(value), encoding="utf-8")


def _canonical(value: Any) -> str:
    if isinstance(value, list):
        return "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in value)
    return json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines()) if path.suffix == ".jsonl" else 1


__all__ = [
    "INVARIANT_NAMES",
    "REQUIRED_ARTIFACTS",
    "PilotAInvariantError",
    "inspect_run",
    "run_replay",
    "validate_archive",
]
