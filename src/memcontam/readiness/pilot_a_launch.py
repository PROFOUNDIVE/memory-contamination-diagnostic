from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Callable, cast

import yaml  # type: ignore[import-untyped]

from memcontam.clients.config import ProviderConfig
from memcontam.clients.cost_guard import CostGuard, CostLimitExceeded
from memcontam.clients.factory import build_llm_client
from memcontam.clients.base import LLMClient
from memcontam.baselines.bot_solve import BOT_SOLVE_PROMPT_VERSION
from memcontam.baselines.common import FINAL_ANSWER_PARSER_VERSION, FINAL_ANSWER_PROMPT_VERSION
from memcontam.experiment.phase12.game24_runner import Game24RuntimeContext, RuntimeIdentities, run_clean_game24_trial
from memcontam.memory.embeddings import BgeM3EmbeddingProvider
from memcontam.memory.checkpoint_v3 import NativeEntry
from memcontam.memory.stores import MemoryEntry
from memcontam.baselines.bot_phase12 import BoTStateV3
from memcontam.baselines.full_history_phase12 import FullHistoryStateV3
from memcontam.baselines.reflexion_phase12 import ReflexionStateV3
from memcontam.baselines.retrieval_rag_phase12 import RagFrozenStateV3
from memcontam.rag.branch_index import build_branch_indices
from memcontam.rag.phase12_corpus import CleanCorpus, build_branch_corpora
from memcontam.readiness.pilot_a_preflight import PreflightError, load_preflight_config
from memcontam.readiness.retrieval_smoke import resolve_bge_cache_path
from memcontam.tasks.game24 import build_instance
from memcontam.verifiers.game24 import verify_expression


DEFAULT_EVIDENCE_ROOT = Path(".sisyphus/evidence/pilot-a-unblock")
PLUMBING_BASELINES = ("nomem", "fh_bounded", "rag_frozen", "bot_style", "reflexion_style")
PLUMBING_MAX_CALLS = 10
PILOT_A_INSTANCE_COUNT = 8
MAX_INPUT_TOKENS_PER_CALL = 4_096
_SECRET_PATTERN = re.compile(r"(?i)(api[_-]?key|authorization|bearer\s|sk-[a-z0-9]{8,}|password|secret)")


class PilotALaunchError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def cost_preview(config_path: Path, *, cost_guard: CostGuard | None = None) -> dict[str, object]:
    payload = _load_config(config_path)
    guard = cost_guard or _cost_guard(payload)
    max_output_tokens = int(payload["decoding"]["max_output_tokens"])
    per_call = guard.estimate_cost(
        input_tokens=MAX_INPUT_TOKENS_PER_CALL, output_tokens=max_output_tokens
    )
    plumbing = per_call * PLUMBING_MAX_CALLS
    pilot_a = plumbing * PILOT_A_INSTANCE_COUNT
    try:
        guard.check_before_dispatch(pilot_a)
    except CostLimitExceeded as error:
        raise PilotALaunchError("PROJECTED_COST_EXCEEDS_CEILING") from error
    return {
        "hard_ceiling_usd": 5.0,
        "max_input_tokens_per_call": MAX_INPUT_TOKENS_PER_CALL,
        "max_output_tokens": max_output_tokens,
        "pilot_a_instance_count": PILOT_A_INSTANCE_COUNT,
        "plumbing_max_cost_usd": plumbing,
        "projected_max_cost_usd": pilot_a,
        "safe": True,
        "scientific_result": False,
    }


def run_plumbing(
    config_path: Path,
    *,
    arm: str,
    instances: int,
    allow_live_calls: bool,
    scientific_result: bool,
    run_id: str = "phase12-pilot-a-plumbing",
    artifact_root: Path | None = None,
    evidence_root: Path | None = None,
    client_factory: Callable[[], object] | None = None,
    context_factory: Callable[[LLMClient, str, str], Game24RuntimeContext] | None = None,
) -> dict[str, object]:
    _validate_plumbing_request(arm, instances, allow_live_calls, scientific_result)
    payload = _load_config(config_path)
    cost_preview(config_path)
    _validate_run_id(run_id)
    provider = ProviderConfig.from_run_config(payload)
    if provider.provider != "openai_responses" or not provider.live_calls_enabled:
        raise PilotALaunchError("LIVE_CALL_CONFIG_REQUIRED")
    client = (
        client_factory()
        if client_factory is not None
        else build_llm_client(
            provider, stage="pilot", execution_class="live", allow_live_calls=allow_live_calls
        )
    )
    live_client = cast(LLMClient, client)
    context = (
        context_factory(live_client, run_id, payload["provider"]["model_id"])
        if context_factory is not None
        else _live_context(live_client, run_id, payload["provider"]["model_id"])
    )
    if context.branch != "clean" or context.task.task_name != "game24":
        raise PilotALaunchError("CLEAN_GAME24_CONTEXT_REQUIRED")
    root = artifact_root or _artifact_root()
    run_dir = root / "runs" / run_id
    if run_dir.exists():
        raise PilotALaunchError("RUN_ID_ALREADY_EXISTS")
    run_dir.mkdir(parents=True)
    try:
        results = run_clean_game24_trial(context)
        _write_plumbing_archive(run_dir, config_path, provider, context, results, client)
        report = validate_plumbing_archive(run_dir)
    except Exception:
        _write_json(run_dir / "failed.json", {"scientific_result": False, "status": "failed"})
        raise
    if report["overall"] != "pass":
        raise PilotALaunchError(str(report["reason_code"]))
    evidence = evidence_root or DEFAULT_EVIDENCE_ROOT
    _write_json(evidence / "t7-plumbing.json", report)
    return report


def validate_plumbing_archive(run_dir: Path) -> dict[str, object]:
    required = (
        "run.json",
        "resolved_config.json",
        "provider_profile.json",
        "trials.jsonl",
        "calls.jsonl",
        "failures.jsonl",
        "retrieval_events.jsonl",
        "context_events.jsonl",
        "decision_ledger.json",
        "public_artifact_manifest.json",
    )
    if any(not (run_dir / name).is_file() for name in required):
        return _failed_archive(run_dir, "REQUIRED_ARTIFACT_MISSING")
    manifest = _read_json(run_dir / "public_artifact_manifest.json")
    if not isinstance(manifest, dict) or manifest.get("status") != "completed":
        return _failed_archive(run_dir, "ARCHIVE_MANIFEST_INVALID")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        return _failed_archive(run_dir, "ARCHIVE_MANIFEST_INVALID")
    mismatches = sum(
        not isinstance(record, dict)
        or record.get("sha256") != _sha256(run_dir / name)
        or record.get("count") != _line_count(run_dir / name)
        for name, record in artifacts.items()
        if (run_dir / name).is_file()
    )
    expected_artifacts = {name for name in required if name != "public_artifact_manifest.json"}
    if mismatches or set(artifacts) != expected_artifacts:
        return _failed_archive(run_dir, "ARCHIVE_HASH_MISMATCH", mismatches)
    payloads = {name: _read_archive(run_dir / name) for name in artifacts}
    if any(value is None for value in payloads.values()) or _SECRET_PATTERN.search(json.dumps(payloads)):
        return _failed_archive(run_dir, "SECRET_OR_JSON_INVALID")
    trials = payloads["trials.jsonl"]
    calls = payloads["calls.jsonl"]
    ledger = payloads["decision_ledger.json"]
    if not isinstance(trials, list) or not isinstance(calls, list) or not isinstance(ledger, dict):
        return _failed_archive(run_dir, "ARCHIVE_JSON_INVALID")
    baselines = [row.get("baseline") for row in trials if isinstance(row, dict)]
    if baselines != list(PLUMBING_BASELINES):
        return _failed_archive(run_dir, "BASELINE_PANEL_MISMATCH")
    if any(row.get("arm") != "Clean" or row.get("scientific_result") is not False for row in trials if isinstance(row, dict)):
        return _failed_archive(run_dir, "PLUMBING_SCOPE_VIOLATION")
    reference_failure = _validate_plumbing_references(
        trials,
        calls,
        payloads["retrieval_events.jsonl"],
        payloads["context_events.jsonl"],
    )
    if reference_failure is not None:
        return _failed_archive(run_dir, reference_failure)
    stages = {
        baseline: [row.get("stage") for row in calls if isinstance(row, dict) and row.get("baseline") == baseline]
        for baseline in PLUMBING_BASELINES
    }
    expected_bot_stages = ["bot_problem_distill", "bot_instantiate_solve", "bot_thought_distill"]
    if not stages["bot_style"] or stages["bot_style"] != expected_bot_stages[: len(stages["bot_style"])]:
        return _failed_archive(run_dir, "BOT_CALL_ORDERING_FAILED")
    reflexion = stages["reflexion_style"]
    if not reflexion or reflexion[0] != "reflexion_generate":
        return _failed_archive(run_dir, "REFLEXION_CALL_ORDERING_FAILED")
    rag_calls = [row for row in calls if isinstance(row, dict) and row.get("baseline") == "rag_frozen"]
    if len(rag_calls) != 1 or not rag_calls[0].get("source_span_ids"):
        return _failed_archive(run_dir, "ANSWER_SOURCE_SPAN_JOIN_FAILED")
    if ledger.get("cost_total", 5.0) >= 5.0 or ledger.get("scientific_result") is not False:
        return _failed_archive(run_dir, "OPERATIONS_RECONCILIATION_FAILED")
    if not _operations_reconcile(ledger, calls):
        return _failed_archive(run_dir, "OPERATIONS_RECONCILIATION_FAILED")
    provider_profile = payloads["provider_profile.json"]
    if not isinstance(provider_profile, dict):
        return _failed_archive(run_dir, "ARCHIVE_JSON_INVALID")
    return {
        "arm": "Clean",
        "artifacts": artifacts,
        "baselines": list(PLUMBING_BASELINES),
        "cost_total": ledger["cost_total"],
        "hash_mismatches": 0,
        "instance_count": 1,
        "live_provider_calls": len(calls),
        "manifest_hash": _sha256(run_dir / "public_artifact_manifest.json"),
        "model": provider_profile["model"],
        "overall": "pass",
        "provider": provider_profile["provider"],
        "reason_code": None,
        "retry_total": ledger["retry_total"],
        "run_dir": str(run_dir),
        "scientific_result": False,
        "secret_findings": 0,
        "unresolved_references": 0,
    }


def _validate_plumbing_references(
    trials: list[object], calls: list[object], retrieval_events: object, context_events: object
) -> str | None:
    if not all(isinstance(row, dict) for row in trials):
        return "ARCHIVE_JSON_INVALID"
    trial_rows = [cast(dict[str, object], row) for row in trials]
    trial_ids = [row.get("trial_id") for row in trial_rows]
    if any(not isinstance(trial_id, str) for trial_id in trial_ids) or len(set(trial_ids)) != len(trial_ids):
        return "TRIAL_CALL_REFERENCE_FAILED"
    if not all(isinstance(row, dict) for row in calls):
        return "ARCHIVE_JSON_INVALID"
    calls_by_trial: dict[str, list[str]] = {trial_id: [] for trial_id in trial_ids if isinstance(trial_id, str)}
    call_ids: set[str] = set()
    for row in calls:
        call = cast(dict[str, object], row)
        call_id = call.get("call_id")
        trial_id = call.get("trial_id")
        if not isinstance(call_id, str) or call_id in call_ids or not isinstance(trial_id, str) or trial_id not in calls_by_trial:
            return "CALL_TRIAL_REFERENCE_FAILED"
        call_ids.add(call_id)
        calls_by_trial[trial_id].append(call_id)
    for trial in trial_rows:
        trial_id = cast(str, trial["trial_id"])
        declared_calls = trial.get("calls")
        answer_call_id = trial.get("answer_call_id")
        if (
            not isinstance(declared_calls, list)
            or any(not isinstance(call_id, str) for call_id in declared_calls)
            or len(set(declared_calls)) != len(declared_calls)
            or set(declared_calls) != set(calls_by_trial[trial_id])
        ):
            return "TRIAL_CALL_REFERENCE_FAILED"
        if (
            not isinstance(answer_call_id, str)
            or _resolve_answer_call_id(answer_call_id, trial.get("baseline"), declared_calls) is None
        ):
            return "ANSWER_CALL_REFERENCE_FAILED"
    for events in (retrieval_events, context_events):
        if not isinstance(events, list) or any(
            not isinstance(event, dict)
            or not _event_trial_id_resolves(event.get("trial_id"), calls_by_trial)
            for event in events
        ):
            return "EVENT_TRIAL_REFERENCE_FAILED"
    return None


def _resolve_answer_call_id(
    answer_call_id: str, baseline: object, declared_calls: list[str]
) -> str | None:
    if answer_call_id in declared_calls:
        return answer_call_id
    match = re.fullmatch(r".+:call:(\d+)", answer_call_id)
    if not isinstance(baseline, str) or match is None:
        return None
    normalized_call_id = f"{baseline}:{match.group(1)}"
    return normalized_call_id if normalized_call_id in declared_calls else None


def _event_trial_id_resolves(event_trial_id: object, calls_by_trial: dict[str, list[str]]) -> bool:
    if isinstance(event_trial_id, str) and event_trial_id in calls_by_trial:
        return True
    prefixes = {trial_id.rpartition(":")[0] for trial_id in calls_by_trial}
    return (
        isinstance(event_trial_id, str)
        and len(prefixes) == 1
        and event_trial_id == f"{prefixes.pop()}:build-1"
    )


def _operations_reconcile(ledger: dict[str, object], calls: list[object]) -> bool:
    if not all(isinstance(call, dict) for call in calls):
        return False
    call_rows = [cast(dict[str, object], call) for call in calls]
    retry_counts = [call.get("retry_count") for call in call_rows]
    if (
        any(type(retry_count) is not int or retry_count < 0 for retry_count in retry_counts)
        or type(ledger.get("live_provider_calls")) is not int
        or type(ledger.get("retry_total")) is not int
        or type(ledger.get("cost_total")) not in {int, float}
    ):
        return False
    counts_and_retries_match = (
        ledger["live_provider_calls"] == len(call_rows)
        and ledger["retry_total"] == sum(cast(int, retry_count) for retry_count in retry_counts)
    )
    cost_presence = ["cost_usd" in call for call in call_rows]
    if not any(cost_presence):
        return counts_and_retries_match and float(cast(float, ledger["cost_total"])) >= 0
    costs = [call.get("cost_usd") for call in call_rows]
    return (
        all(cost_presence)
        and all(type(cost) in {int, float} and float(cast(float, cost)) >= 0 for cost in costs)
        and counts_and_retries_match
        and math.isclose(
            float(cast(float, ledger["cost_total"])),
            sum(float(cast(float, cost)) for cost in costs),
            abs_tol=1e-12,
        )
    )


def evaluate_pilot_a_admission(
    config_path: Path, *, evidence_root: Path | None = None
) -> dict[str, object]:
    _load_config(config_path)
    root = evidence_root or DEFAULT_EVIDENCE_ROOT
    for filename, code in (
        ("t5-f1c.json", "T5_F1C_EVIDENCE_REQUIRED"),
        ("t5-micro-retrieval.json", "T5_RETRIEVAL_EVIDENCE_REQUIRED"),
        ("t6-invariants.json", "T6_INVARIANT_EVIDENCE_REQUIRED"),
        ("t6-archive.json", "T6_ARCHIVE_EVIDENCE_REQUIRED"),
    ):
        if not _passing_json(root / filename):
            return _blocked_admission(code)
    plumbing = root / "t7-plumbing.json"
    if not _passing_json(plumbing) or not _valid_plumbing_report(_read_json(plumbing)):
        return _blocked_admission("T7_PLUMBING_EVIDENCE_REQUIRED")
    return {"admitted": True, "reason_code": None, "scientific_result": False}


def run_scientific_pilot_a(
    config_path: Path,
    *,
    allow_live_calls: bool,
    artifact_root: Path | None = None,
    evidence_root: Path | None = None,
    client_factory: Callable[[], object] | None = None,
    context_factory: Callable[[LLMClient, str, str], Game24RuntimeContext] | None = None,
    run_id: str | None = None,
) -> dict[str, object]:
    admission = evaluate_pilot_a_admission(config_path, evidence_root=evidence_root)
    if admission["admitted"] is not True:
        raise PilotALaunchError(str(admission["reason_code"]))
    from memcontam.readiness.pilot_a_scientific import (
        ScientificPilotAError,
        run_scientific_pilot_a as execute,
    )

    try:
        return execute(
            config_path,
            allow_live_calls=allow_live_calls,
            artifact_root=artifact_root,
            client_factory=client_factory,
            context_factory=context_factory,
            run_id=run_id,
        )
    except ScientificPilotAError as error:
        raise PilotALaunchError(error.code) from error


def validate_scientific_pilot_a_archive(run_dir: Path) -> dict[str, object]:
    from memcontam.readiness.pilot_a_scientific_archive import validate_scientific_archive

    return validate_scientific_archive(run_dir)


def write_handoff(
    config_path: Path,
    *,
    evidence_root: Path = DEFAULT_EVIDENCE_ROOT,
    destination: Path | None = None,
) -> dict[str, object]:
    preview = cost_preview(config_path)
    admission = evaluate_pilot_a_admission(config_path, evidence_root=evidence_root)
    plumbing_path = evidence_root / "t7-plumbing.json"
    handoff = {
        "config_hash": _sha256(config_path),
        "exact_human_launch_command": _human_launch_command(config_path),
        "estimated_pilot_a_max_cost_usd": preview["projected_max_cost_usd"],
        "f1c_evidence": _evidence_ref(evidence_root / "t5-f1c.json"),
        "implementation_commit": _implementation_commit(),
        "invariant_report": _evidence_ref(evidence_root / "t6-invariants.json"),
        "no_live_call_constraint": (
            None
            if admission["admitted"] is True
            else "NO_LIVE_OPENAI_REQUESTS_IN_ACTIVE_SESSION"
        ),
        "plumbing_archive": _evidence_ref(plumbing_path) if _passing_json(plumbing_path) else None,
        "reason_code": admission["reason_code"],
        "scientific_result": False,
        "status": (
            "READY_FOR_HUMAN_AUTHORIZED_PILOT_A"
            if admission["admitted"] is True
            else "BLOCKED_AWAITING_HUMAN_AUTHORIZATION"
        ),
    }
    output = destination or evidence_root / "t7-handoff.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(handoff, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return handoff


def _load_config(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and payload.get("config_kind") == "phase12_pilot_a_preflight_v1":
            load_preflight_config(path)
    except (OSError, PreflightError, yaml.YAMLError) as error:
        raise PilotALaunchError("INVALID_PILOT_A_CONFIG") from error
    if not isinstance(payload, dict) or payload.get("config_kind") not in {
        "phase12_pilot_a_preflight_v1",
        "phase12_pilot_a_scientific_v1",
    }:
        raise PilotALaunchError("INVALID_PILOT_A_CONFIG")
    return payload


def _cost_guard(payload: dict[str, Any]) -> CostGuard:
    cost = payload["cost"]
    return CostGuard(
        input_per_million_usd=cost["input_per_1m_tokens"],
        cached_input_per_million_usd=cost["cached_input_per_1m_tokens"],
        output_per_million_usd=cost["output_per_1m_tokens"],
        hard_ceiling_usd=cost["hard_ceiling"],
        warning_usd=cost["warning"],
    )


def _validate_plumbing_request(
    arm: str, instances: int, allow_live_calls: bool, scientific_result: bool
) -> None:
    if arm != "Clean":
        raise PilotALaunchError("CLEAN_ARM_REQUIRED")
    if instances != 1:
        raise PilotALaunchError("ONE_INSTANCE_REQUIRED")
    if scientific_result:
        raise PilotALaunchError("PLUMBING_SCIENTIFIC_RESULT_FORBIDDEN")
    if not allow_live_calls:
        raise PilotALaunchError("LIVE_CALL_AUTHORIZATION_REQUIRED")


def _blocked_admission(code: str) -> dict[str, object]:
    return {"admitted": False, "reason_code": code, "scientific_result": False}


def _passing_json(path: Path) -> bool:
    payload = _read_json(path)
    return isinstance(payload, dict) and payload.get("overall") == "pass"


def _valid_plumbing_report(payload: object) -> bool:
    return isinstance(payload, dict) and (
        payload.get("arm") == "Clean"
        and payload.get("instance_count") == 1
        and payload.get("baselines") == list(PLUMBING_BASELINES)
        and payload.get("scientific_result") is False
    )


def _read_json(path: Path) -> object | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def _evidence_ref(path: Path) -> dict[str, str] | None:
    return {"path": str(path), "sha256": _sha256(path)} if path.is_file() else None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _implementation_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _human_launch_command(config_path: Path) -> str:
    return (
        "python -m memcontam.cli phase12 pilot-a "
        f"--config {config_path} --allow-live-calls"
    )


def _live_context(client: LLMClient, run_id: str, model: str) -> Game24RuntimeContext:
    embedder = BgeM3EmbeddingProvider(cache_folder=resolve_bge_cache_path(), local_files_only=True)
    corpus = CleanCorpus.from_documents(
        [
            {"id": "game24-clean-a", "text": "Use rational intermediate values."},
            {"id": "game24-clean-b", "text": "Check arithmetic exactly."},
            {"id": "game24-clean-c", "text": "Use all four values exactly once."},
        ],
        corpus_id="game24-plumbing-clean",
    )
    branches = build_branch_corpora(
        corpus,
        {
            "false": {"id": "false", "text": "Use only integer intermediates."},
            "correct": {"id": "correct", "text": "Fractions are valid."},
            "irrelevant": {"id": "irrelevant", "text": "Sort alphabetically."},
        },
    )
    indices = build_branch_indices(branches, embedder, filter_policy=None)
    task = build_instance({"sample_id": "game24-build-1", "numbers": [1, 3, 4, 6]})
    return Game24RuntimeContext(
        task=task,
        client=client,
        model=model,
        verifier=lambda answer, seen_task: verify_expression(
            answer, seen_task.input["numbers"], seen_task.verifier_spec["target"]
        ),
        decoding={"temperature": 0.0, "top_p": 1.0, "max_output_tokens": 2048},
        branch="clean",
        identities=RuntimeIdentities(run_id, f"{run_id}:build-1", 1),
        embedding_provider=embedder,
        baseline_configs={"fh_bounded": {"context_window_tokens": 10_000}},
        initial_states={
            "fh_bounded": FullHistoryStateV3(records=[]),
            "rag_frozen": RagFrozenStateV3(
                "clean", branches.branches["clean"], indices.branches["clean"]
            ),
            "bot_style": BoTStateV3(
                entries=cast(list[MemoryEntry | NativeEntry], [
                    MemoryEntry(
                        entry_id="bot-clean-a",
                        content="Use rational intermediate values.",
                        memory_type="thought_template",
                        metadata={
                            "description": "Use rational intermediate values.",
                            "category": "procedure-based",
                        },
                    ),
                    MemoryEntry(
                        entry_id="bot-clean-b",
                        content="Check arithmetic exactly.",
                        memory_type="thought_template",
                        metadata={"description": "Check arithmetic exactly.", "category": "procedure-based"},
                    ),
                ]),
                clean_competitor_ids=("bot-clean-a", "bot-clean-b"),
                active_capacity=3,
            ),
            "reflexion_style": ReflexionStateV3(reflections=[], active_capacity=3),
        },
    )


def _write_plumbing_archive(
    run_dir: Path,
    config_path: Path,
    provider: ProviderConfig,
    context: Game24RuntimeContext,
    results: dict[str, Any],
    client: object,
) -> None:
    trials: list[dict[str, object]] = []
    calls: list[dict[str, object]] = []
    retrieval: list[dict[str, object]] = []
    contexts: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for baseline in PLUMBING_BASELINES:
        result = results[baseline]
        outcome = result.outcome
        calls_for_trial = []
        trial_id = f"{run_dir.name}:{baseline}"
        call_id_map: dict[str, str] = {}
        response_hashes: dict[str, str] = {}
        for index, call in enumerate(outcome.method_calls, start=1):
            source_ids = [
                span.entry_id
                for span in call.source_spans
                if getattr(span, "entry_id", None) is not None
            ]
            call_id = f"{baseline}:{index}"
            call_id_map[str(call.call_id)] = call_id
            response_hash = hashlib.sha256(call.raw_response.encode("utf-8")).hexdigest()
            response_hashes[call_id] = response_hash
            calls.append(
                {
                    "baseline": baseline,
                    "call_id": call_id,
                    "cost_usd": _call_cost(call.token_usage, provider),
                    "latency_ms": call.latency_ms,
                    "messages": call.messages,
                    "retry_count": call.retry_count,
                    "response_text": call.raw_response,
                    "response_text_sha256": response_hash,
                    "source_span_ids": source_ids,
                    "stage": call.stage,
                    "token_usage": dict(call.token_usage),
                    "trial_id": trial_id,
                }
            )
            calls_for_trial.append(call_id)
        answer_call_id = call_id_map.get(str(outcome.answer_call_id))
        trials.append(
            {
                "answer_call_id": answer_call_id,
                "arm": "Clean",
                "baseline": baseline,
                "calls": calls_for_trial,
                "metadata": _jsonable(outcome.metadata),
                "scientific_result": False,
                "status": outcome.status,
                "trial_id": trial_id,
            }
        )
        if result.retrieval_event is not None:
            retrieval.append(_plumbing_event(result.retrieval_event, trial_id, "retrieval"))
        if result.context_event is not None:
            contexts.append(_plumbing_event(result.context_event, trial_id, "context"))
        if outcome.status != "succeeded":
            parser_contracts = {
                "nomem": FINAL_ANSWER_PARSER_VERSION,
                "fh_bounded": FINAL_ANSWER_PARSER_VERSION,
                "rag_frozen": FINAL_ANSWER_PARSER_VERSION,
                "bot_style": "bot_solve_json_v1",
                "reflexion_style": FINAL_ANSWER_PARSER_VERSION,
            }
            prompt_contracts = {
                "nomem": FINAL_ANSWER_PROMPT_VERSION,
                "fh_bounded": FINAL_ANSWER_PROMPT_VERSION,
                "rag_frozen": FINAL_ANSWER_PROMPT_VERSION,
                "bot_style": BOT_SOLVE_PROMPT_VERSION,
                "reflexion_style": FINAL_ANSWER_PROMPT_VERSION,
            }
            failures.append(
                {
                    "answer_call_id": answer_call_id,
                    "baseline": baseline,
                    "call_ids": calls_for_trial,
                    "error_type": outcome.error_type,
                    "failure_class": outcome.failure_disposition,
                    "parser_contract": parser_contracts[baseline],
                    "prompt_contract": prompt_contracts[baseline],
                    "raw_response_hash": response_hashes.get(answer_call_id or ""),
                    "scientific_ineligibility_reason": outcome.scientific_ineligibility_reason,
                    "trial_id": trial_id,
                }
            )
    cost_total = sum(float(cast(float, call["cost_usd"])) for call in calls)
    retry_total = sum(
        int(retry_count)
        for call in calls
        if isinstance((retry_count := call.get("retry_count")), int)
    )
    _write_json(
        run_dir / "run.json",
        {
            "run_id": run_dir.name,
            "status": "completed",
            "scientific_result": False,
            "evidence_layer": "build",
        },
    )
    _write_json(
        run_dir / "resolved_config.json",
        {"config_sha256": _sha256(config_path), "execution_class": "live", "scientific_result": False},
    )
    _write_json(
        run_dir / "provider_profile.json",
        {"model": context.model, "provider": provider.provider, "service_tier": provider.service_tier},
    )
    _write_jsonl(run_dir / "trials.jsonl", trials)
    _write_jsonl(run_dir / "calls.jsonl", calls)
    _write_jsonl(run_dir / "failures.jsonl", failures)
    _write_jsonl(run_dir / "retrieval_events.jsonl", retrieval)
    _write_jsonl(run_dir / "context_events.jsonl", contexts)
    _write_json(
        run_dir / "decision_ledger.json",
        {
            "cost_total": cost_total,
            "live_provider_calls": len(calls),
            "retry_total": retry_total,
            "scientific_result": False,
        },
    )
    files = (
        "run.json",
        "resolved_config.json",
        "provider_profile.json",
        "trials.jsonl",
        "calls.jsonl",
        "failures.jsonl",
        "retrieval_events.jsonl",
        "context_events.jsonl",
        "decision_ledger.json",
    )
    _write_json(
        run_dir / "public_artifact_manifest.json",
        {
            "artifacts": {
                name: {"count": _line_count(run_dir / name), "sha256": _sha256(run_dir / name)}
                for name in files
            },
            "status": "completed",
        },
    )


def _call_cost(token_usage: dict[str, int], provider: ProviderConfig) -> float:
    prompt_tokens = token_usage.get("prompt_tokens", 0)
    cached_tokens = token_usage.get("cached_prompt_tokens", 0)
    completion_tokens = token_usage.get("completion_tokens", 0)
    if (
        any(type(value) is not int or value < 0 for value in (prompt_tokens, cached_tokens, completion_tokens))
        or cached_tokens > prompt_tokens
    ):
        raise PilotALaunchError("COST_ACCOUNTING_INVALID")
    return (
        (prompt_tokens - cached_tokens) * provider.input_per_million_usd
        + cached_tokens * provider.cached_input_per_million_usd
        + completion_tokens * provider.output_per_million_usd
    ) / 1_000_000


def _plumbing_event(event: object, trial_id: str, kind: str) -> dict[str, object]:
    payload = _jsonable(event)
    event_id = f"{trial_id}:{kind}"
    payload["event_id"] = event_id
    payload["trial_id"] = trial_id
    payload[f"{kind}_id"] = event_id
    return payload


def _artifact_root() -> Path:
    value = os.environ.get("MEMCONTAM_ARTIFACT_ROOT")
    if not value:
        raise PilotALaunchError("MEMCONTAM_ARTIFACT_ROOT_REQUIRED")
    root = Path(value)
    if not root.is_absolute() or not root.is_dir():
        raise PilotALaunchError("MEMCONTAM_ARTIFACT_ROOT_REQUIRED")
    return root


def _validate_run_id(run_id: str) -> None:
    candidate = Path(run_id)
    if candidate.is_absolute() or ".." in candidate.parts or len(candidate.parts) != 1:
        raise PilotALaunchError("INVALID_RUN_ID")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_archive(path: Path) -> object | None:
    if path.suffix == ".jsonl":
        try:
            return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
    return _read_json(path)


def _line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines()) if path.suffix == ".jsonl" else 1


def _jsonable(value: object) -> dict[str, object]:
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        payload = dump(mode="json")
        if isinstance(payload, dict):
            return payload
    if isinstance(value, dict):
        return value
    raise PilotALaunchError("ARCHIVE_SERIALIZATION_FAILED")


def _failed_archive(run_dir: Path, code: str, hash_mismatches: int = 0) -> dict[str, object]:
    return {
        "artifacts": {},
        "hash_mismatches": hash_mismatches,
        "overall": "fail",
        "reason_code": code,
        "run_dir": str(run_dir),
        "scientific_result": False,
        "unresolved_references": 1,
    }
