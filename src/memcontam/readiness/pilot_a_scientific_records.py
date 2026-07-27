from __future__ import annotations

import hashlib
import subprocess
from typing import Any

from memcontam.clients.config import ProviderConfig
from memcontam.experiment.phase12.live_suffix import LiveSuffixTrial


ROW_NAMES = (
    "trials", "calls", "retrieval_events", "context_events",
    "failures",
    "memory_events",
    "admission_events",
    "intervention_events",
    "checkpoint_events",
    "eligibility_events",
    "audit_labels",
    "seed_status",
)


def record_prefix(rows, seed, contexts, prefix, provider) -> None:  # noqa: ANN001
    for baseline, results in prefix.trial_results_by_baseline.items():
        for context, result in zip(contexts, results, strict=True):
            _record_trial(
                rows,
                seed,
                baseline,
                "clean",
                "branch_free_prefix",
                context.task.sample_id,
                result.outcome,
                provider,
            )
    for decision in prefix.selection.decisions:
        baseline = decision.condition_id.removesuffix("-clean")
        task_id = contexts[decision.checkpoint_index - 1].task.sample_id
        trial_id = f"seed:{seed}:branch_free_prefix:{baseline}:clean:{task_id}"
        rows["eligibility_events"].append(
            _event(
                trial_id,
                "eligibility",
                baseline=baseline,
                baseline_family=decision.baseline_family,
                checkpoint_id=decision.checkpoint_id,
                checkpoint_index=decision.checkpoint_index,
                condition_id=decision.condition_id,
                eligible=decision.eligible,
                horizon=decision.horizon,
                reason_codes=list(decision.reason_codes),
                seed=seed,
            )
        )
    joint = prefix.selection.joint_eligibility
    rows["seed_status"].append(
        {
            "seed": seed,
            "eligible": not prefix.selection.blocked,
            "status": "blocked" if prefix.selection.blocked else "selected",
            "reason": prefix.selection.block_reason,
            "selected_checkpoint": prefix.selection.selected_trial_index,
            "fallback_checkpoint_used": False,
            "joint_eligible_indices": list(joint.joint_eligible_indices),
            "primary_intersection": list(joint.primary_intersection),
            "baseline_eligible": joint.baseline_eligible,
            "ineligibility_reasons": {
                baseline: list(reasons) for baseline, reasons in joint.ineligibility_reasons.items()
            },
            "not_estimable": joint.not_estimable,
        }
    )


def record_suffix(rows, seed, memory_runs, nomem_trials, branches, provider) -> None:  # noqa: ANN001
    for baseline, run in memory_runs.items():
        for trial in run.trials:
            _record_suffix_trial(rows, seed, trial, provider)
        selected_trial = next(trial for trial in run.trials if trial.arm == "clean")
        checkpoint = branches[baseline].arms["clean"].checkpoint
        rows["checkpoint_events"].append(
            _event(
                _trial_id(seed, selected_trial),
                "checkpoint",
                checkpoint_id=checkpoint.identity.checkpoint_id,
            )
        )
        for event in branches[baseline].events:
            if event.kind == "intervention_applied":
                trial = next(item for item in run.trials if item.arm == event.arm)
                rows["intervention_events"].append(
                    _event(
                        _trial_id(seed, trial),
                        "intervention",
                        arm=event.arm,
                        candidate_triplet_id=event.candidate_triplet_id,
                        native_render_id=event.native_render_id,
                        injected_root_id=event.injected_root_id,
                        source_identity=event.source_identity,
                    )
                )
        filtered = branches[baseline].arms["filter"]
        filter_trial = next(item for item in run.trials if item.arm == "filter")
        decision = next(
            item
            for item in filtered.filter_state.decisions
            if item.entry_id == filtered.injected_root_id
        )
        rows["admission_events"].append(
            _event(
                _trial_id(seed, filter_trial),
                "admission",
                entry_id=filtered.injected_root_id,
                decision=decision.state,
                reason=decision.reason,
                policy_version="operational-evidence-filter-v4",
            )
        )
        rows["audit_labels"].append(
            {
                "baseline": baseline,
                "candidate_triplet_id": "game24-fraction-intermediate-v1",
                "injected_root_id": filtered.injected_root_id,
                "seed": seed,
            }
        )
    for trial in nomem_trials:
        _record_suffix_trial(rows, seed, trial, provider)


def artifacts(config_path, config, run_id, provider, rows, parent_run_id=None, run_status="completed", status_reason=None):  # noqa: ANN001
    seeds = [int(item["seed"]) for item in config["trajectory_seeds"]]
    eligible = [item["seed"] for item in rows["seed_status"] if item["eligible"]]
    cost_total = sum(call["cost_usd"] for call in rows["calls"])
    retry_total = sum(call["retry_count"] for call in rows["calls"])
    return {
        "run.json": {
            "evidence_layer": "calibration",
            "implementation_commit": _git_head(),
            "parent_run_id": parent_run_id,
            "run_family": "pilot_a",
            "run_id": run_id,
            "status": run_status,
            **({"status_reason": status_reason} if status_reason is not None else {}),
            "scientific_result": True,
        },
        "resolved_config.json": {
            "config": config,
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        },
        "provider_profile.json": {
            "endpoint": "responses",
            "model": config["provider"]["model_id"],
            "provider": provider.provider,
        },
        "decision_ledger.json": {
            "cost_total": cost_total,
            "eligible_seeds": eligible,
            "hard_cost_ceiling_usd": float(config["cost"]["hard_ceiling"]),
            "live_provider_calls": len(rows["calls"]),
            "nomem": {"aliases": 3, "underlying_executions_per_seed": 1},
            "retry_total": retry_total,
            "scientific_result": True,
            "trajectory_seeds": seeds,
            "prefix": {
                "completed_trials": sum(
                    trial["trial_kind"] == "branch_free_prefix" for trial in rows["trials"]
                )
            },
            "eligibility": rows["eligibility_events"],
            "joint": rows["seed_status"],
            "failure": rows["failures"],
            "provenance": {
                "implementation_commit": _git_head(),
                "parent_run_id": parent_run_id,
                "run_id": run_id,
            },
        },
        "trials.jsonl": rows["trials"],
        "calls.jsonl": rows["calls"],
        "retrieval_events.jsonl": rows["retrieval_events"],
        "context_events.jsonl": rows["context_events"],
        "failures.jsonl": rows["failures"],
        "memory_events.jsonl": rows["memory_events"],
        "admission_events.jsonl": rows["admission_events"],
        "intervention_events.jsonl": rows["intervention_events"],
        "checkpoint_events.jsonl": rows["checkpoint_events"],
        "eligibility_events.jsonl": rows["eligibility_events"],
        "seed_status.jsonl": rows["seed_status"],
        "audit/audit_labels.jsonl": rows["audit_labels"],
    }


def record_terminal_failure(rows, error: BaseException, status: str, reason: str) -> None:  # noqa: ANN001
    rows["failures"].append(
        {
            "error_type": type(error).__name__,
            "failure_class": reason,
            "failure_kind": "runner",
            "provenance": "scientific_pilot_a_runner",
            "status": status,
        }
    )


def _record_suffix_trial(rows, seed: int, trial: LiveSuffixTrial, provider: ProviderConfig) -> None:  # noqa: ANN001
    _record_trial(
        rows,
        seed,
        trial.baseline,
        trial.arm,
        "nomem_singleton" if trial.baseline == "nomem" else "memory_branch",
        trial.suffix_id,
        trial.outcome,
        provider,
    )


def _record_trial(rows, seed, baseline, arm, kind, task_id, outcome, provider) -> None:  # noqa: ANN001
    trial_id = f"seed:{seed}:{kind}:{baseline}:{arm}:{task_id}"
    call_ids: list[str] = []
    for call in outcome.method_calls:
        call_id = f"{trial_id}:call:{len(call_ids) + 1}"
        call_ids.append(call_id)
        rows["calls"].append(
            {
                "baseline": baseline,
                "call_id": call_id,
                "cost_usd": _call_cost(call.token_usage, provider),
                "latency_ms": call.latency_ms,
                "messages": call.messages,
                "response_text": call.raw_response,
                "retry_count": call.retry_count,
                "source_span_ids": [span.entry_id for span in call.source_spans if span.entry_id],
                "stage": call.stage,
                "token_usage": dict(call.token_usage),
                "trial_id": trial_id,
            }
        )
    answer_sources: list[str] = next(
        (
            row["source_span_ids"]
            for row in reversed(rows["calls"])
            if row["trial_id"] == trial_id
        ),
        [],
    )
    retrieved_ids = [
        entry.get("entry_id") for entry in outcome.retrieved_memory if entry.get("entry_id")
    ]
    rows["retrieval_events"].append(
        _event(trial_id, "retrieval", retrieved_entry_ids=retrieved_ids)
    )
    rows["context_events"].append(
        _event(trial_id, "context", final_entry_ids=answer_sources)
    )
    rows["trials"].append(
        {
            "answer_call_id": call_ids[-1] if call_ids else None,
            "arm": arm,
            "baseline": baseline,
            "calls": call_ids,
            "context_event_id": f"{trial_id}:context",
            "parsed_answer": outcome.parsed_answer,
            "retrieval_event_id": f"{trial_id}:retrieval",
            "scientific_result": True,
            "seed": seed,
            "status": outcome.status,
            "task_id": task_id,
            "trial_id": trial_id,
            "trial_kind": kind,
            "verifier_result": _jsonable(outcome.verifier_result),
        }
    )
    if outcome.memory_write_event is not None:
        rows["memory_events"].append(
            _event(trial_id, "memory", payload=outcome.memory_write_event)
        )
    if outcome.status == "failed":
        rows["failures"].append(
            {
                "error_type": outcome.error_type,
                "failure_class": outcome.failure_disposition,
                "failure_kind": (
                    "model_behavior"
                    if outcome.error_type == "BaselineOutputError"
                    else "engineering"
                ),
                "trial_id": trial_id,
            }
        )


def _event(trial_id: str, kind: str, **values: Any) -> dict[str, Any]:
    event_id = f"{trial_id}:{kind}"
    return {
        "event_id": event_id,
        "record_type": f"{kind}_event",
        "schema_version": "logging_v3",
        "trial_id": trial_id,
        **values,
    }


def _trial_id(seed: int, trial: LiveSuffixTrial) -> str:
    kind = "nomem_singleton" if trial.baseline == "nomem" else "memory_branch"
    return f"seed:{seed}:{kind}:{trial.baseline}:{trial.arm}:{trial.suffix_id}"


def _call_cost(usage: dict[str, int], provider: ProviderConfig) -> float:
    prompt = usage.get("prompt_tokens", 0)
    cached = usage.get("cached_prompt_tokens", 0)
    completion = usage.get("completion_tokens", 0)
    return (
        (prompt - cached) * provider.input_per_million_usd
        + cached * provider.cached_input_per_million_usd
        + completion * provider.output_per_million_usd
    ) / 1_000_000


def _jsonable(value: Any) -> Any:
    dump = getattr(value, "model_dump", None)
    return dump(mode="json") if callable(dump) else value


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


__all__ = ["ROW_NAMES", "artifacts", "record_prefix", "record_suffix", "record_terminal_failure"]
