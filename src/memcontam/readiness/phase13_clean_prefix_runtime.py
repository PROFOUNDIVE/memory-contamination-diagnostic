from __future__ import annotations

from pathlib import Path
from typing import Any

from memcontam.clients.base import LLMClient
from memcontam.experiment.phase12.live_prefix import run_live_clean_prefix
from memcontam.rag.branch_index import EmbeddingProvider

from memcontam.readiness.phase13_clean_prefix import (
    SEEDS,
    TASKS,
    Phase13CalibrationError,
    load_clean_prefix_config_bytes,
    resolve_output_root,
)
from memcontam.readiness.phase13_clean_prefix_authorization import (
    assert_execution_state,
    verify_authorization,
)
from memcontam.readiness.phase13_clean_prefix_metering import MeteredClient
from .phase13_clean_prefix_archive import (
    append_trajectory_records,
    rates,
    write_archive,
)
from .phase13_clean_prefix_context import (
    build_contexts,
    conditions,
    load_corpus_rows,
    load_instances,
)


def execute_clean_prefix_calibration(
    config_path: Path,
    run_id: str,
    *,
    client: LLMClient,
    embedding_provider: EmbeddingProvider,
    artifact_root: Path | None = None,
    request_path: Path | None = None,
    authorization_path: Path | None = None,
    expected_authorization_sha256: str | None = None,
    allow_live_calls: bool = False,
) -> dict[str, Any]:
    if (
        request_path is None
        or authorization_path is None
        or expected_authorization_sha256 is None
    ):
        raise Phase13CalibrationError("CALIBRATION_AUTHORIZATION_REQUIRED")
    verified = verify_authorization(
        config_path=config_path,
        run_id=run_id,
        request_path=request_path,
        authorization_path=authorization_path,
        expected_authorization_sha256=expected_authorization_sha256,
        allow_live_calls=allow_live_calls,
    )
    assert_execution_state(verified)
    config = load_clean_prefix_config_bytes(verified.config_bytes)
    authorized_root = resolve_output_root(config)
    if artifact_root is not None and artifact_root.resolve() != authorized_root:
        raise Phase13CalibrationError("CALIBRATION_OUTPUT_ROOT_MISMATCH")
    root = authorized_root
    run_dir = root / run_id
    if run_dir.exists():
        raise Phase13CalibrationError("RUN_ID_ALREADY_EXISTS")
    run_dir.mkdir(parents=True)
    metered = MeteredClient(client, config)
    trials: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    eligibility: list[dict[str, Any]] = []
    seed_status: list[dict[str, Any]] = []
    try:
        instances = load_instances(config)
        corpus_rows = load_corpus_rows(config)
        for task in TASKS:
            for seed in SEEDS:
                contexts = build_contexts(
                    config,
                    run_id,
                    task,
                    seed,
                    instances[task],
                    corpus_rows[task],
                    metered,
                    embedding_provider,
                )
                result = run_live_clean_prefix(
                    seed=seed,
                    contexts=contexts,
                    conditions=conditions(),
                    suffix_horizon=1,
                )
                append_trajectory_records(
                    task, seed, result, trials, calls, checkpoints, eligibility
                )
                seed_status.append(
                    {
                        "task": task,
                        "seed": seed,
                        "status": "eligible" if not result.selection.blocked else "ineligible",
                    }
                )
        frozen_rates = rates(seed_status)
        write_archive(
            run_dir,
            verified.config_bytes,
            config,
            run_id,
            {
                "trials": trials,
                "calls": calls,
                "checkpoints": checkpoints,
                "eligibility": eligibility,
                "seed_status": seed_status,
                "failures": [],
            },
            frozen_rates,
            _accounting(metered),
            verified.request_bytes,
            verified.authorization_bytes,
        )
    except Exception as error:  # noqa: BLE001  # noqa: BROAD_EXCEPT_OK
        failure_code = (
            error.code
            if isinstance(error, Phase13CalibrationError)
            else "CALIBRATION_RUNTIME_FAILURE"
        )
        write_archive(
            run_dir,
            verified.config_bytes,
            config,
            run_id,
            {
                "trials": trials,
                "calls": metered.call_records,
                "checkpoints": checkpoints,
                "eligibility": eligibility,
                "seed_status": seed_status,
                "failures": [
                    {"code": failure_code, "exception_type": type(error).__name__}
                ],
            },
            rates(seed_status),
            _accounting(metered),
            verified.request_bytes,
            verified.authorization_bytes,
            status="invalidated",
        )
        raise
    return {
        "status": "completed",
        "run_dir": str(run_dir),
        "trajectory_count": len(seed_status),
        "filter_calls": 0,
        "rates": frozen_rates,
    }


def _accounting(metered: MeteredClient) -> dict[str, Any]:
    return {
        "semantic_calls": metered.semantic_calls,
        "semantic_calls_dispatched": metered.semantic_calls_dispatched,
        "transport_attempts_observed": metered.transport_attempts,
        "input_tokens_observed": metered.input_tokens,
        "output_tokens_observed": metered.output_tokens,
        "cost_usd_observed": metered.cost_usd,
        "reserved_max_cost_usd": metered.reserved_max_cost_usd,
        "filter_calls": 0,
    }
