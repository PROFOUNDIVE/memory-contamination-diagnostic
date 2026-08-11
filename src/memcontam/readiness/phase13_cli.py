from __future__ import annotations

import argparse
import json
from pathlib import Path

from memcontam.clients.config import ProviderConfig
from memcontam.clients.cost_guard import CostGuard
from memcontam.clients.openai_responses import OpenAIResponsesClient
from memcontam.memory.embeddings import BgeM3EmbeddingProvider
from memcontam.readiness.phase13_clean_prefix import (
    Phase13CalibrationError,
    load_clean_prefix_config_bytes,
    prepare_clean_prefix,
)
from memcontam.readiness.phase13_clean_prefix_authorization import verify_authorization
from memcontam.readiness.phase13_clean_prefix_runtime import execute_clean_prefix_calibration
from memcontam.readiness.retrieval_smoke import resolve_bge_cache_path


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    phase13 = subparsers.add_parser("phase13")
    commands = phase13.add_subparsers(dest="phase13_command", required=True)
    prepare = commands.add_parser("prepare-clean-prefix")
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--run-id", required=True)
    prepare.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run-clean-prefix")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--run-id", required=True)
    run.add_argument("--request", type=Path, required=True)
    run.add_argument("--authorization", type=Path, required=True)
    run.add_argument("--expected-authorization-sha256", required=True)
    run.add_argument("--allow-live-calls", action="store_true")


def run(args: argparse.Namespace) -> None:
    try:
        if args.phase13_command == "prepare-clean-prefix":
            payload = prepare_clean_prefix(args.config, args.run_id, args.output)
            print(json.dumps(payload, sort_keys=True))
            return
        verified = verify_authorization(
            config_path=args.config,
            run_id=args.run_id,
            request_path=args.request,
            authorization_path=args.authorization,
            expected_authorization_sha256=args.expected_authorization_sha256,
            allow_live_calls=args.allow_live_calls,
        )
        config = load_clean_prefix_config_bytes(verified.config_bytes)
        provider = ProviderConfig.from_run_config(config)
        guard = CostGuard(
            input_per_million_usd=provider.input_per_million_usd,
            cached_input_per_million_usd=provider.cached_input_per_million_usd,
            output_per_million_usd=provider.output_per_million_usd,
            warning_usd=40.0,
            hard_ceiling_usd=config["budget"]["hard_ceiling_microusd"] / 1_000_000,
        )
        client = OpenAIResponsesClient(
            provider,
            allow_live_calls=True,
            cost_guard=guard,
        )
        embedder = BgeM3EmbeddingProvider(
            cache_folder=resolve_bge_cache_path(),
            local_files_only=True,
        )
        result = execute_clean_prefix_calibration(
            args.config,
            args.run_id,
            client=client,
            embedding_provider=embedder,
            request_path=args.request,
            authorization_path=args.authorization,
            expected_authorization_sha256=args.expected_authorization_sha256,
            allow_live_calls=args.allow_live_calls,
        )
        print(json.dumps(result, sort_keys=True))
    except Phase13CalibrationError as error:
        raise SystemExit(error.code) from error
