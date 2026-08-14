from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from memcontam.readiness.phase13_clean_prefix import (
    Phase13CalibrationError,
    load_clean_prefix_config_bytes,
    prepare_clean_prefix,
)
from memcontam.readiness.phase13_authority import (
    Phase13AuthorityError,
    parse_authority_freeze,
    parse_authority_requirements,
)
from memcontam.readiness.phase13_authority_files import AuthorityFileError, read_regular_nofollow
from memcontam.readiness.phase13_execution_contract import (
    Phase13ExecutionError,
    validate_execution_closure,
)
from memcontam.readiness.phase13_provenance import (
    Phase13ProvenanceError,
    validate_provenance_bundle,
)


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
    authority = commands.add_parser("validate-authority-freeze")
    authority.add_argument("--freeze", type=Path, required=True)
    authority.add_argument("--requirements", type=Path, required=True)
    execution = commands.add_parser("validate-execution-registry")
    execution.add_argument("--root", type=Path, required=True)
    execution.add_argument("--freeze", type=Path, required=True)
    execution.add_argument("--requirements", type=Path, required=True)
    provenance = commands.add_parser("validate-provenance")
    provenance.add_argument("--root", type=Path, required=True)
    provenance.add_argument("--manifest", type=Path, required=True)
    provenance.add_argument("--seal", type=Path, required=True)


def run(args: argparse.Namespace) -> None:
    try:
        if args.phase13_command == "validate-authority-freeze":
            requirements = parse_authority_requirements(read_regular_nofollow(args.requirements))
            freeze = parse_authority_freeze(read_regular_nofollow(args.freeze), requirements)
            print(json.dumps({"freeze_id": freeze.freeze_id, "status": "valid"}, sort_keys=True))
            return
        if args.phase13_command == "validate-execution-registry":
            registry = validate_execution_closure(
                read_regular_nofollow(args.freeze),
                read_regular_nofollow(args.requirements),
                args.root,
            )
            print(json.dumps({"registry_id": registry.registry_id, "status": "valid"}, sort_keys=True))
            return
        if args.phase13_command == "validate-provenance":
            report = validate_provenance_bundle(args.root, args.manifest, args.seal)
            print(json.dumps(asdict(report), sort_keys=True))
            return
        if args.phase13_command == "prepare-clean-prefix":
            payload = prepare_clean_prefix(args.config, args.run_id, args.output)
            print(json.dumps(payload, sort_keys=True))
            return
        from memcontam.clients.config import ProviderConfig
        from memcontam.clients.cost_guard import CostGuard
        from memcontam.clients.openai_responses import OpenAIResponsesClient
        from memcontam.memory.embeddings import BgeM3EmbeddingProvider
        from memcontam.readiness.phase13_clean_prefix_authorization import verify_authorization
        from memcontam.readiness.phase13_clean_prefix_runtime import execute_clean_prefix_calibration
        from memcontam.readiness.retrieval_smoke import resolve_bge_cache_path

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
    except (
        Phase13AuthorityError,
        AuthorityFileError,
        Phase13CalibrationError,
        Phase13ExecutionError,
        Phase13ProvenanceError,
    ) as error:
        raise SystemExit(error.code) from error
