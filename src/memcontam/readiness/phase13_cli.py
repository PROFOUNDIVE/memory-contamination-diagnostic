from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from pydantic import BaseModel

from memcontam.clients.base import LLMClient
from memcontam.readiness.phase13_provider_models import ExecutionTemplateIdentity
from memcontam.readiness.phase13_provider_runtime import Phase13V2ProviderRuntime
from memcontam.readiness.phase13_calibration_v2_runtime import (
    AuthorizedTrajectoryExecution,
    execute_calibration_trajectory,
)

from memcontam.readiness.phase13_clean_prefix import (
    Phase13CalibrationError,
    load_clean_prefix_config_bytes,
    prepare_clean_prefix,
)
from memcontam.readiness.phase13_calibration_v2 import (
    CalibrationV2ConfigError,
    prepare_calibration_v2,
    validate_calibration_v2,
)
from memcontam.readiness.phase13_terminal import (
    CalibrationV2ExternalBlock,
    render_terminal,
)

MAIN_ALIASES = ("main", "main-a", "run-main", "run-main-a", "authorize-main", "request-main")


def build_calibration_v2_provider(
    client: LLMClient,
    root: Path,
    intended_template: ExecutionTemplateIdentity,
) -> Phase13V2ProviderRuntime:
    return Phase13V2ProviderRuntime.from_provider(client, root, intended_template)


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
    for command in (
        "validate-calibration-v2",
        "prepare-calibration-v2",
        "run-calibration-v2",
        "validate-derived-windows",
    ):
        calibration_v2 = commands.add_parser(command)
        calibration_v2.add_argument("--config", type=Path, required=True)
        if command == "run-calibration-v2":
            calibration_v2.add_argument("--report", type=Path)
            calibration_v2.add_argument("--request", type=Path)
            calibration_v2.add_argument("--authorization", type=Path)
            calibration_v2.add_argument("--expected-authorization-sha256")
            calibration_v2.add_argument("--allow-live-calls", action="store_true")
    archive = commands.add_parser("validate-calibration-v2-archive")
    archive.add_argument("--archive", type=Path)
    archive.add_argument("--config", type=Path)
    for command in MAIN_ALIASES:
        commands.add_parser(command)


def run(args: argparse.Namespace) -> None:
    try:
        if args.phase13_command in MAIN_ALIASES:
            raise SystemExit("MAIN_A_EXECUTION_FORBIDDEN")
        if args.phase13_command == "validate-calibration-v2":
            print(render_terminal(validate_calibration_v2(args.config)))
            return
        if args.phase13_command == "prepare-calibration-v2":
            print(render_terminal(prepare_calibration_v2(args.config)))
            print("AWAITING_CALIBRATION_V2_AUTHORIZATION")
            return
        if args.phase13_command == "validate-derived-windows":
            validate_calibration_v2(args.config)
            authorized = getattr(args, "authorized_execution", None)
            source = getattr(args, "source_trajectory", None)
            if isinstance(authorized, AuthorizedTrajectoryExecution) and source is not None:
                from memcontam.readiness.phase13_prefix_reuse import derive_prefix_windows

                print(json.dumps(derive_prefix_windows(authorized.request, source), default=_json_value, sort_keys=True))
                return
            raise SystemExit(render_terminal(CalibrationV2ExternalBlock()))
        if args.phase13_command == "validate-calibration-v2-archive":
            from memcontam.manifests.phase13_archive_validation import validate_phase13_archive

            archive = getattr(args, "archive", None)
            if isinstance(archive, Path):
                print(json.dumps(validate_phase13_archive(archive).to_dict(), sort_keys=True))
                return
            validate_calibration_v2(args.config)
            raise SystemExit(render_terminal(CalibrationV2ExternalBlock()))
        if args.phase13_command == "run-calibration-v2":
            authorized = getattr(args, "authorized_execution", None)
            if isinstance(authorized, AuthorizedTrajectoryExecution):
                validate_calibration_v2(args.config)
                result = execute_calibration_trajectory(authorized.request)
                print(json.dumps(result, default=_json_value, sort_keys=True))
                return
            report_path = getattr(args, "report", None)
            if isinstance(report_path, Path):
                from memcontam.readiness.phase13_calibration_v2_lifecycle import (
                    LifecycleInvocation,
                    run_calibration_v2_lifecycle,
                )

                report = run_calibration_v2_lifecycle(
                    LifecycleInvocation(
                        args.config,
                        report_path,
                        getattr(args, "request", None),
                        getattr(args, "authorization", None),
                        getattr(args, "expected_authorization_sha256", None),
                        getattr(args, "allow_live_calls", False),
                        os.environ,
                    ),
                    lambda: (_ for _ in ()).throw(AssertionError("real execution forbidden")),
                )
                print(json.dumps(asdict(report), sort_keys=True))
                print(report.terminal)
                print(report.main_terminal)
                return
            request = getattr(args, "request", None)
            authorization_path = getattr(args, "authorization", None)
            expected_digest = getattr(args, "expected_authorization_sha256", None)
            client = getattr(args, "provider_client", None)
            identity = getattr(args, "execution_template", None)
            if not (
                isinstance(request, Path)
                and isinstance(authorization_path, Path)
                and isinstance(expected_digest, str)
                and client is not None
                and isinstance(identity, ExecutionTemplateIdentity)
            ):
                raise SystemExit(
                    f"{render_terminal(CalibrationV2ExternalBlock())}\nMAIN_A_EXECUTION_FORBIDDEN"
                )
            from memcontam.readiness.phase13_calibration_v2_authorization import (
                CalibrationV2AuthorizationError,
                verify_calibration_v2_authorization,
            )

            try:
                verify_calibration_v2_authorization(
                    config_path=args.config,
                    request_path=request,
                    authorization_path=authorization_path,
                    expected_authorization_sha256=expected_digest,
                    allow_live_calls=getattr(args, "allow_live_calls", False),
                    environment=getattr(args, "authorization_environment", None),
                    now=getattr(args, "authorization_now", None),
                )
            except CalibrationV2AuthorizationError as error:
                raise SystemExit(error.code) from error
            build_calibration_v2_provider(client, Path(__file__).resolve().parents[3], identity)
            print("CALIBRATION_V2_AUTHORIZED")
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
    except Phase13CalibrationError as error:
        raise SystemExit(error.code) from error
    except CalibrationV2ConfigError as error:
        raise SystemExit(error.code) from error


def _json_value(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    raise TypeError(type(value).__name__)
