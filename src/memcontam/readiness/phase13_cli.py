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
from memcontam.readiness.phase13_core_datasets import CoreDatasetError
from memcontam.readiness.phase13_legacy_rag_audit import LegacyRagAuditError
from memcontam.readiness.phase13_legacy_rag_errors import LegacyRagValidationError
from memcontam.readiness.phase13_legacy_rag_materialize import LegacyRagMaterializationError
from .phase13_observability_validate import (
    Phase13ObservabilityValidationError,
)
from .phase13_main_readiness import Phase13MainReadinessError
from memcontam.readiness.phase13_readiness0_live import Readiness0LiveError
from memcontam.readiness.phase13_readiness0_cli import (
    add_readiness0_live_parser,
    run_readiness0_live_command,
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
    materialize = commands.add_parser("materialize-core-datasets")
    materialize.add_argument("--output", type=Path, required=True)
    validate_core = commands.add_parser("validate-core-datasets")
    validate_core.add_argument("--root", type=Path, required=True)
    validate_core.add_argument("--trajectory-seed", type=int, required=True)
    audit_legacy = commands.add_parser("audit-legacy-rag")
    audit_legacy.add_argument("--evaluation-root", type=Path, required=True)
    audit_legacy.add_argument("--output", type=Path, required=True)
    materialize_legacy = commands.add_parser("materialize-legacy-rag")
    materialize_legacy.add_argument("--repository-root", type=Path, required=True)
    materialize_legacy.add_argument("--opaque-exclusion", type=Path, required=True)
    materialize_legacy.add_argument("--cache-root", type=Path, required=True)
    materialize_legacy.add_argument("--output", type=Path, required=True)
    validate_legacy = commands.add_parser("validate-legacy-rag")
    validate_legacy.add_argument("--repository-root", type=Path, required=True)
    validate_legacy.add_argument("--root", type=Path, required=True)
    validate_legacy.add_argument("--expected-manifest-sha256", required=True)
    observability = commands.add_parser("validate-observability")
    observability.add_argument("--repository-root", type=Path, required=True)
    observability.add_argument("--root", type=Path, required=True)
    observability.add_argument("--expected-manifest-sha256", required=True)
    main_readiness = commands.add_parser("validate-main-readiness")
    main_readiness.add_argument("--repository-root", type=Path, required=True)
    main_readiness.add_argument("--root", type=Path, required=True)
    main_readiness.add_argument("--expected-manifest-sha256", required=True)
    add_readiness0_live_parser(commands)


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
            provenance_report = validate_provenance_bundle(args.root, args.manifest, args.seal)
            print(json.dumps(asdict(provenance_report), sort_keys=True))
            return
        if args.phase13_command == "prepare-clean-prefix":
            payload = prepare_clean_prefix(args.config, args.run_id, args.output)
            print(json.dumps(payload, sort_keys=True))
            return
        if args.phase13_command == "materialize-core-datasets":
            from memcontam.readiness.phase13_core_datasets import materialize_core_datasets

            manifest = materialize_core_datasets(args.output)
            print(json.dumps(manifest.model_dump(mode="json"), sort_keys=True))
            return
        if args.phase13_command == "validate-core-datasets":
            from memcontam.readiness.phase13_core_datasets import validate_core_datasets

            core_report = validate_core_datasets(args.root, trajectory_seed=args.trajectory_seed)
            print(json.dumps(core_report.model_dump(mode="json"), sort_keys=True))
            return
        if args.phase13_command == "audit-legacy-rag":
            from memcontam.readiness.phase13_legacy_rag_audit import (
                build_opaque_exclusion_registry,
            )

            audit_report = build_opaque_exclusion_registry(args.evaluation_root, args.output)
            print(json.dumps(audit_report.model_dump(mode="json"), sort_keys=True))
            return
        if args.phase13_command == "validate-legacy-rag":
            from memcontam.readiness.phase13_legacy_rag_validate import (
                validate_legacy_rag_package,
            )

            legacy_report = validate_legacy_rag_package(
                args.root,
                args.repository_root,
                args.expected_manifest_sha256,
            )
            print(json.dumps(legacy_report.model_dump(mode="json"), sort_keys=True))
            return
        if args.phase13_command == "validate-observability":
            from .phase13_observability_validate import (
                validate_phase13_observability_package,
            )

            observability_report = validate_phase13_observability_package(
                args.root,
                args.repository_root,
                args.expected_manifest_sha256,
            )
            print(json.dumps(observability_report.model_dump(mode="json"), sort_keys=True))
            return
        if args.phase13_command == "validate-main-readiness":
            from .phase13_main_readiness import validate_main_readiness

            readiness_report = validate_main_readiness(
                args.root,
                args.repository_root,
                args.expected_manifest_sha256,
            )
            print(json.dumps(readiness_report.model_dump(mode="json"), sort_keys=True))
            return
        if args.phase13_command == "run-readiness0-live":
            run_readiness0_live_command(args)
            return
        if args.phase13_command == "materialize-legacy-rag":
            from memcontam.memory.embeddings import BgeM3EmbeddingProvider
            from memcontam.readiness.phase13_legacy_rag_materialize import (
                LegacyRagMaterializationRequest,
                materialize_legacy_rag_package,
            )

            try:
                legacy_embedder = BgeM3EmbeddingProvider(
                    cache_folder=args.cache_root,
                    local_files_only=True,
                )
            except RuntimeError as error:
                raise LegacyRagMaterializationError(
                    "LEGACY_RAG_BGE_RESOURCE_UNAVAILABLE"
                ) from error
            materialization_report = materialize_legacy_rag_package(
                LegacyRagMaterializationRequest(
                    args.output,
                    args.repository_root,
                    args.opaque_exclusion,
                    legacy_embedder,
                )
            )
            print(json.dumps(materialization_report.model_dump(mode="json"), sort_keys=True))
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
        CoreDatasetError,
        LegacyRagAuditError,
        LegacyRagMaterializationError,
        LegacyRagValidationError,
        Phase13ObservabilityValidationError,
        Phase13MainReadinessError,
        Readiness0LiveError,
    ) as error:
        raise SystemExit(error.code) from error
