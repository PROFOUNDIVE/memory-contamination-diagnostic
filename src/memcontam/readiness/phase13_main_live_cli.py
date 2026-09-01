from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from pydantic import ValidationError

from memcontam.readiness.phase13_authority_files import read_regular_nofollow
from memcontam.readiness.phase13_main_execution import (
    Phase13MainExecutionError,
    validate_main_authorization,
)
from memcontam.readiness.phase13_main_execution_models import MainExecutionFreeze
from memcontam.readiness.phase13_main_live_contract import (
    MainLiveContractError,
    load_main_live_contract,
    validate_main_live_contract,
)
from memcontam.readiness.phase13_main_live_dispatch import (
    DurableMainDispatch,
    MainLiveDispatchError,
    summarize_telemetry,
)
from memcontam.readiness.phase13_main_production import build_production_objects
from memcontam.readiness.phase13_main_production_backend import (
    MainProductionBackend,
    MainProductionBackendError,
)
from memcontam.readiness.phase13_main_live_runtime import MainLiveRuntimeError, ProductionMainRuntime
from memcontam.readiness.phase13_main_runner import (
    MainRunError,
    MainRunRequest,
    open_main_run,
    prepare_main_run,
    run_pending,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="phase13-main-a-live")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--repository-root", type=Path, required=True)
    validate.add_argument("--package", type=Path, required=True)
    validate.add_argument("--authorization", type=Path, required=True)
    validate.add_argument("--expected-authorization-sha256", required=True)
    telemetry = commands.add_parser("telemetry")
    telemetry.add_argument("--evidence-root", type=Path, required=True)
    for name in ("run", "resume"):
        execute = commands.add_parser(name)
        execute.add_argument("--repository-root", type=Path, required=True)
        execute.add_argument("--package", type=Path, required=True)
        execute.add_argument("--authorization", type=Path, required=True)
        execute.add_argument("--expected-authorization-sha256", required=True)
        execute.add_argument("--run-root", type=Path, required=True)
        execute.add_argument("--run-id", required=True)
        execute.add_argument("--tranche-ceiling-krw", type=int, required=True)
        execute.add_argument("--max-units", type=int)
        execute.add_argument("--cache-root", type=Path, default=Path(".cache/phase13-main"))
        execute.add_argument("--allow-live-calls", action="store_true", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        if args.command == "telemetry":
            print(summarize_telemetry(args.evidence_root).model_dump_json())
            return
        if args.command in {"run", "resume"}:
            if args.max_units != 0:
                ProductionMainRuntime.preflight()
            runtime = ProductionMainRuntime(args.repository_root, args.cache_root)
            request = MainRunRequest(
                args.repository_root,
                args.package,
                args.authorization,
                args.expected_authorization_sha256,
                args.run_root,
                args.run_id,
            )
            evidence_root = args.run_root / args.run_id
            ledger = prepare_main_run(request) if args.command == "run" else open_main_run(request)
            backend = MainProductionBackend(
                evidence_root,
                runtime.execute_prefix,
                runtime.execute_ordinary,
                ledger.completed_evidence_sha256,
            )
            dispatch = DurableMainDispatch(evidence_root, backend)
            report = run_pending(
                ledger,
                dispatch,
                tranche_ceiling_krw=args.tranche_ceiling_krw,
                max_units=args.max_units,
            )
            telemetry = summarize_telemetry(evidence_root)
            if telemetry.unit_count != report.completed_count:
                raise MainLiveDispatchError("MAIN_LIVE_TELEMETRY_LEDGER_MISMATCH")
            print(
                json.dumps(
                    {**asdict(report), "provider_calls_issued": telemetry.provider_call_count},
                    sort_keys=True,
                )
            )
            return
        authorization = validate_main_authorization(
            args.repository_root,
            args.package,
            args.authorization,
            args.expected_authorization_sha256,
        )
        package = MainExecutionFreeze.model_validate_json(read_regular_nofollow(args.package))
        contract = load_main_live_contract(
            args.repository_root / "data/phase13/main/main_live_contract_v1.json"
        )
        validate_main_live_contract(contract, package)
        units = build_production_objects(package)
        prefix_count = sum(unit.kind == "CLEAN_PREFIX" for unit in units)
        print(
            json.dumps(
                {
                    "authorization_id": authorization.authorization_id,
                    "main_a_status": authorization.main_a_status,
                    "prefix_count": prefix_count,
                    "provider_calls_issued": 0,
                    "status": "READY_NO_CALLS",
                    "unit_count": len(units),
                },
                sort_keys=True,
            )
        )
    except (
        MainLiveContractError,
        MainLiveDispatchError,
        MainLiveRuntimeError,
        MainProductionBackendError,
        MainRunError,
        Phase13MainExecutionError,
    ) as error:
        raise SystemExit(error.code) from error
    except (OSError, ValidationError, ValueError) as error:
        raise SystemExit("MAIN_LIVE_PREFLIGHT_INVALID") from error


if __name__ == "__main__":
    main()
