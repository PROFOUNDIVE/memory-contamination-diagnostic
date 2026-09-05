from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .phase13_main_execution_models import MainAuthorizationReport
from .phase13_main_v3_runner import V3MainRun
from .phase13_v3_entrypoint import SelectedExecutionV3, SelectionRequest, select_execution


def build_parser(prog: str, *, live: bool) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in (("validate", "run", "resume") if live else ("validate", "run", "resume", "status")):
        command = commands.add_parser(name, allow_abbrev=False)
        command.add_argument("--repository-root", type=Path, required=True)
        command.add_argument("--package", type=Path, required=True)
        command.add_argument("--authorization", type=Path, required=True)
        command.add_argument("--authority-root", type=Path)
        digest = command.add_mutually_exclusive_group(required=True)
        digest.add_argument("--expected-authorization-sha256")
        digest.add_argument("--expected-authorization-sha256-file", type=Path)
        command.add_argument("--cache-root", type=Path, default=Path(".omo/evidence/phase13-cache"))
        if name != "validate":
            command.add_argument("--run-root", type=Path, required=True)
            command.add_argument("--run-id", required=True)
            command.add_argument("--max-units", type=int)
            command.add_argument("--tranche-ceiling-krw", type=int, default=450000,
                                 required=live and name in {"run", "resume"})
            command.add_argument("--allow-live-calls", action="store_true", required=live and name in {"run", "resume"})
    if live:
        telemetry = commands.add_parser("telemetry", allow_abbrev=False)
        telemetry.add_argument("--evidence-root", type=Path, required=True)
    return parser


def execute_command(args: argparse.Namespace, *, live: bool) -> None:
    if args.command == "telemetry":
        from .phase13_main_live_dispatch import summarize_telemetry
        print(summarize_telemetry(args.evidence_root).model_dump_json())
        return
    if (getattr(args, "max_units", None) is not None and args.max_units < 0
        or not 0 <= getattr(args, "tranche_ceiling_krw", 450000) <= 450000):
        from .phase13_v3_entrypoint import EntrypointError
        raise EntrypointError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
    request = SelectionRequest(args.repository_root, args.package, args.authorization,
        args.authority_root, args.expected_authorization_sha256_file, getattr(args, "run_id", None),
        args.expected_authorization_sha256)
    selected = select_execution(request, args.command)
    match selected:
        case MainAuthorizationReport():
            print(json.dumps({**selected.model_dump(mode="json"), "status": "VALIDATED_HISTORICAL_ONLY", "provider_calls_issued": 0}))
        case SelectedExecutionV3():
            try:
                selected.preflight(request.repository_root)
                if args.command == "validate":
                    print(json.dumps({"status": "READY_NO_CALLS", "provider_calls_issued": 0,
                        "authorization_id": selected.authorization.authorization_id, "main_a_status": "NOT_STARTED",
                        "unit_count": len(selected.package.production),
                        "prefix_count": sum(unit.kind == "CLEAN_PREFIX" for unit in selected.package.production)}))
                    return
                run = V3MainRun.open(selected, args.run_root / args.run_id, create=args.command == "run")
                try:
                    if args.command == "resume":
                        run.dispatcher().recover()
                    report = (run.execute(args.cache_root, max_units=args.max_units,
                        tranche_ceiling_krw=args.tranche_ceiling_krw) if live and args.command in {"run", "resume"}
                        else run.status())
                    print(json.dumps(asdict(report), sort_keys=True))
                finally:
                    run.close()
            finally:
                selected.close()
