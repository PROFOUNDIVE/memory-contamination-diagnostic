from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from memcontam.readiness.phase13_main_runner import (
    MainRunError,
    MainRunRequest,
    open_main_run,
    prepare_main_run,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="phase13-main-a")
    commands = parser.add_subparsers(dest="command", required=True)
    for command_name in ("run", "status", "resume"):
        command = commands.add_parser(command_name)
        command.add_argument("--repository-root", type=Path, required=True)
        command.add_argument("--package", type=Path, required=True)
        command.add_argument("--authorization", type=Path, required=True)
        command.add_argument("--expected-authorization-sha256", required=True)
        command.add_argument("--run-root", type=Path, required=True)
        command.add_argument("--run-id", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    request = MainRunRequest(
        args.repository_root,
        args.package,
        args.authorization,
        args.expected_authorization_sha256,
        args.run_root,
        args.run_id,
    )
    try:
        ledger = prepare_main_run(request) if args.command == "run" else open_main_run(request)
        status = ledger.status()
        if args.command == "resume" and status.in_flight_count:
            raise MainRunError("MAIN_RUN_IN_FLIGHT_RECONCILIATION_REQUIRED")
        print(json.dumps(asdict(status), sort_keys=True))
    except MainRunError as error:
        raise SystemExit(error.code) from error


if __name__ == "__main__":
    main()
