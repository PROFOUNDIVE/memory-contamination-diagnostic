from __future__ import annotations

import argparse
import json
from pathlib import Path

from memcontam.readiness.phase13_main_execution import (
    Phase13MainExecutionError,
    validate_main_authorization,
    validate_main_execution_freeze,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="phase13-main-execution")
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("validate-freeze")
    freeze.add_argument("--repository-root", type=Path, required=True)
    freeze.add_argument("--package", type=Path, required=True)
    authorization = commands.add_parser("validate-authorization")
    authorization.add_argument("--repository-root", type=Path, required=True)
    authorization.add_argument("--package", type=Path, required=True)
    authorization.add_argument("--authorization", type=Path, required=True)
    authorization.add_argument("--expected-authorization-sha256", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        if args.command == "validate-freeze":
            report = validate_main_execution_freeze(args.repository_root, args.package)
        else:
            report = validate_main_authorization(
                args.repository_root,
                args.package,
                args.authorization,
                args.expected_authorization_sha256,
            )
        print(json.dumps(report.model_dump(mode="json"), sort_keys=True))
    except Phase13MainExecutionError as error:
        raise SystemExit(error.code) from error


if __name__ == "__main__":
    main()
