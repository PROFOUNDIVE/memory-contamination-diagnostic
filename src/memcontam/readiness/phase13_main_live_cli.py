from __future__ import annotations

import argparse
import json
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
    MainLiveDispatchError,
    summarize_telemetry,
)
from memcontam.readiness.phase13_main_production import build_production_objects


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
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        if args.command == "telemetry":
            print(summarize_telemetry(args.evidence_root).model_dump_json())
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
    except (MainLiveContractError, MainLiveDispatchError, Phase13MainExecutionError) as error:
        raise SystemExit(error.code) from error
    except (OSError, ValidationError, ValueError) as error:
        raise SystemExit("MAIN_LIVE_PREFLIGHT_INVALID") from error


if __name__ == "__main__":
    main()
