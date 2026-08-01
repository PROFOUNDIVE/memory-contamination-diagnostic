from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Mapping


def add_calibration_parsers(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    validate = commands.add_parser("validate-calibration-config")
    validate.add_argument("--config", type=Path, required=True)
    for name, freeze in (("screening-cost-preview", "--freeze-a"), ("bct-cost-preview", "--freeze-b")):
        preview = commands.add_parser(name)
        preview.add_argument("--config", type=Path, required=True)
        preview.add_argument(freeze, type=Path, required=True)
        preview.add_argument("--ledger", type=Path, required=True)
        preview.add_argument("--output", type=Path, required=True)
    for name, authorization in (("screen-controls", "screening"), ("bct-run", "bct")):
        run = commands.add_parser(name)
        run.add_argument("--config", type=Path, required=True)
        run.add_argument("--freeze-a" if authorization == "screening" else "--freeze-b", type=Path, required=True)
        run.add_argument("--authorization-request", type=Path, required=True)
        run.add_argument("--authorization", type=Path)
        run.add_argument("--expected-authorization-sha256")
        run.add_argument("--artifact-root", type=Path, required=True)
        run.add_argument("--run-id", required=True)
        run.add_argument("--stage-result", type=Path, required=True)
    archive = commands.add_parser("validate-bct-archive")
    archive.add_argument("--config", type=Path, required=True)
    archive.add_argument("--freeze-b", type=Path, required=True)
    archive.add_argument("--archive", type=Path, required=True)
    archive.add_argument("--output", type=Path, required=True)
    readiness = commands.add_parser("pilot-b-readiness")
    readiness.add_argument("--bundle", type=Path, required=True)
    readiness.add_argument("--code-prespec", type=Path, required=True)
    readiness.add_argument("--fixture", type=Path)
    readiness.add_argument("--stage-result", type=Path, required=True)


def run_calibration_command(args: argparse.Namespace, handlers: Mapping[str, Callable[[], None]]) -> None:
    try:
        handlers[args.filter_v5_command]()
    except KeyError as error:
        raise ValueError("CALIBRATION_COMMAND_UNKNOWN") from error
