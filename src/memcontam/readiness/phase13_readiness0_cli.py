from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path


def add_readiness0_live_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = commands.add_parser("run-readiness0-live")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--expected-authorization-sha256", required=True)
    parser.add_argument("--f1c-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--core-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--allow-live-calls", action="store_true")


def run_readiness0_live_command(args: argparse.Namespace) -> None:
    import memcontam.readiness.phase13_readiness0_live as live

    pilot = live.run_readiness0_live(
        request_path=args.request,
        authorization_path=args.authorization,
        expected_authorization_sha256=args.expected_authorization_sha256,
        f1c_registry_path=args.f1c_registry,
        repository_root=args.repository_root,
        core_root=args.core_root,
        cache_root=args.cache_root,
        output_dir=args.output,
        allow_live_calls=args.allow_live_calls,
    )
    print(json.dumps(asdict(pilot), sort_keys=True))


__all__ = ["add_readiness0_live_parser", "run_readiness0_live_command"]
