from __future__ import annotations

import argparse

from memcontam.readiness.phase13_cli import add_parser


def test_phase13_parser_registers_legacy_rag_commands() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_parser(subparsers)

    cases = (
        ("audit-legacy-rag", ["--evaluation-root", "main", "--output", "opaque.json"]),
        (
            "materialize-legacy-rag",
            [
                "--repository-root", ".", "--opaque-exclusion", "opaque.json",
                "--cache-root", "cache", "--output", "legacy",
            ],
        ),
        (
            "validate-legacy-rag",
            [
                "--repository-root", ".", "--root", "legacy",
                "--expected-manifest-sha256", "0" * 64,
            ],
        ),
    )
    for command, options in cases:
        args = parser.parse_args(["phase13", command, *options])
        assert args.phase13_command == command
