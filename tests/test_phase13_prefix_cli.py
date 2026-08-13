from __future__ import annotations

import argparse

from memcontam.readiness.phase13_cli import add_parser


def test_phase13_cli_exposes_derived_window_validation_caller() -> None:
    parser = argparse.ArgumentParser()
    add_parser(parser.add_subparsers(dest="command", required=True))

    parsed = parser.parse_args(
        ["phase13", "validate-derived-windows", "--config", "config.yaml"]
    )

    assert parsed.phase13_command == "validate-derived-windows"
