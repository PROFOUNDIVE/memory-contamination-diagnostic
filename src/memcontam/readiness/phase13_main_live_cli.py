from __future__ import annotations

import argparse

from .phase13_main_command import build_parser, execute_command


def _parser() -> argparse.ArgumentParser:
    return build_parser("phase13-main-a-live", live=True)


def main() -> None:
    args = _parser().parse_args()
    try:
        execute_command(args, live=True)
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(getattr(error, "code", "MAIN_LIVE_PREFLIGHT_INVALID")) from error


if __name__ == "__main__":
    main()
