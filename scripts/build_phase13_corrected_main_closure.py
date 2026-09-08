from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Literal, assert_never

from pydantic import TypeAdapter

from memcontam.readiness.phase13_v3_builder import (
    audit, build_mr_p4, build_mr_p5, build_mr_p6,
    validate_mr_p4, validate_mr_p5, validate_mr_p6,
)
from memcontam.readiness.phase13_v3_publication import ArtifactError, P4_PATHS, P5_PATHS, P6_PATHS


def _root(path: Path, suffix: str) -> Path:
    if ".." in path.parts or "\\" in str(path):
        raise ArtifactError("MAIN_PATH_UNSAFE")
    expected = Path(suffix).parts
    if path.parts[-len(expected):] != expected:
        raise ArtifactError("MAIN_HISTORICAL_OUTPUT_FORBIDDEN")
    return path.absolute().parents[len(expected) - 1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic V3 closure; authorization ends at STOP.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("mr-p4", "mr-p5", "mr-p6", "validate", "audit"):
        command = commands.add_parser(name)
        command.add_argument("--repository-root", type=Path, required=True)
        command.add_argument("--authority-root", type=Path, required=True)
        if name == "mr-p4":
            command.add_argument("--governed-source-commit", required=True)
            command.add_argument("--output-root", type=Path, required=True)
        elif name == "mr-p5":
            command.add_argument("--mr-p4", type=Path, required=True)
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--output-root", type=Path)
        elif name == "mr-p6":
            command.add_argument("--mr-p5", type=Path, required=True)
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--sha256-output", type=Path, required=True)
        elif name == "validate":
            command.add_argument("--stage", choices=("mr-p4", "mr-p5", "mr-p6"), required=True)
            command.add_argument("--artifact", type=Path, required=True)
            command.add_argument("--sha256-file", type=Path)
            command.add_argument("--print-governed-source-commit", action="store_true")
        else:
            command.add_argument("--mr-p4", type=Path)
            command.add_argument("--mr-p5", type=Path)
            command.add_argument("--mr-p6", type=Path)
            command.add_argument("--compare-output-root", type=Path)
            for receipt in ("initial-head", "historical-baseline", "initial-status"):
                command.add_argument("--" + receipt, type=Path)
    args = parser.parse_args()
    command_name = TypeAdapter(Literal["mr-p4", "mr-p5", "mr-p6", "validate", "audit"]).validate_python(args.command)
    repository, authority = args.repository_root.absolute(), args.authority_root.absolute()
    try:
        match command_name:
            case "mr-p4":
                result = build_mr_p4(repository, authority, args.output_root, governed_source_commit=args.governed_source_commit)
            case "mr-p5":
                output = _root(args.mr_p4, P4_PATHS[-1])
                if _root(args.output, P5_PATHS[-1]) != output or (args.output_root is not None and args.output_root.absolute() != output):
                    raise ArtifactError("MAIN_ARTIFACT_BINDING_MISMATCH")
                result = build_mr_p5(repository, authority, output)
            case "mr-p6":
                output = _root(args.mr_p5, P5_PATHS[-1])
                if _root(args.output, P6_PATHS[0]) != output or _root(args.sha256_output, P6_PATHS[1]) != output:
                    raise ArtifactError("MAIN_ARTIFACT_BINDING_MISMATCH")
                result = build_mr_p6(repository, authority, output)
            case "validate":
                stage = TypeAdapter(Literal["mr-p4", "mr-p5", "mr-p6"]).validate_python(args.stage)
                match stage:
                    case "mr-p4":
                        result = validate_mr_p4(repository, authority, _root(args.artifact, P4_PATHS[-1]))
                        if args.print_governed_source_commit:
                            print(result.governed_source.governed_source_commit)
                            return 0
                    case "mr-p5":
                        result = validate_mr_p5(repository, authority, _root(args.artifact, P5_PATHS[-1]))
                    case "mr-p6":
                        output = _root(args.artifact, P6_PATHS[0])
                        if args.sha256_file is None or _root(args.sha256_file, P6_PATHS[1]) != output:
                            raise ArtifactError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
                        result = validate_mr_p6(repository, authority, output)
                    case unreachable_stage:
                        assert_never(unreachable_stage)
                if args.print_governed_source_commit:
                    raise ArtifactError("MAIN_ARTIFACT_BINDING_MISMATCH")
            case "audit":
                output = repository / "data/phase13/main"
                for value, path in ((args.mr_p4, P4_PATHS[-1]), (args.mr_p5, P5_PATHS[-1]), (args.mr_p6, P6_PATHS[0])):
                    if value is not None and _root(value, path) != output:
                        raise ArtifactError("MAIN_ARTIFACT_BINDING_MISMATCH")
                for value, name in ((args.initial_head, "phase13-initial-head.txt"), (args.initial_status, "phase13-initial-status.bin"), (args.historical_baseline, "phase13-historical-baseline.json")):
                    if value is not None and value.absolute() != repository / ".omo/evidence" / name:
                        raise ArtifactError("MAIN_ARTIFACT_BINDING_MISMATCH")
                audit(repository, authority, output, compare_output_root=args.compare_output_root)
                print("APPROVE CLOSED -> FROZEN -> AUTHORIZED_EXECUTION -> STOP")
                return 0
            case unreachable_command:
                assert_never(unreachable_command)
        print(json.dumps({"status": result.status, "provider_calls": 0, "measured_trajectories": 0}))
        return 0
    except (ValueError, OSError) as error:
        print(getattr(error, "code", "MAIN_ARTIFACT_BINDING_MISMATCH"), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
