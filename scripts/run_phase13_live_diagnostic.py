from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from pydantic import ValidationError

from memcontam.readiness.phase13_authority_files import read_regular_nofollow
from memcontam.readiness.phase13_diagnostic_cli import (
    load_registration,
    preflight,
    register,
    run_registered,
    verify_registered,
)
from memcontam.readiness.phase13_diagnostic_intake import (
    IntakePaths,
    PendingPrerequisites,
    candidate_from_intake,
)
from memcontam.readiness.phase13_live_diagnostic_models import LiveDiagnosticError
from memcontam.readiness.phase13_main_live_dispatch import persist_evidence_bytes


def main() -> int:
    parser = argparse.ArgumentParser(description="Registered Diagnostic native execution; no Main authority.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "register", "run", "verify", "report"):
        command = commands.add_parser(name)
        command.add_argument("--registration", type=Path, required=name != "register")
        command.add_argument("--output", type=Path)
        if name == "register":
            command.add_argument("--output-dir", type=Path, required=True)
            for option in ("scope", "eligibility", "budget-binding", "execution-approval", "preparation"):
                command.add_argument("--" + option, type=Path)
        if name in {"run", "verify", "report"}:
            command.add_argument("--run-root", type=Path, required=True)
        if name == "run":
            command.add_argument("--child-execution-id", required=True)
            command.add_argument("--registration-sha256", required=True)
    args = parser.parse_args()
    try:
        if args.command == "preflight":
            result = preflight(args.registration)
        elif args.command == "register":
            if args.registration is not None:
                if any(value is not None for value in (args.scope, args.eligibility, args.budget_binding, args.execution_approval, args.preparation)):
                    raise LiveDiagnosticError("DIAGNOSTIC_AMBIGUOUS_INTAKE")
                candidate = args.registration
            else:
                if any(value is None for value in (args.scope, args.eligibility, args.budget_binding, args.execution_approval)):
                    raise LiveDiagnosticError("DIAGNOSTIC_INTAKE_REQUIRED")
                candidate = candidate_from_intake(IntakePaths(args.scope, args.eligibility,
                    args.budget_binding, args.execution_approval, args.preparation))
            result = {"registration_sha256": register(candidate, args.output_dir / "registration.json")}
        elif args.command == "run":
            candidate = load_registration(args.registration)
            child = next((row for row in candidate.subject.children if row.child_execution_id == args.child_execution_id), None)
            if (hashlib.sha256(read_regular_nofollow(args.registration)).hexdigest() != args.registration_sha256
                or child is None or args.run_root != Path(child.run_root)):
                raise LiveDiagnosticError("DIAGNOSTIC_EXACT_LAUNCH_MISMATCH")
            result = run_registered(args.registration, args.child_execution_id, os.environ.get("OPENAI_API_KEY", ""))
        else:
            result = {"integrity_only_not_launch_authority": True,
                      "evidence_purpose": load_registration(args.registration).admission.science.purpose,
                      **verify_registered(args.registration, args.run_root)}
        raw = (json.dumps(result, sort_keys=True) + "\n").encode()
        if args.output is not None:
            persist_evidence_bytes(args.output, raw, "DIAGNOSTIC_REPORT_ALREADY_EXISTS")
        print(raw.decode(), end="")
        return 0
    except PendingPrerequisites as error:
        print(error.result.model_dump_json(exclude={"subject"}))
        return 2
    except LiveDiagnosticError as error:
        print(json.dumps({"status": "REFUSED", "error_type": type(error).__name__, "code": error.code}))
        return 2
    except ValidationError as error:
        print(json.dumps({"status": "REFUSED", "error_type": "ValidationError",
            "code": "DIAGNOSTIC_INPUT_SCHEMA_INVALID", "fields": [
                {"path": ".".join(str(part) for part in row["loc"]), "type": row["type"]}
                for row in error.errors(include_input=False, include_context=False, include_url=False)]}))
        return 2
    except (ValueError, OSError) as error:
        print(json.dumps({"status": "REFUSED", "error_type": type(error).__name__}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
