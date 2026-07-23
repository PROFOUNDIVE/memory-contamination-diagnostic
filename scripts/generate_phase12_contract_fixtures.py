from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "phase12"
REPOSITORY_COMMIT = "830b89c8c169ffa9cdea472887fdae134dbae7cf"
EXPERIMENT_DESIGN_SHA256 = "984fe2881690d93a8ccced87abf03de4bf0012158462cf07ed23505414073eb0"
MANIFEST_SHA256 = "b129d5a66166b0ec5fc8bb4b32c2ab2a76e09238d074b91687322a97828adda8"
MANIFEST_VERSION = "phase12-fixtures-v13"
FIXTURE_SEED = 12026
FIXTURE_FILENAMES = (
    "FX-AGG-001.json",
    "FX-ARCHIVE-001.json",
    "FX-BEHAVIOR-001.json",
    "FX-BRANCH-001.json",
    "FX-CANDIDATE-001.json",
    "FX-CONFIG-001.json",
    "FX-E2E-001.json",
    "FX-ELIGIBILITY-001.json",
    "FX-FILTER-001.json",
    "FX-OUTCOME-001.json",
    "FX-P12I-001.json",
    "FX-P12I-PASS-001.json",
    "FX-RAG-001.json",
    "FX-ROUTE-001.json",
    "FX-SCHEMA-001.json",
    "FX-SEQUENTIAL-001.json",
    "FX-TOOL-001.json",
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _anchor_errors(fixture_bytes: dict[str, bytes]) -> list[str]:
    errors: list[str] = []
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{REPOSITORY_COMMIT}^{{commit}}"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        env={**os.environ, "GIT_MASTER": "1"},
    )
    if result.returncode:
        errors.append("REPOSITORY_COMMIT_MISMATCH")
    values = tuple(fixture_bytes.values())
    if not any(REPOSITORY_COMMIT.encode("ascii") in value for value in values):
        errors.append("REPOSITORY_COMMIT_MISMATCH")
    if not any(EXPERIMENT_DESIGN_SHA256.encode("ascii") in value for value in values):
        errors.append("EXPERIMENT_DESIGN_HASH_MISMATCH")
    return errors


def _check(fixture_root: Path) -> tuple[dict[str, bytes], list[str]]:
    fixture_bytes: dict[str, bytes] = {}
    errors: list[str] = []
    manifest_path = fixture_root / "manifest.json"
    if not manifest_path.is_file():
        return fixture_bytes, ["MANIFEST_MISSING:manifest.json"]
    manifest_bytes = manifest_path.read_bytes()
    if _sha256(manifest_bytes) != MANIFEST_SHA256:
        errors.append("MANIFEST_HASH_MISMATCH:manifest.json")
    try:
        manifest = json.loads(manifest_bytes)
        registered = manifest["files"]
    except (KeyError, TypeError, json.JSONDecodeError):
        return fixture_bytes, [*errors, "MANIFEST_INVALID:manifest.json"]
    if manifest.get("fixture_manifest_version") != MANIFEST_VERSION:
        errors.append("MANIFEST_VERSION_MISMATCH")
    if manifest.get("seed") != FIXTURE_SEED:
        errors.append("FIXTURE_SEED_MISMATCH")
    if not isinstance(registered, dict) or set(registered) != set(FIXTURE_FILENAMES):
        errors.append("MANIFEST_FILE_SET_MISMATCH")
        registered = {}
    if {path.name for path in fixture_root.glob("FX-*.json")} != set(FIXTURE_FILENAMES):
        errors.append("FIXTURE_FILE_SET_MISMATCH")
    payloads: dict[str, dict[str, object]] = {}
    for filename in FIXTURE_FILENAMES:
        path = fixture_root / filename
        if not path.is_file():
            errors.append(f"FIXTURE_MISSING:{filename}")
            continue
        value = path.read_bytes()
        fixture_bytes[filename] = value
        if b"\r\n" in value:
            errors.append(f"FIXTURE_LINE_ENDING_MISMATCH:{filename}")
        if not value.endswith(b"\n"):
            errors.append(f"FIXTURE_TRAILING_NEWLINE_MISSING:{filename}")
        if _sha256(value) != registered.get(filename):
            errors.append(f"FIXTURE_HASH_MISMATCH:{filename}")
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            errors.append(f"FIXTURE_JSON_INVALID:{filename}")
            continue
        if not isinstance(payload, dict) or payload.get("fixture_id") != filename.removesuffix(
            ".json"
        ):
            errors.append(f"FIXTURE_ID_MISMATCH:{filename}")
            continue
        payloads[filename] = payload
    fixture_ids = {payload["fixture_id"] for payload in payloads.values()}
    for filename, payload in payloads.items():
        compose = payload.get("compose")
        if isinstance(compose, list) and any(item not in fixture_ids for item in compose):
            errors.append(f"FIXTURE_REFERENCE_MISSING:{filename}")
    return fixture_bytes, errors


def _write(fixture_root: Path) -> None:
    if fixture_root.resolve() == DEFAULT_FIXTURE_ROOT.resolve():
        raise ValueError("UNAUTHORIZED_FIXTURE_WRITE")
    for filename in (*FIXTURE_FILENAMES, "manifest.json"):
        (fixture_root / filename).write_bytes((DEFAULT_FIXTURE_ROOT / filename).read_bytes())


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate canonical Phase-12 fixture bytes.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    parser.add_argument("--fixture-root", type=Path, default=DEFAULT_FIXTURE_ROOT)
    parser.add_argument("--plan-revision-only", action="store_true")
    args = parser.parse_args()

    if args.check and args.plan_revision_only:
        parser.error("PLAN_REVISION_ONLY_FLAG_FORBIDDEN")
    if args.write and not args.plan_revision_only:
        parser.error("PLAN_REVISION_ONLY_REQUIRED")
    if not args.fixture_root.is_dir():
        parser.error("FIXTURE_ROOT_MISSING")

    if args.write:
        fixture_bytes, errors = _check(DEFAULT_FIXTURE_ROOT)
        errors.extend(_anchor_errors(fixture_bytes))
        if errors:
            print("\n".join(dict.fromkeys(errors)), file=sys.stderr)
            return 1
        try:
            _write(args.fixture_root)
        except ValueError as error:
            parser.error(str(error))
        return 0

    fixture_bytes, errors = _check(args.fixture_root)
    errors.extend(_anchor_errors(fixture_bytes))
    if errors:
        print("\n".join(dict.fromkeys(errors)), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
