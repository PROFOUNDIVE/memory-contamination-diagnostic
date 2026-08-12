from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

from memcontam.main_registry import Task
from memcontam.readiness.phase13_authority_files import AuthorityFileError, read_regular_nofollow

TASKS: Final[tuple[Task, ...]] = ("game24", "math_equation_balancer", "word_sorting")
SOURCE_NAMES: Final[dict[Task, str]] = {
    "game24": "game24_main_v1.jsonl",
    "math_equation_balancer": "math_equation_balancer_main_v1.jsonl",
    "word_sorting": "word_sorting_main_v1.jsonl",
}
FORBIDDEN_TOKENS: Final = frozenset({"outcome", "outcomes", "eligibility", "eligible"})


class AuthorityError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(read_regular_nofollow(path))
    except AuthorityFileError as error:
        raise AuthorityError(str(error)) from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AuthorityError("REGISTRY_INPUT_MALFORMED") from error
    if not isinstance(value, dict):
        raise AuthorityError("REGISTRY_INPUT_MALFORMED")
    return value


def load_jsonl(path: Path) -> list[dict[str, object]]:
    try:
        values = [json.loads(line) for line in read_regular_nofollow(path).splitlines()]
    except AuthorityFileError as error:
        raise AuthorityError(str(error)) from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AuthorityError("REGISTRY_INPUT_MALFORMED") from error
    if any(not isinstance(value, dict) for value in values):
        raise AuthorityError("REGISTRY_INPUT_MALFORMED")
    return values


def pilot_signatures(root: Path, task: Task) -> frozenset[str]:
    rows = load_jsonl(root / f"data/tasks/{task}_pilot.jsonl")
    signatures: set[str] = set()
    for row in rows:
        match task:
            case "game24":
                values = row.get("numbers")
                if not isinstance(values, list) or not all(isinstance(item, int) for item in values):
                    raise AuthorityError("PILOT_REGISTRY_INVALID")
                signatures.add(",".join(str(item) for item in sorted(values)))
            case "math_equation_balancer":
                verifier = row.get("verifier_spec")
                if not isinstance(verifier, dict) or not isinstance(verifier.get("target_value"), int):
                    raise AuthorityError("PILOT_REGISTRY_INVALID")
                signatures.add(",".join((*re.findall(r"-?\d+", str(row.get("input"))), str(verifier["target_value"]))))
            case "word_sorting":
                values = row.get("words")
                if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
                    raise AuthorityError("PILOT_REGISTRY_INVALID")
                signatures.add("|".join(sorted(values)))
    return frozenset(signatures)


def candidate_signatures(root: Path) -> dict[str, str]:
    triplets = load_json(root / "data/phase12/registries/candidate_registry_v1.json").get("triplets")
    if not isinstance(triplets, list) or len(triplets) != len(TASKS):
        raise AuthorityError("CANDIDATE_REGISTRY_INVALID")
    signatures: dict[str, str] = {}
    for triplet in triplets:
        if not isinstance(triplet, dict):
            raise AuthorityError("CANDIDATE_REGISTRY_INVALID")
        task, example = triplet.get("task"), triplet.get("counterexample")
        match task, example:
            case "game24", str(value):
                numbers = sorted(int(item) for item in re.findall(r"\d+", value.rsplit("=", 1)[0]))
                signatures[task] = ",".join(str(item) for item in numbers)
            case "math_equation_balancer", str(value):
                signatures[task] = ",".join(re.findall(r"-?\d+", value))
            case "word_sorting", list(values) if all(isinstance(item, str) for item in values):
                signatures[task] = "|".join(sorted(values))
            case _:
                raise AuthorityError("CANDIDATE_REGISTRY_INVALID")
    return signatures


def reject_forbidden_fields(value: object) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            snake = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key).lower()
            tokens = frozenset(part for part in re.split(r"[^a-z0-9]+", snake) if part)
            forbidden_rate = "rate" in tokens and "limit" not in tokens
            forbidden_result = "verifier" in tokens and "result" in tokens
            if tokens & FORBIDDEN_TOKENS or forbidden_rate or forbidden_result:
                raise AuthorityError("FORBIDDEN_SEMANTIC_FIELD")
            reject_forbidden_fields(nested)
    elif isinstance(value, list):
        for nested in value:
            reject_forbidden_fields(nested)


def validate_authority_hashes(root: Path, claimed: Mapping[str, object]) -> None:
    expected = {
        "main_manifest_sha256": root / "data/phase13/main/main_registry_manifest_v1.json",
        "exclusions_sha256": root / "data/phase13/main/exclusions_v1.json",
        "candidate_registry_sha256": root / "data/phase12/registries/candidate_registry_v1.json",
    }
    for key, path in expected.items():
        try:
            digest = hashlib.sha256(read_regular_nofollow(path)).hexdigest()
        except AuthorityFileError as error:
            raise AuthorityError(str(error)) from error
        if claimed.get(key) != digest:
            raise AuthorityError("INPUT_AUTHORITY_HASH_MISMATCH")
    pilots = claimed.get("pilot_registry_sha256")
    if not isinstance(pilots, dict):
        raise AuthorityError("INPUT_AUTHORITY_HASH_MISMATCH")
    for task in TASKS:
        path = root / f"data/tasks/{task}_pilot.jsonl"
        try:
            digest = hashlib.sha256(read_regular_nofollow(path)).hexdigest()
        except AuthorityFileError as error:
            raise AuthorityError(str(error)) from error
        if pilots.get(task) != digest:
            raise AuthorityError("INPUT_AUTHORITY_HASH_MISMATCH")


def authenticated_source(root: Path, task: Task) -> tuple[list[dict[str, object]], dict[str, str]]:
    manifest = load_json(root / "data/phase13/main/main_registry_manifest_v1.json")
    registries = manifest.get("registries")
    if not isinstance(registries, dict) or not isinstance(registries.get(task), dict):
        raise AuthorityError("REGISTRY_INPUT_MALFORMED")
    entry = registries[task]
    name, digest = entry.get("path"), entry.get("sha256")
    if not isinstance(name, str) or not isinstance(digest, str):
        raise AuthorityError("REGISTRY_INPUT_MALFORMED")
    if name != SOURCE_NAMES[task] or Path(name).name != name:
        raise AuthorityError("MAIN_SOURCE_PATH_INVALID")
    relative = f"data/phase13/main/{name}"
    path = root / relative
    main_root = (root / "data/phase13/main").resolve()
    if path.parent.resolve() != main_root:
        raise AuthorityError("MAIN_SOURCE_PATH_INVALID")
    try:
        source_raw = read_regular_nofollow(path)
    except AuthorityFileError as error:
        raise AuthorityError(str(error)) from error
    if hashlib.sha256(source_raw).hexdigest() != digest:
        raise AuthorityError("MAIN_SOURCE_HASH_MISMATCH")
    return load_jsonl(path), {"path": relative, "sha256": digest}


def validate_selected_source_rows(
    selected: Sequence[Mapping[str, object]], source_prefix: Sequence[Mapping[str, object]]
) -> None:
    if len(selected) != len(source_prefix):
        raise AuthorityError("CALIBRATION_SOURCE_ROW_MISMATCH")
    for selected_row, source_row in zip(selected, source_prefix, strict=True):
        payload = {key: value for key, value in selected_row.items() if key not in {"sample_id", "row_sha256"}}
        expected = {key: value for key, value in source_row.items() if key != "sample_id"}
        if payload != expected:
            raise AuthorityError("CALIBRATION_SOURCE_ROW_MISMATCH")


def validate_signature_layers(
    root: Path, task: Task, selected: Sequence[str], remainder: Sequence[str], reserved: object
) -> None:
    selected_set = set(selected)
    if selected_set & pilot_signatures(root, task):
        raise AuthorityError("PILOT_SIGNATURE_OVERLAP")
    exclusions = load_json(root / "data/phase13/main/exclusions_v1.json").get("excluded_signatures")
    if not isinstance(exclusions, dict) or not isinstance(exclusions.get(task), list):
        raise AuthorityError("REGISTRY_INPUT_MALFORMED")
    controls = set(exclusions[task]) - set(pilot_signatures(root, task))
    if selected_set & controls:
        raise AuthorityError("CANDIDATE_CONTROL_SIGNATURE_OVERLAP")
    if selected_set & set(remainder):
        raise AuthorityError("PROSPECTIVE_MAIN_V2_SIGNATURE_OVERLAP")
    if not isinstance(reserved, dict):
        raise AuthorityError("RESERVED_EXTENSION_NOT_EMPTY")
    signatures = reserved.get("signatures")
    if isinstance(signatures, list) and signatures:
        raise AuthorityError("RESERVED_EXTENSION_SIGNATURE_OVERLAP")
    if reserved != {"count": 0, "registry_status": "blocked_pending_future_registry", "signatures": []}:
        raise AuthorityError("RESERVED_EXTENSION_NOT_EMPTY")
