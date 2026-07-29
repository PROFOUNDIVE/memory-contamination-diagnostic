from __future__ import annotations

import ast
import hashlib
import subprocess
import sys
from pathlib import Path

from pydantic import ValidationError

from memcontam.experiment.phase12.filter_challenge.bct import BCTReadiness
from memcontam.experiment.phase12.filter_challenge.build_archive_models import (
    BuildArchiveReport,
)
from memcontam.experiment.phase12.filter_challenge.domain_schema import (
    policy_visible_schema_boundary_valid,
    public_domain_schema_hashes,
)
from memcontam.experiment.phase12.filter_challenge.evidence import (
    validate_evidence_bundle,
)
from memcontam.experiment.phase12.filter_challenge.evidence_contract import (
    EVIDENCE_FILENAMES,
    EvidenceBuildError,
    canonical_json_bytes,
    json_value_from_bytes,
    sha256_path,
)
from memcontam.experiment.phase12.filter_challenge.final_verifier_types import FinalVerifierError
from memcontam.experiment.phase12.filter_challenge.mft import MergedMftReport
from memcontam.experiment.phase12.filter_challenge.mft_state_models import JsonValue


_FORBIDDEN_PROVIDER_TARGETS = {
    "memcontam.clients.factory.build_llm_client",
    "memcontam.clients.openai_compatible.OpenAICompatibleClient",
    "memcontam.clients.openai_responses.OpenAIResponsesClient",
}
_SHORT_PROVIDER_TARGETS = {
    target.rsplit(".", 1)[-1]: target for target in _FORBIDDEN_PROVIDER_TARGETS
}


def verify_code_quality(
    repository_root: Path,
    base_commit: str,
    implementation_commit: str,
    evidence_root: Path,
    validation_summary: Path,
) -> dict[str, JsonValue]:
    changed = _git(repository_root, "diff", "--name-only", base_commit, implementation_commit)
    paths = tuple(path for path in changed.splitlines() if path.endswith(".py"))
    if not paths:
        raise FinalVerifierError("CODE_QUALITY_PYTHON_FILES_REQUIRED")
    findings = [finding for path in paths for finding in _structural_findings(repository_root / path)]
    if findings or not _evidence_serialization_valid(evidence_root, validation_summary):
        raise FinalVerifierError("CODE_QUALITY_REJECTED")
    commands = (
        _command(repository_root, "ruff", (sys.executable, "-m", "ruff", "check", *paths)),
        _command(repository_root, "mypy", (sys.executable, "-m", "mypy", *paths)),
        _command(repository_root, "diff-check", ("git", "diff", "--check", base_commit, implementation_commit)),
    )
    if any(command["exit_code"] != 0 for command in commands):
        raise FinalVerifierError("CODE_QUALITY_REJECTED")
    return {"commands": list(commands), "findings": []}


def _structural_findings(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:
        return [f"syntax:{path.as_posix()}"]
    if "filter_challenge" not in path.as_posix():
        return []
    aliases = _provider_aliases(tree)
    found = any(
        isinstance(node, ast.Call) and _resolved_name(node.func, aliases) in _FORBIDDEN_PROVIDER_TARGETS
        for node in ast.walk(tree)
    )
    return [f"provider:{path.as_posix()}"] if found else []


def _provider_aliases(tree: ast.Module) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in tree.body:
        match node:
            case ast.Import(names=names):
                for item in names:
                    aliases[item.asname or item.name.split(".")[0]] = item.name
            case ast.ImportFrom(module=module, names=names) if module is not None:
                for item in names:
                    aliases[item.asname or item.name] = f"{module}.{item.name}"
            case ast.Assign(targets=targets, value=value):
                resolved = _resolved_name(value, aliases)
                if resolved is not None:
                    for target in targets:
                        if isinstance(target, ast.Name):
                            aliases[target.id] = resolved
            case _:
                continue
    return aliases


def _resolved_name(node: ast.expr, aliases: dict[str, str]) -> str | None:
    match node:
        case ast.Name(id=name):
            return aliases.get(name, _SHORT_PROVIDER_TARGETS.get(name))
        case ast.Attribute(value=value, attr=attribute):
            base = _resolved_name(value, aliases)
            return f"{base}.{attribute}" if base is not None else None
        case ast.Call(func=ast.Name(id="getattr"), args=(base, ast.Constant(value=attribute))):
            resolved = _resolved_name(base, aliases)
            return f"{resolved}.{attribute}" if isinstance(attribute, str) and resolved is not None else None
        case _:
            return None


def _evidence_serialization_valid(evidence_root: Path, validation_summary: Path) -> bool:
    try:
        bundle = validate_evidence_bundle(evidence_root)
        summary = json_value_from_bytes(validation_summary.read_bytes(), "VALIDATION_SUMMARY_INVALID")
        reports = {
            name: json_value_from_bytes((evidence_root / name).read_bytes(), "EVIDENCE_REPORT_INVALID")
            for name in EVIDENCE_FILENAMES
        }
        policy = reports["policy_schema_hashes.json"]
        mft = reports["mft_fv5_report.json"]
        archive = reports["archive_validation_report.json"]
        readiness = reports["bct_readiness_report.json"]
        if not all(isinstance(value, dict) for value in (policy, mft, archive, readiness, summary)):
            return False
        if not isinstance(policy, dict) or not isinstance(mft, dict) or not isinstance(archive, dict):
            return False
        if not isinstance(readiness, dict) or not isinstance(summary, dict):
            return False
        header = bundle.header
        if (
            header.get("implementation_commit") != summary.get("implementation_commit")
            or header.get("validation_summary_sha256") != sha256_path(validation_summary)
            or summary.get("provider_calls_issued") != 0
            or {"evidence_commit", "implementation_manifest_sha256"} & set(header)
            or policy.get("domain_model_schema_hashes") != public_domain_schema_hashes()
            or policy.get("policy_visible_schema_boundary") != "pass"
            or not policy_visible_schema_boundary_valid()
        ):
            return False
        MergedMftReport.model_validate_json(canonical_json_bytes(mft.get("report")))
        BuildArchiveReport.model_validate_json(canonical_json_bytes(archive.get("report")))
        BCTReadiness.model_validate_json(canonical_json_bytes(readiness.get("report")))
        return all(canonical_json_bytes(value) == (evidence_root / name).read_bytes() for name, value in reports.items())
    except (EvidenceBuildError, KeyError, OSError, ValidationError):
        return False


def _command(root: Path, command_id: str, command: tuple[str, ...]) -> dict[str, JsonValue]:
    result = subprocess.run(command, cwd=root, check=False, capture_output=True, text=True)
    return {
        "command_id": command_id,
        "exit_code": result.returncode,
        "stderr_sha256": hashlib.sha256(result.stderr.encode()).hexdigest(),
        "stdout_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
    }


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(("git", "-C", str(root), *arguments), check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise FinalVerifierError("CODE_QUALITY_BASE_INVALID")
    return result.stdout.strip()
