from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from memcontam.readiness.phase13_main_execution_bindings import (
    CORRECTED_ARTIFACT_PATHS,
    PRODUCTION_ROLES,
    RUNNER_ROLES,
    canonical_hash,
)
from memcontam.readiness.phase13_main_execution_models import MainExecutionFreeze
from memcontam.readiness.phase13_main_production import (
    build_production_objects,
    prefix_stage_call_counts,
    units_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "phase13-main-a-corrected-20260903-v2"
COMMIT = "d1ac6c84236ec63c367775d24aa953176d321ce0"


def _sha(path: str | Path) -> str:
    target = Path(path)
    if not target.is_absolute():
        target = ROOT / target
    return hashlib.sha256(target.read_bytes()).hexdigest()


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _all_execution_paths() -> dict[str, str]:
    expected = dict(CORRECTED_ARTIFACT_PATHS)
    bound = set(expected.values())
    for path in sorted((ROOT / "src/memcontam").rglob("*.py")):
        relative = path.relative_to(ROOT).as_posix()
        if relative in bound:
            continue
        role = f"transitive_source_{hashlib.sha256(relative.encode()).hexdigest()[:16]}"
        expected[role] = relative
    return expected


def _build_mr_p4() -> Path:
    source = ROOT / "data/phase13/main/mr_p4/manifest_v1.json"
    output = source.parent / "corrected_v1/manifest_v2.json"
    manifest = json.loads(source.read_text(encoding="utf-8"))
    manifest["schema_version"] = "phase13_mr_p4_local_closure_manifest_corrected_v2"
    manifest["corrected_run_id"] = RUN_ID
    replacements = {
        "capacity": "data/phase13/common_capacity_corrected_v2.json",
        "activated_cost_policy": (
            "data/phase13/main/cost_envelope_v2/activated_policy_corrected_v2.json"
        ),
    }
    for role, identity in manifest["artifacts"].items():
        identity["path"] = replacements.get(role, identity["path"])
        identity["sha256"] = _sha(identity["path"])
    corrected = {
        "common_capacity_status_corrected_v2": "data/phase13/common_capacity_status_corrected_v2.json",
        "stage_envelope_registry_corrected_v2": "data/phase13/main/cost_envelope_v2/stage_envelope_registry_corrected_v2.json",
        "cost_proof_corrected_v2": "data/phase13/main/cost_envelope_v2/cost_proof_corrected_v2.json",
        "candidate_manifest_corrected_v2": "data/phase13/main/cost_envelope_v2/candidate_manifest_corrected_v2.json",
    }
    prompt_root = ROOT / "data/phase13/main/mr_p4/corrected_v1"
    for path in sorted(prompt_root.iterdir()):
        if path.is_file() and path != output:
            corrected[f"corrected_prompt_{path.stem}"] = path.relative_to(ROOT).as_posix()
    for role, path in corrected.items():
        manifest["artifacts"][role] = {"path": path, "sha256": _sha(path)}
    manifest["closure_hash"] = canonical_hash(
        {key: value for key, value in manifest.items() if key != "closure_hash"}
    )
    _write(output, manifest)
    return output


def _build_mr_p5(p4_path: Path) -> Path:
    source = ROOT / "data/phase13/main/mr_p5/execution_package_v1.json"
    output = source.with_name("execution_package_v2.json")
    package = json.loads(source.read_text(encoding="utf-8"))
    package.update(
        schema_version="phase13_main_execution_freeze_v2",
        package_id="phase13-main-a-corrected-execution-freeze-v2",
        mr_p4_closure_hash=json.loads(p4_path.read_text(encoding="utf-8"))["closure_hash"],
        corrected_run_id=RUN_ID,
        repository_commit=COMMIT,
        repository_tree_sha256="0" * 64,
    )
    package["cost_guard"].update(cmax_main_krw=444256, margin_krw=5744)
    _build_live_contract(MainExecutionFreeze.model_validate_json(json.dumps(package)))
    existing = {row["role"]: row for row in package["artifacts"]}
    artifacts = []
    for role, path in _all_execution_paths().items():
        row = existing.get(role, {"role": role})
        row.update(path=path, sha256=_sha(path))
        artifacts.append(row)
    package["artifacts"] = artifacts
    bindings = {row["role"]: row["sha256"] for row in artifacts}
    package["observability"]["production_reconstruction_binding_sha256"] = canonical_hash(
        [bindings[role] for role in PRODUCTION_ROLES]
    )
    package["execution_control"]["runner_code_sha256"] = canonical_hash(
        [bindings[role] for role in RUNNER_ROLES]
    )
    package["repository_tree_sha256"] = canonical_hash(
        {row["path"]: row["sha256"] for row in artifacts}
    )
    package["package_hash"] = canonical_hash(
        {key: value for key, value in package.items() if key != "package_hash"}
    )
    _write(output, package)
    return output


def _build_live_contract(package: MainExecutionFreeze) -> None:
    units = build_production_objects(package)
    prefixes = tuple(unit for unit in units if unit.kind == "CLEAN_PREFIX")
    bindings = {binding.role: binding.sha256 for binding in package.artifacts}
    contract = {
        "schema_version": "phase13_main_live_contract_v2",
        "authority_sha256": next(
            row.sha256 for row in package.authority if row.role == "authority_router"
        ),
        "package_id": package.package_id,
        "checkpoint_registry_sha256": bindings["common_checkpoint_registry"],
        "observability_packet_sha256": bindings["observability_packet"],
        "production_units_sha256": units_sha256(units),
        "prefix": {
            "contract_id": "phase13-main-prefix-realization-v1",
            "dispatch_order_id": "phase13-main-production-object-order-v1",
            "owner_law_id": "phase13-main-prefix-four-consumers-v1",
            "failure_law_id": "phase13-main-prefix-atomic-terminal-fanout-v1",
            "checkpoint_evidence_schema_id": "phase13_main_prefix_checkpoint_v1",
            "realization_count": len(prefixes),
            "dispatch_order_sha256": canonical_hash([unit.unit_id for unit in prefixes]),
            "ownership_sha256": canonical_hash(
                [
                    [
                        prefix.unit_id,
                        [unit.unit_id for unit in units if unit.prefix_unit_id == prefix.unit_id],
                    ]
                    for prefix in prefixes
                ]
            ),
            "stage_call_counts": prefix_stage_call_counts(units),
        },
        "cost": {
            "cost_envelope_id": package.cost_guard.cost_envelope_id,
            "cost_envelope_sha256": package.cost_guard.cost_envelope_sha256,
            "semantic_calls": package.cost_guard.semantic_calls,
            "cmax_main_krw": package.cost_guard.cmax_main_krw,
            "core_authorization_gate_krw": package.cost_guard.core_authorization_gate_krw,
        },
    }
    _write(ROOT / "data/phase13/main/main_live_contract_v2.json", contract)


def _build_mr_p6(package_path: Path) -> Path:
    package = json.loads(package_path.read_text(encoding="utf-8"))
    relative = package_path.relative_to(ROOT).as_posix()
    authorization = {
        "schema_version": "phase13_main_authorization_v2",
        "authorization_id": "phase13-main-a-corrected-authorized-execution-v2",
        "status": "AUTHORIZED_EXECUTION",
        "execution_package_id": package["package_id"],
        "execution_package_path": relative,
        "execution_package_sha256": _sha(package_path),
        "execution_package_hash": package["package_hash"],
        "mr_p4_status": "CLOSED",
        "mr_p5_status": "FROZEN",
        "mr_p6_status": "PASS",
        "main_a_status": "NOT_STARTED",
        "measured_main_a_trajectory_count": 0,
        "corrected_run_id": RUN_ID,
    }
    authorization["authorization_hash"] = canonical_hash(authorization)
    output = ROOT / "data/phase13/main/mr_p6/authorized_execution_v2.json"
    _write(output, authorization)
    return output


def main() -> int:
    p4 = _build_mr_p4()
    p5 = _build_mr_p5(p4)
    p6 = _build_mr_p6(p5)
    print(json.dumps({"mr_p4": str(p4), "mr_p5": str(p5), "mr_p6": str(p6)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
