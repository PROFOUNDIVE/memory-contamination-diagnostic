from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from memcontam.readiness.phase13_authority_files import load_authority_v3
from memcontam.readiness.phase13_main_checkpoint import CommonCheckpointRegistry
from memcontam.readiness.phase13_main_production import ProductionObject, _stages
from memcontam.readiness.phase13_main_resource_contract import RESOURCE_PATHS
from memcontam.readiness.phase13_v3_authority_models import V3Identity
from memcontam.readiness.phase13_v3_builder_inputs import PREFIX, STAGES, STATIC_PATHS
from memcontam.readiness.phase13_v3_cost import activate_policy, build_witness, freeze_base
from memcontam.readiness.phase13_v3_cost_binding import MRP4Costs, bind_package_costs
from memcontam.readiness.phase13_v3_cost_models import (
    CostUnit,
    FinalOrder,
    PrefreezeBindings,
    StageOccurrences,
    canonical_bytes,
    digest,
)
from memcontam.readiness.phase13_v3_entrypoint import SelectionRequest
from memcontam.readiness.phase13_v3_entrypoint_models import (
    ExecutionResourceV3,
    MainAuthorizationV3,
    MainExecutionPackageV3,
    MainLiveContractV3,
)
from memcontam.readiness.phase13_v3_publication import P4_PATHS, P5_PATHS
from memcontam.readiness.phase13_v3_resource_files import FileBinding, read_files
from memcontam.readiness.phase13_v3_runtime_identity import ROOT, freeze_runtime_identity
from memcontam.readiness.phase13_v3_source_closure import (
    GovernedInventory,
    ResourceClosure,
    freeze_governed,
    freeze_resources,
)

from .phase13_corrective_identity import corrective_identity
from .phase13_count_fake import fake_count_pricing

AUTHORITY = Path("/home/hyunwoo/gdrive_undergrad_research/PeerJ fast-track/References/Theoretical Artifacts")
REPAIR_ROOT = Path(__file__).resolve().parents[1]
# Historical v2 tests read frozen artifacts; current disposable packages use REPAIR_ROOT.
RESOURCE_ROOT = Path("/home/hyunwoo/git/memory-contamination-diagnostic-phase13-main-execution-entrypoint-closure")


@pytest.fixture(scope="session")
def entrypoint_bytes():
    return build_entrypoint_bytes((0,))


def build_entrypoint_bytes(
    seeds: tuple[int, ...], governed_source: GovernedInventory | None = None,
    mr_p4_closure: FileBinding | None = None, generated_closure: ResourceClosure | None = None,
    *, execution_identity: V3Identity | None = None,
    production_units: tuple[ProductionObject, ...] | None = None,
) -> dict[str, bytes]:
    execution_identity = execution_identity or corrective_identity()
    authority = load_authority_v3(AUTHORITY, identity=execution_identity)
    identity = freeze_runtime_identity()
    generated_roles = {"activated_policy", "base_inputs", "cost_witness"}
    resources = {row.binding.path: row.raw for row in read_files(
        REPAIR_ROOT, tuple(path for role, path in RESOURCE_PATHS.items() if role not in generated_roles))}
    if production_units is None:
        unit_ids = tuple(hashlib.sha256(json.dumps(["phase13-main-a-disjoint-unit-id-v1", "NO_MEMORY_SINGLETON",
            seed, "game24", None, "NOT_APPLICABLE"], separators=(",", ":")).encode()).hexdigest() for seed in seeds)
        checkpoint_raw = resources[RESOURCE_PATHS["common_checkpoint_registry"]]
        checkpoint = CommonCheckpointRegistry.model_validate_json(checkpoint_raw)
        units = tuple(ProductionObject(sequence, unit_id, "NO_MEMORY_SINGLETON", seed, "game24", None,
            "NOT_APPLICABLE", None, 0, "game24|nomem", checkpoint.tasks["game24"].seeds[seed].suffix_sample_ids_sha256,
            hashlib.sha256(resources[RESOURCE_PATHS["observability_packet"]]).hexdigest(), hashlib.sha256(checkpoint_raw).hexdigest())
            for sequence, (unit_id, seed) in enumerate(zip(unit_ids, seeds, strict=True)))
    else:
        units = production_units
        unit_ids = tuple(unit.unit_id for unit in units)
    policy = activate_policy(authority)
    bindings = PrefreezeBindings(**{
        name: hashlib.sha256(name.encode()).hexdigest() for name in PrefreezeBindings.model_fields
    })
    cost_units = tuple(CostUnit(unit_id=unit.unit_id, stages=tuple(
        StageOccurrences(stage_id=STAGES[stage], calls=calls) for stage, calls in _stages(unit)
    )) for unit in units)
    from memcontam.readiness.phase13_v3_retry import allocate_retry_reservations
    initial = freeze_base(policy, bindings, cost_units)
    retries = allocate_retry_reservations(units, initial)
    count_operations = sum(stage.calls for unit in cost_units for stage in unit.stages) + len(retries)
    base = freeze_base(policy, bindings, cost_units,
                       retry_reservations=retries, count_pricing=fake_count_pricing(count_operations, "0.001"))
    phase4 = MRP4Costs(policy=base.policy, base=base, witness=build_witness(base))
    for role, model in (("activated_policy", base.policy), ("base_inputs", base), ("cost_witness", phase4.witness)):
        resources[RESOURCE_PATHS[role]] = canonical_bytes(model)
    manifest_path = "data/phase13/main/mr_p4/corrected_v3/manifest_v3.json"
    manifest = FileBinding(path=manifest_path, size=0, sha256=hashlib.sha256(b"").hexdigest())
    closure_raw = json.dumps([manifest.model_dump()], sort_keys=True, separators=(",", ":")).encode() + b"\n"
    if mr_p4_closure is None:
        resources[manifest_path] = b""
    governed_source = governed_source or GovernedInventory(governed_source_commit="a" * 40, rows=(),
        governed_tree_sha256=hashlib.sha256(b"[]\n").hexdigest())
    mr_p4_closure = mr_p4_closure or manifest
    generated_closure = generated_closure or ResourceClosure(rows=(manifest,),
        resource_closure_sha256=hashlib.sha256(closure_raw).hexdigest())
    package = MainExecutionPackageV3(schema_version="phase13_main_execution_freeze_v3", identity=execution_identity,
        package_id=execution_identity.package_id,
        status="FROZEN", authority=authority, runtime_identity=identity, tranche_unit_count=120,
        measured_main_a_trajectory_count=0,
        governed_source=governed_source, mr_p4_closure=mr_p4_closure,
        generated_closure=generated_closure,
        resources=tuple(ExecutionResourceV3(role=role, path=path, size=len(resources[path]),
            sha256=hashlib.sha256(resources[path]).hexdigest()) for role, path in RESOURCE_PATHS.items()),
        production=units, final_order=FinalOrder(unit_ids=unit_ids, runtime_hash=digest(identity),
            request_hash="c" * 64, tokenizer_hash="d" * 64))
    first = bind_package_costs(package, phase4)
    units = tuple(replace(unit, projected_cost_krw=row.projected_krw)
                  for unit, row in zip(units, first.resources.proof.projected_krw, strict=True))
    bound = bind_package_costs(package.model_copy(update={"production": units}), phase4)
    contract = MainLiveContractV3(schema_version="phase13_main_live_contract_v3", identity=execution_identity,
        package_core_hash=bound.package.package_core_hash, cost_proof_hash=bound.package.cost_proof_hash,
        projected_krw=tuple((unit.unit_id, unit.projected_cost_krw) for unit in units), contract_hash="0" * 64)
    contract = contract.model_copy(update={"contract_hash": digest(contract, "contract_hash")})
    package = bound.package.model_copy(update={"live_contract_hash": contract.contract_hash})
    package = package.model_copy(update={"package_hash": digest(package, "package_hash")})
    resources["data/phase13/main/cost_envelope_v3/complete_inputs_v3.json"] = canonical_bytes(bound.resources.complete)
    resources["data/phase13/main/cost_envelope_v3/cost_proof_v3.json"] = canonical_bytes(bound.resources.proof)
    resources["data/phase13/main/main_live_contract_v3.json"] = canonical_bytes(contract)
    resources["package.json"] = canonical_bytes(package)
    authorization = MainAuthorizationV3(schema_version="phase13_main_authorization_v3", identity=execution_identity,
        authorization_id=execution_identity.authorization_id, status="AUTHORIZED_EXECUTION", execution_package_id=execution_identity.package_id,
        execution_package_path="package.json", execution_package_sha256=digest(package), execution_package_hash=package.package_hash,
        authorization_hash="0" * 64, main_a_status="NOT_STARTED", measured_main_a_trajectory_count=0)
    authorization = authorization.model_copy(update={"authorization_hash": digest(authorization, "authorization_hash")})
    resources["authorization.json"] = canonical_bytes(authorization)
    resources["authorization.sha256"] = (digest(authorization) + "\n").encode()
    return resources


@pytest.fixture
def entrypoint_fixture(tmp_path, entrypoint_bytes):
    for path, raw in entrypoint_bytes.items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)

    seal_fixture_closure(tmp_path)
    return SelectionRequest(tmp_path, tmp_path / "package.json", tmp_path / "authorization.json",
        AUTHORITY, tmp_path / "authorization.sha256", corrective_identity().run_id)


def seal_fixture_closure(root: Path) -> None:
    packet = json.loads((ROOT / RESOURCE_PATHS["observability_packet"]).read_bytes())
    identities = (*packet["implementation_identities"].values(), *packet["verifier_identities"].values(),
                  *packet["applicability_identities"].values())
    governed = (
        "pyproject.toml",
        "scripts/build_phase13_corrected_main_closure.py",
        "scripts/diagnose_phase13_mr_p5_closure.py",
        "scripts/build_phase13_main_registries.py",
        "src/memcontam/__init__.py",
        *(row["path"] for row in identities if row["path"].startswith("src/memcontam/")),
    )
    for path in set(governed):
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / path).read_bytes() if (ROOT / path).is_file() else b"")
    environment = {**os.environ, "GIT_MASTER": "1", "GIT_CONFIG_NOSYSTEM": "1"}
    subprocess.run(("git", "-C", str(root), "init", "-q"), check=True, env=environment)
    subprocess.run(("git", "-C", str(root), "add", "."), check=True, env=environment)
    subprocess.run(("git", "-C", str(root), "-c", "user.name=Fixture", "-c",
        "user.email=fixture@invalid", "-c", "core.hooksPath=/dev/null", "commit", "-qm", "fixture"),
        check=True, env=environment)
    commit = subprocess.run(("git", "-C", str(root), "rev-parse", "HEAD"), check=True,
                            capture_output=True, text=True, env=environment).stdout.strip()
    expected = (*STATIC_PATHS, *(PREFIX + path for path in P4_PATHS),
                *(PREFIX + path for path in P5_PATHS[:-1]))
    for path in expected:
        target = root / path
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        source = REPAIR_ROOT / path
        target.write_bytes(source.read_bytes())
    inventory = freeze_governed(root, commit)
    manifest_path = PREFIX + P4_PATHS[-1]
    manifest, = tuple(row for row in freeze_resources(root, expected).rows if row.path == manifest_path)
    package = MainExecutionPackageV3.model_validate_json((root / "package.json").read_bytes())
    seeds = tuple(unit.seed for unit in package.production)
    for path, raw in build_entrypoint_bytes(
        seeds, inventory, manifest, execution_identity=package.identity,
        production_units=package.production,
    ).items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    closure = freeze_resources(root, expected)
    for path, raw in build_entrypoint_bytes(
        seeds, inventory, manifest, closure, execution_identity=package.identity,
        production_units=package.production,
    ).items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
