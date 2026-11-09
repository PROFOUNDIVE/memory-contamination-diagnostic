from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import JsonValue

from memcontam.baselines import dynamic_cheatsheet_phase12 as dc
from memcontam.contamination.phase13_legacy_dc_rs import LegacyDcRsRecord
from memcontam.memory.checkpoint_v3 import NativeEntry, NativeState
from memcontam.readiness import phase13_diagnostic_cli as cli
from memcontam.readiness import phase13_diagnostic_trusted_admission as trust
from memcontam.readiness.phase13_diagnostic_cli_models import NativeInput, RegisteredDiagnostic, RegistrationSubject
from memcontam.readiness.phase13_legacy_rag_audit import _canonical_signature_bytes
from memcontam.verifiers.math_equation_balancer import _parse_input

from .phase13_budget_fixtures import pin, save
from .phase13_cli_fixtures import signed_evidence
from .phase13_recovery_offline import install_offline_denial

EVIDENCE = Path("/home/hyunwoo/phase13_artifacts/phase13-results-and-main-entry-20261022-001")


def metadata_fixture(root: Path, monkeypatch):
    original = EVIDENCE / "task-15/scratch/candidate.json"
    if not original.exists():
        pytest.skip("Task15 retained synthetic metadata not installed")
    install_offline_denial(monkeypatch)
    saved = RegisteredDiagnostic.model_validate_json(original.read_bytes()).subject
    raw = saved.model_dump(mode="json")
    for scope, observation in zip(raw["scopes"], raw["binding"]["observations"], strict=True):
        native = json.loads(scope["input_utf8"])
        if native["baseline"] == "dc_rs" and native["arm"] != "clean":
            state = NativeState.from_mapping(json.loads(scope["state_utf8"]))
            entry = next(entry for entry in state.entries if isinstance(entry, NativeEntry)
                         and entry.entry_id == native["branch"]["injected_root_id"])
            render_id = f"legacy-dc-rs-render-v2::{entry.entry_id}::{hashlib.sha256(entry.content.encode()).hexdigest()[:16]}"
            state_value = state.to_mapping()
            for archive in state_value["native_state"]["archive"]:
                if archive["entry_id"] == entry.entry_id:
                    archive["metadata"]["render_id"] = render_id
            state_value["entries"] = [dc._archive_native(dc.MemoryEntry.model_validate(row)).to_mapping()
                                      for row in state_value["native_state"]["archive"]]
            native["branch"]["native_render_id"] = render_id
            scope["state_utf8"] = json.dumps(state_value, sort_keys=True, separators=(",", ":"))
            scope["input_utf8"] = json.dumps(native, sort_keys=True, separators=(",", ":"))
            for kind in ("input", "state"):
                scope[f"{kind}_sha256"] = hashlib.sha256(scope[f"{kind}_utf8"].encode()).hexdigest()
                observation[f"{kind}_sha256"] = scope[f"{kind}_sha256"]
    proposal = json.loads((EVIDENCE / "diagnostic/scope-proposal.json").read_bytes())
    natives = [NativeInput.model_validate_json(scope["input_utf8"]) for scope in raw["scopes"]]
    keys = {role: Ed25519PrivateKey.generate() for role in ("science", "launch", "credential")}
    _, policy = signed_evidence(trust, root, keys=keys)
    policy_path = root / "synthetic-trust.json"
    save(policy_path, policy.model_dump(mode="json"))
    monkeypatch.setattr(trust, "TRUST_POLICY_PATH", policy_path)
    monkeypatch.setattr(trust, "load_policy", lambda: policy)
    groups, renders, witnesses = [], [], [pin(policy_path)]
    witness = save(root / "native-metadata-qa.json", {"purpose": "SYNTHETIC_QA_METADATA_ONLY_NOT_OWNER_FACTS"})
    witnesses.append(witness)
    for item in proposal["inputs"]:
        native = next(native for native in natives if native.task_name == item["task"])
        item["task_instance"] = native.tasks[0].model_dump(mode="json")
        if item["task"] == "math_equation_balancer":
            parsed = _parse_input(native.tasks[0].input["input"])
            assert parsed is not None
            signature: dict[str, int | list[int] | list[str]] = {"ordered_operands": list(parsed[0]), "target_value": parsed[1]}
        else:
            signature = {"tokens": sorted(native.tasks[0].input["words"])}
        item["canonical_signature"] = hashlib.sha256(_canonical_signature_bytes(signature)).hexdigest()
    inventory: list[JsonValue] = [None] * 17
    for item in proposal["inputs"]:
        inventory[item["inventory_index"]] = {"task": item["task"], "task_instance": item["task_instance"],
                                               "canonical_signature": item["canonical_signature"]}
    inventory_pin = save(root / "synthetic-qa-inventory.json", inventory)
    for item in proposal["inputs"]:
        item["inventory_binding"] = inventory_pin
    for index, group in enumerate(proposal["checkpoint_groups"]):
        scope, native = next((scope, native) for scope, native in zip(saved.scopes, natives, strict=True)
            if (native.task_name, native.baseline, native.arm) == (group["task"], group["baseline"], "clean"))
        group["suffix_sample_id"] = native.tasks[0].sample_id
        state_path = root / f"state-{index}.json"
        state_path.write_text(scope.state_utf8)
        state = pin(state_path)
        witnesses.append(state)
        groups.append({"group_id": group["group_id"], "prefix_id": native.branch.prefix_identity,
            "source_identity": native.branch.source_identity, "checkpoint_id": f"synthetic-checkpoint-{index}",
            "checkpoint_index": native.branch.checkpoint_index, "prefix_sample_ids": ["synthetic-prefix-only"],
            "state_binding": state, "evidence": witness})
    for unit in proposal["ordinary_units"]:
        native = next(native for native in natives
            if (native.task_name, native.baseline, native.arm) == (unit["task"], unit["baseline"], unit["arm"]))
        unit["sample_id"] = native.tasks[0].sample_id
        unit["input_binding"] = inventory_pin
        if unit["render"] is not None:
            if unit["baseline"] == "dc_rs":
                scope = raw["scopes"][natives.index(native)]
                state = NativeState.from_mapping(json.loads(scope["state_utf8"]))
                entry = next(entry for entry in state.entries if isinstance(entry, NativeEntry)
                             and entry.entry_id == native.branch.injected_root_id)
                payload = json.loads(entry.content)
                assert entry.render_id is not None
                record = LegacyDcRsRecord(role="false" if unit["arm"] == "contam" else unit["arm"],
                    candidate_id=entry.entry_id, query=payload["input"], response=payload["raw_output"],
                    query_sha256=hashlib.sha256(payload["input"].encode()).hexdigest(),
                    response_sha256=hashlib.sha256(payload["raw_output"].encode()).hexdigest(),
                    serialized_content_sha256=hashlib.sha256(entry.content.encode()).hexdigest(), render_id=entry.render_id)
                unit["render"].update(record.model_dump(mode="json"), triplet_id=native.branch.candidate_triplet_id)
            else:
                renders.append({"unit_id": unit["unit_id"], "renderer_id": "synthetic-bot-renderer",
                    "render_id": native.branch.native_render_id, "evidence": witness})
    proposal_path = root / "synthetic-proposal.json"
    save(proposal_path, proposal)
    witnesses.append(pin(proposal_path))
    witnesses.extend(item["inventory_binding"] for item in proposal["inputs"])
    witnesses.append(proposal["dc_contract"]["family"])
    witnesses.extend(unit["render"]["family_binding"] for unit in proposal["ordinary_units"] if unit["render"] is not None)
    sources = {row.path: pin(Path(row.path)) for row in saved.binding.source_files}
    sources.update({str(path): pin(path) for path in cli.cli_source_paths()})
    sources.update({row["path"]: row for row in witnesses})
    runtime = json.loads(Path(saved.runtime_registration.path).read_bytes())
    contract = json.loads(Path(runtime["runtime_contract"]["path"]).read_bytes())
    contract["observations"] = raw["binding"]["observations"]
    contract_pin = save(root / "synthetic-runtime-contract.json", contract)
    runtime["runtime_contract"] = contract_pin
    sources[contract_pin["path"]] = contract_pin
    runtime["authorization_sha256"] = pin(root / "synthetic-owner.json")["sha256"]
    runtime_pin = save(root / "runtime-registration.json", runtime)
    sources[runtime_pin["path"]] = runtime_pin
    raw["registration_path"] = str(root / "registration.json")
    raw["parent_root"] = str(root / "run")
    raw["binding"]["ledger_path"] = str(root / "run/ledger.sqlite3")
    raw["binding"]["source_files"] = list(sources.values())
    raw["runtime_registration"] = runtime_pin
    raw["binding"]["registration_sha256"] = runtime_pin["sha256"]
    raw["binding"]["authorization_sha256"] = runtime["authorization_sha256"]
    from memcontam.readiness.phase13_live_diagnostic_models import LiveDiagnosticBinding
    binding = LiveDiagnosticBinding.model_validate_json(json.dumps(raw["binding"]))
    for child in raw["children"]:
        child["run_root"] = str(root / "run" / child["child_execution_id"])
    for scope in raw["scopes"]:
        scope.update(binding_sha256=binding.sha256(), source_files=list(sources.values()))
    interim = RegistrationSubject.model_validate_json(json.dumps(raw))
    unit_ids = {old.unit_id: new.unit_id for old, new in zip(saved.units(), interim.units(), strict=True)}
    for operation in raw["operations"]:
        operation.update(unit_id=unit_ids[operation["unit_id"]], execution_binding_sha256=binding.sha256(),
            attempt_id=hashlib.sha256((binding.sha256() + operation["request_id"]).encode()).hexdigest())
    subject = RegistrationSubject.model_validate_json(json.dumps(raw))
    metadata = root / "preparation.json"
    save(metadata, {"schema_version": "phase13_diagnostic_proposal_preparation_v1", "proposal": pin(proposal_path),
        "subject": subject.model_dump(mode="json"), "groups": groups, "renders": renders})
    admission, _ = signed_evidence(trust, root, subject.sha256(), keys)
    eligibility = root / "signed-admission.json"
    save(eligibility, admission.model_dump(mode="json"))
    return proposal_path, metadata, eligibility, subject, keys
