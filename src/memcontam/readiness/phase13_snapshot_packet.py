from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Final, Literal

import pydantic
from pydantic import JsonValue
from pydantic_core import _pydantic_core

from memcontam.readiness.phase13_authority_files import (
    authority_directory,
    read_regular_nofollow,
)

from .phase13_snapshot_packet_evidence import (
    FilePin,
    Frozen,
    LocalAuthority,
    MechanicalReceipt,
    PacketError,
    PendingInventory,
    current_qa_sources,
    json_bytes,
    json_object,
    pin_file,
    reopen_native,
)

AUTHORITY_SHA256: Final = "5cf0c3772739aa5124866752a99bcf5dfcbb77bd3e364de438b2ebc54eee34d4"
FINANCIAL_SHA256: Final = "8ec9ca4c645c3c8a5d972ba4daaa4c78a01cd59cf726ef874bc53a080ab47c6c"
MECHANICAL_CLAIM_SHA256: Final = {
    "accounting": "768e76ba6a4ced34c3cc4feea0cb456cdcfbc051c1fb2d79e45611021cbc3250",
    "durable": "f417276c92b9c253bed2713370e63c418cfdf6d022f8e69b7c8874ee12c928f8",
    "semantics": "25942c82b56e5205c1a2508431e523dd81d1f14b26068a1d077c1320d8285cb4",
}
ROOT: Final = Path(__file__).resolve().parents[3]
FILES: Final = ("proposal.json", "registration-proposal.json", "input-manifest.json",
                "readiness.json", "launch-request.md")
PENDING_FIELDS: Final = ("owner_fact", "human_panel", "protocol_owner_disposition", "dc_family",
    "scientific_schedule", "run_id", "stage_limits", "count_limits", "attempt_retry_limits",
    "price_risk_acceptance", "launch_instruction", "scientific_registration")


class PreparationProposal(Frozen):
    authority_manifest: FilePin
    financial_authorization: FilePin
    pending: FilePin
    handoff: FilePin
    mechanics: tuple[MechanicalReceipt, ...]
    qa_reopens: tuple[FilePin, ...]
    scientific_adoption: Literal[False] = False
    paid_launch: Literal[False] = False
    real_spend_cap_usd: Literal[0] = 0
    owner_fact: None = None
    human_panel: None = None
    protocol_owner_disposition: None = None
    dc_family: None = None
    scientific_schedule: None = None
    run_id: None = None
    stage_limits: None = None
    count_limits: None = None
    attempt_retry_limits: None = None
    price_risk_acceptance: None = None
    launch_instruction: None = None
    scientific_registration: None = None


def documents(proposal: PreparationProposal, origin: FilePin) -> dict[str, bytes]:
    if proposal.authority_manifest.sha256 != AUTHORITY_SHA256:
        raise PacketError("PACKET_AUTHORITY_PIN_INVALID")
    if proposal.financial_authorization.sha256 != FINANCIAL_SHA256:
        raise PacketError("PACKET_FINANCIAL_BOUNDARY_INVALID")
    authority = LocalAuthority.model_validate_json(proposal.authority_manifest.read())
    PendingInventory.model_validate_json(proposal.pending.read())
    pins = [origin, proposal.authority_manifest, proposal.financial_authorization,
            proposal.pending, proposal.handoff]
    for pin in pins:
        pin.read()
    for entry in authority.entries:
        pin = FilePin(path=entry.local_path, sha256=entry.artifact_sha256)
        if len(pin.read()) != entry.byte_length:
            raise PacketError("PACKET_AUTHORITY_SIZE_MISMATCH")
        pins.append(pin)
    if (len(proposal.mechanics) != 3 or {row.name for row in proposal.mechanics}
            != {"accounting", "durable", "semantics"}):
        raise PacketError("PACKET_REQUIRED_MECHANICS_MISSING")
    for receipt in proposal.mechanics:
        if receipt.claim.sha256 != MECHANICAL_CLAIM_SHA256[receipt.name]:
            raise PacketError("PACKET_QA_CLAIM_PIN_INVALID")
        pins.extend(receipt.verify())
    pins.extend(current_qa_sources(proposal.mechanics, ROOT))
    objects: list[JsonValue] = []
    baselines = set()
    for descriptor in proposal.qa_reopens:
        dependencies, native = reopen_native(descriptor)
        pins.extend(dependencies)
        objects.append(native)
        baselines.add(str(native["baseline"]))
    if baselines != {"bot_style", "dc_rs"}:
        raise PacketError("PACKET_BOTH_NATIVE_BASELINES_REQUIRED")
    sources = (*sorted((ROOT / "src").rglob("*.py")), ROOT / "pyproject.toml",
               ROOT / "requirements.lock", ROOT / "scripts/prepare_phase13_snapshot_diagnostic.py",
               Path(sys.executable).resolve(), Path(pydantic.__file__).resolve(),
               Path(_pydantic_core.__file__).resolve())
    pins.extend(pin_file(path) for path in sources)
    by_path = {pin.path: pin for pin in pins}
    if any(by_path[pin.path] != pin for pin in pins):
        raise PacketError("PACKET_CONFLICTING_BINDINGS")
    registration: dict[str, JsonValue] = {
        "status": "PROPOSED_NOT_ADOPTED", "development_only": True,
        "scientific_adoption": False, "paid_launch": False,
        "proposed_metadata": {"inputs_per_task": 8, "units": 128,
                              "status": "PROPOSED_NOT_ADOPTED"},
        "tasks": ["math_equation_balancer", "word_sorting"],
        "baselines": ["bot_style", "dc_rs"], "arms": ["clean", "correct", "irrelevant", "contam"],
        "scientific_schedule": None, "scientific_native_objects": None,
        "scientific_input_state_bindings": None, "scientific_registration": None,
        "completed_scientific_units": 0,
    }
    readiness: dict[str, JsonValue] = {
        "status": "DIAGNOSTIC_READY_FOR_SEPARATE_AUTHORIZATION", "mechanical_qa": "PASSED",
        "launch_eligible": False, "launch_eligibility_reason": "PREPARATION_ONLY_NOT_AUTHORIZATION",
        "paid_launch": False, "scientific_adoption": False, "real_spend_cap_usd": 0,
        "provider_calls": 0, "count_calls": 0, "scientific_runs": 0,
        "pending": {name: {"status": "PENDING", "value": None} for name in PENDING_FIELDS},
        "authority": {"as_of": authority.as_of, "current_freshness": authority.current_freshness,
            "source_generation": authority.source_generation, "enforced_immutability": False,
            "local_manifest_sha256": proposal.authority_manifest.sha256, "required_for_lane": False},
        "financial_boundary": {"currency": "USD", "combined_incremental_provider_budget": 45,
            "scope": "all lanes combined; financial boundary only", "source": proposal.financial_authorization.model_dump(mode="json"),
            "future_run_cap": None, "actual_combined_spend": None, "remaining": None},
        "accounting_model": {"bot_style": ["bot_problem_distill", "bot_instantiate_solve", "bot_thought_distill"],
            "dc_rs": ["dc_rs_synthesize", "dc_rs_generate"], "per_stage": {
                "generation_calls": "bind actual receipts", "count_calls": "separately bind actual receipts",
                "attempts_retries": "retain every attempt; pending future limits",
                "unknown_price": "unknown, never zero", "reservation": "atomic shared authority before any future dispatch",
                "completed_receipt": "no redispatch after reopen"}},
        "limitations": ["Native mechanics use deterministic test transport, not provider fidelity.",
            "QA objects are synthetic development evidence, not scientific candidate/family bindings.",
            "Checksums detect tamper, not enforced immutability; current freshness is unknown."],
    }
    manifest: dict[str, JsonValue] = {
        "proposal_source": origin.model_dump(mode="json"),
        "bindings": [by_path[name].model_dump(mode="json") for name in sorted(by_path)],
        "qa_native_objects": objects, "runtime": {"python": sys.version, "executable": str(Path(sys.executable).resolve())},
    }
    request = ("# Separate Authorization Request (Not Granted)\n\n"
        "Mechanical preparation only. No scientific adoption, provider dispatch or launch is authorized.\n"
        "The shared USD45 financial boundary remains unchanged; this preparation has real spend cap0.\n"
        "128 units are PROPOSED_NOT_ADOPTED metadata, not an executed or registered batch.\n\n"
        + "\n".join(f"- {name}: PENDING (null; attributable external evidence required)" for name in PENDING_FIELDS)
        + "\n\nCurrent cloud freshness/source generation unknown; enforced immutability false.\n")
    return {"proposal.json": origin.read(), "registration-proposal.json": json_bytes(registration),
            "input-manifest.json": json_bytes(manifest), "readiness.json": json_bytes(readiness),
            "launch-request.md": request.encode()}


def checksums(output: Path) -> bytes:
    return "".join(f"{hashlib.sha256(read_regular_nofollow(output / name)).hexdigest()}  {name}\n"
                   for name in FILES).encode()


def prepare_packet(proposal_path: Path, output: Path) -> dict[str, JsonValue]:
    if ".." in output.parts:
        raise PacketError("PACKET_OUTPUT_ALIAS")
    output = Path(os.path.abspath(output))
    if output.exists() or output.is_symlink():
        raise PacketError("PACKET_OUTPUT_COLLISION")
    origin = pin_file(Path(os.path.abspath(proposal_path)))
    proposal = PreparationProposal.model_validate_json(origin.read())
    payloads = documents(proposal, origin)
    manifest = json_object(payloads["input-manifest.json"])
    bindings = manifest["bindings"]
    assert isinstance(bindings, list)
    for binding in bindings:
        assert isinstance(binding, dict) and isinstance(binding["path"], str)
        parent = Path(binding["path"]).parent
        if parent != proposal_path.absolute().parent and output.is_relative_to(parent):
            raise PacketError("PACKET_OUTPUT_NESTED_IN_INPUT")
    if output != output.resolve():
        raise PacketError("PACKET_OUTPUT_ALIAS")
    sums = "".join(f"{hashlib.sha256(payloads[name]).hexdigest()}  {name}\n" for name in FILES).encode()
    with authority_directory(output.parent) as parent:
        os.mkdir(output.name, dir_fd=parent)
        with authority_directory(output) as directory:
            for name, raw in {**payloads, "SHA256SUMS": sums}.items():
                descriptor = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                     0o600, dir_fd=directory)
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(raw)
                    stream.flush()
                    os.fsync(stream.fileno())
            os.fsync(directory)
        os.fsync(parent)
    return json_object(payloads["readiness.json"])


def verify_packet(output: Path) -> dict[str, JsonValue]:
    output = Path(os.path.abspath(output))
    if output != output.resolve() or {path.name for path in output.iterdir()} != {*FILES, "SHA256SUMS"}:
        raise PacketError("PACKET_MEMBERS_OR_ALIAS_INVALID")
    for path in output.iterdir():
        if path.stat().st_nlink != 1:
            raise PacketError("PACKET_OUTPUT_ALIAS")
        pin_file(path)
    if read_regular_nofollow(output / "SHA256SUMS") != checksums(output):
        raise PacketError("PACKET_CHECKSUM_MISMATCH")
    manifest = json_object(read_regular_nofollow(output / "input-manifest.json"))
    origin = FilePin.model_validate(manifest["proposal_source"])
    proposal = PreparationProposal.model_validate_json(read_regular_nofollow(output / "proposal.json"))
    expected = documents(proposal, origin)
    if any(read_regular_nofollow(output / name) != raw for name, raw in expected.items()):
        raise PacketError("PACKET_RECOMPUTATION_MISMATCH")
    return json_object(expected["readiness.json"])
