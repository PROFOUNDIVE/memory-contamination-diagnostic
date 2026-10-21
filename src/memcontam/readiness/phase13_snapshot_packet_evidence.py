"""Read-only bindings for existing snapshot QA receipts, never launch authority."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Literal
from xml.etree import ElementTree

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

from memcontam.readiness.phase13_authority_files import read_regular_nofollow
from memcontam.readiness.phase13_cost_policy_models import Sha256


class PacketError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class FilePin(Frozen):
    path: str
    sha256: Sha256

    def read(self) -> bytes:
        path = Path(self.path)
        if path != path.resolve():
            raise PacketError("PACKET_FILE_ALIAS")
        raw = read_regular_nofollow(path)
        if hashlib.sha256(raw).hexdigest() != self.sha256:
            raise PacketError("PACKET_BINDING_MISMATCH:" + self.path)
        return raw


def pin_file(path: Path) -> FilePin:
    raw = read_regular_nofollow(path)
    pin = FilePin(path=str(path), sha256=hashlib.sha256(raw).hexdigest())
    pin.read()
    return pin


class Claim(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)
    task_complete: Literal[True]
    artifacts_sha256: dict[str, Sha256] = Field(min_length=1)
    source_sha256: dict[str, Sha256] = Field(default_factory=dict)
    protected_source_sha256: dict[str, Sha256] = Field(default_factory=dict)
    branches: dict[str, SourceBranch] = Field(default_factory=dict)


class SourceBranch(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)
    root: str
    files_sha256: dict[str, Sha256]


class MechanicalReceipt(Frozen):
    name: Literal["accounting", "durable", "semantics"]
    claim: FilePin
    suite: FilePin

    def verify(self) -> tuple[FilePin, ...]:
        claim = Claim.model_validate_json(self.claim.read())
        pins = tuple(FilePin(path=str(Path(self.claim.path).parent / name), sha256=digest)
                     for name, digest in claim.artifacts_sha256.items())
        for pin in pins:
            pin.read()
        if self.suite not in pins:
            raise PacketError("PACKET_UNBOUND_QA_SUITE")
        suites = ElementTree.fromstring(self.suite.read()).iter("testsuite")
        rows = tuple(suites)
        if not rows or any(int(row.get("tests", "0")) <= 0 or any(
            int(row.get(key, "0")) != 0 for key in ("failures", "errors", "skipped")
        ) for row in rows):
            raise PacketError("PACKET_MECHANICS_INCOMPLETE")
        return (self.claim, self.suite, *pins)


class AuthorityEntry(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)
    local_path: str
    artifact_sha256: Sha256
    byte_length: int


class LocalAuthority(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)
    as_of: str
    current_freshness: Literal["unknown"]
    source_generation: Literal["unknown"]
    enforced_immutability: Literal[False]
    entries: tuple[AuthorityEntry, ...]


class PendingPanel(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)
    status: Literal["UNVERIFIED"]
    authentic_rating_rows: Literal[0]


class PendingOwner(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)
    status: Literal["PENDING"]
    owner_statement_received: Literal[False]


class PendingGates(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)
    human_panel: PendingPanel
    owner_off_registry_fact: PendingOwner


class PendingDecisions(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)
    scientific_adoption: Literal[False]
    paid_launch: Literal[False]


class FinancialBoundary(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)
    combined_incremental_provider_budget: Literal[45]


class PendingInventory(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)
    gates: PendingGates
    decisions: PendingDecisions
    financial_boundary: FinancialBoundary


def reopen_native(descriptor: FilePin) -> tuple[tuple[FilePin, ...], dict[str, JsonValue]]:
    from memcontam.readiness.phase13_main_live_evidence import (
        MainReconciliationEvidence,
        MainUnitEvidence,
    )
    from memcontam.readiness.phase13_main_production import ProductionObject
    from memcontam.readiness.phase13_main_runner_ledger import MainRunLedger
    from memcontam.readiness.phase13_main_runner_models import MainRunBinding
    from memcontam.readiness.phase13_main_runner_store import expected_metadata

    class Reopen(Frozen):
        unit: ProductionObject
        binding: MainRunBinding

    opened = Reopen.model_validate_json(descriptor.read())
    root = Path(descriptor.path).parent
    database = root / "ledger.sqlite3"
    if Path(str(database) + "-wal").exists():
        raise PacketError("PACKET_QA_LEDGER_REQUIRES_CHECKPOINT")
    if opened.binding.core_authorization_gate_krw != 0:
        raise PacketError("PACKET_QA_CAP_NOT_ZERO")
    ledger = MainRunLedger(database, (opened.unit,), 0)
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro&immutable=1", uri=True)) as connection:
        connection.execute("PRAGMA query_only=ON")
        if dict(connection.execute("SELECT key, value FROM metadata")) != expected_metadata(
            opened.binding, (opened.unit,),
        ):
            raise PacketError("PACKET_QA_LEDGER_IDENTITY_MISMATCH")
        ledger._validate_integrity(connection)
    paths = (*sorted((root / "units").glob("*.json")),
             *sorted((root / "reconciliation").glob("*.json")))
    if len(paths) != 1:
        raise PacketError("PACKET_QA_DURABLE_RECORD_REQUIRED")
    record_pin = pin_file(paths[0])
    if paths[0].parent.name == "units":
        record = MainUnitEvidence.model_validate_json(record_pin.read())
    else:
        record = MainReconciliationEvidence.model_validate_json(record_pin.read()).diagnostic_partial
    if record is None or record.evidence.runtime_evidence.diagnostic_scope is None:
        raise PacketError("PACKET_QA_NATIVE_SCOPE_REQUIRED")
    scope = record.evidence.runtime_evidence.diagnostic_scope
    return ((descriptor, pin_file(database), record_pin), {
        "baseline": opened.unit.memory_baseline, "provenance": "provider_free_qa",
        "state_sha256": scope.state_sha256, "input_sha256": scope.input_sha256,
        "native_scope": scope.model_dump(mode="json"),
        "post_state_utf8": record.evidence.runtime_evidence.diagnostic_post_state_utf8,
        "scientific_binding": None,
    })


def json_bytes(value: JsonValue) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode()


def json_object(raw: bytes) -> dict[str, JsonValue]:
    return TypeAdapter(dict[str, JsonValue]).validate_json(raw)


def current_qa_sources(receipts: tuple[MechanicalReceipt, ...], root: Path) -> tuple[FilePin, ...]:
    sources: dict[str, FilePin] = {}
    by_name = {receipt.name: receipt for receipt in receipts}
    for name in ("accounting", "durable", "semantics"):
        claim = Claim.model_validate_json(by_name[name].claim.read())
        for relative, digest in {**claim.protected_source_sha256, **claim.source_sha256}.items():
            path = str(root / relative)
            sources[path] = FilePin(path=path, sha256=digest)
        for branch in claim.branches.values():
            for relative, digest in branch.files_sha256.items():
                path = str(Path(branch.root) / relative)
                sources[path] = FilePin(path=path, sha256=digest)
    for pin in sources.values():
        pin.read()
    return tuple(sources.values())
