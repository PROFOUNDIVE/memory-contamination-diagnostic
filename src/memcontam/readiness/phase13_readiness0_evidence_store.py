from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from memcontam.experiment.phase12.filter_challenge.mft_state_models import JsonValue
from memcontam.readiness.phase13_readiness0_live_models import (
    ArtifactBinding,
    CaseEvidence,
    EvidenceManifest,
    F1CRuntimeMetadata,
)


@dataclass(frozen=True, slots=True)
class EvidenceStoreInputs:
    output_dir: Path
    request_sha256: str
    authorization_sha256: str
    f1c_sha256: str
    f1c_runtime: F1CRuntimeMetadata


class EvidenceStore:
    def __init__(self, inputs: EvidenceStoreInputs) -> None:
        self._inputs = inputs
        self._rows: list[CaseEvidence] = []
        inputs.output_dir.mkdir(parents=True, exist_ok=True)
        self._cases_path = inputs.output_dir / "cases.jsonl"
        self._cases_path.write_bytes(b"")

    def append(
        self,
        row: CaseEvidence,
        status: Literal["PARTIAL", "FAILED"],
    ) -> None:
        raw = (row.model_dump_json() + "\n").encode()
        with self._cases_path.open("ab") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        self._rows.append(row)
        self._write_manifest(status, None if status == "PARTIAL" else "READINESS0_CASE_FAILED")

    def fail(self, code: str) -> None:
        self._write_manifest("FAILED", code)

    def pass_(self) -> str:
        return self._write_manifest("PASS", None)

    def _write_manifest(
        self,
        status: Literal["PASS", "FAILED", "PARTIAL"],
        failure_code: str | None,
    ) -> str:
        terminal = self._rows[-1] if status == "FAILED" and self._rows else None
        manifest = EvidenceManifest(
            schema_version="phase13_readiness0_live_evidence_manifest_v1",
            status=status,
            request_sha256=self._inputs.request_sha256,
            authorization_sha256=self._inputs.authorization_sha256,
            f1c_registry_sha256=self._inputs.f1c_sha256,
            f1c_runtime=self._inputs.f1c_runtime,
            cases=ArtifactBinding(
                path="cases.jsonl",
                sha256=hashlib.sha256(self._cases_path.read_bytes()).hexdigest(),
            ),
            case_count=len(self._rows),
            provider_call_count=sum(row.provider_calls for row in self._rows),
            scientific_result=False,
            main_result=False,
            measured_main_a_trajectory_count=0,
            terminal_case_id=None if terminal is None else terminal.case_id,
            terminal_stage=None if terminal is None or not terminal.stages else terminal.stages[-1],
            failure_code=failure_code,
            manifest_hash="0" * 64,
        )
        manifest = manifest.model_copy(
            update={"manifest_hash": _canonical_hash(manifest.model_dump(mode="json", exclude={"manifest_hash"}))}
        )
        raw = (manifest.model_dump_json(indent=2) + "\n").encode()
        path = self._inputs.output_dir / "evidence_manifest.json"
        temporary = self._inputs.output_dir / ".evidence_manifest.tmp"
        with temporary.open("wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(self._inputs.output_dir, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return hashlib.sha256(raw).hexdigest()


def _canonical_hash(payload: dict[str, JsonValue]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


__all__ = ["EvidenceStore", "EvidenceStoreInputs"]
