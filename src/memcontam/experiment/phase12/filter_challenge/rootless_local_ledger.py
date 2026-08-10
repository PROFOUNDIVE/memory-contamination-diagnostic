from __future__ import annotations

# allow: SIZE_OK — the closed signed ledger grammar and its recovery validator are one authority.

import hashlib
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from memcontam.experiment.phase12.filter_challenge.rootless_local_acknowledgement import (
    parse_timestamp,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_contract import (
    JsonValue,
    RootlessContractError,
    canonical_json_file,
    parse_canonical_object,
    public_key_from_seed,
    sign_object,
    verify_object_signature,
)

PROFILE: Final = "local_rootless_non_authoritative"
RESERVATION_INPUT_TOKENS: Final = 4096
RESERVATION_OUTPUT_TOKENS: Final = 640
INPUT_NANOUSD_PER_TOKEN: Final = 2500
OUTPUT_NANOUSD_PER_TOKEN: Final = 10_000
RESERVATION_NANOUSD: Final = 16_640_000
SCREENING_CAP_NANOUSD: Final = 2_000_000_000
BCT_CAP_NANOUSD: Final = 8_000_000_000
CUMULATIVE_CAP_NANOUSD: Final = 10_000_000_000
SCREENING_CALL_CAP: Final = 90
BCT_CALL_CAP: Final = 480
CUMULATIVE_CALL_CAP: Final = SCREENING_CALL_CAP + BCT_CALL_CAP
_MAX_LEDGER_ENTRIES: Final = CUMULATIVE_CALL_CAP * 2 + 2
_HEX: Final = re.compile(r"[0-9a-f]{64}\Z")
_ID: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_FILE_FLAGS: Final = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC

Stage = Literal["screening", "bct"]
NotIssuedReason = Literal[
    "DOWNSTREAM_NOT_ISSUED_AFTER_PARSE_FAILURE",
    "DOWNSTREAM_NOT_ISSUED_AFTER_PREDECESSOR_FAILURE",
    "ROOTLESS_INPUT_CAP_EXCEEDED",
]
CompileStatus = Literal["compiled", "blocked_predecessor"]
TerminalStatus = Literal[
    "completed_estimable", "not_estimable", "blocked", "review_required", "interrupted"
]


@dataclass(frozen=True, slots=True)
class LedgerReservation:
    slot_id: str
    idempotency_key: str
    compiler_sha256: str
    static_input_sha256: str
    predecessor_receipt_sha256: str | None
    request_sha256: str
    request_bytes: int
    compiled_input_tokens: int


@dataclass(frozen=True, slots=True)
class ProviderUsage:
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass(frozen=True, slots=True)
class LedgerAppend:
    record: dict[str, JsonValue]
    head: dict[str, JsonValue]
    record_sha256: str
    head_sha256: str
    record_path: Path
    head_path: Path


@dataclass(frozen=True, slots=True)
class LedgerSnapshot:
    cumulative_issued: int
    cumulative_not_issued: int
    cumulative_settled_nanousd: int
    cumulative_retained_nanousd: int
    active_unsettled_nanousd: int
    terminal: bool


def actual_cost_nanousd(usage: ProviderUsage) -> int:
    if (
        min(
            usage.input_tokens,
            usage.cached_input_tokens,
            usage.output_tokens,
            usage.total_tokens,
        )
        < 0
        or usage.cached_input_tokens > usage.input_tokens
        or usage.total_tokens != usage.input_tokens + usage.output_tokens
        or usage.input_tokens > RESERVATION_INPUT_TOKENS
        or usage.output_tokens > RESERVATION_OUTPUT_TOKENS
    ):
        raise RootlessContractError("ROOTLESS_USAGE_INVALID")
    return (
        usage.input_tokens * INPUT_NANOUSD_PER_TOKEN
        + usage.output_tokens * OUTPUT_NANOUSD_PER_TOKEN
    )


class GlobalLedger:
    """Append-only cooperative accounting under the broker's process-lifetime lock."""

    def __init__(self, root: Path, seed: bytes, attempt_id: str, stage: Stage) -> None:
        if len(seed) != 32 or _ID.fullmatch(attempt_id) is None:
            raise RootlessContractError("ROOTLESS_LEDGER_INVALID")
        self.root = root
        self.seed = seed
        self.attempt_id = attempt_id
        self.stage: Stage = stage
        self.records_root = root / "ledger" / "global" / "records"
        self.heads_root = root / "ledger" / "global" / "heads"
        self.records_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.heads_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._records, self._heads = self._recover()

    @property
    def head_sha256(self) -> str | None:
        return self._artifact_hash(self._head_paths()[len(self._heads) - 1]) if self._heads else None

    def snapshot(self) -> LedgerSnapshot:
        issued = sum(1 for record in self._records if record["record_kind"] == "reservation")
        not_issued = sum(1 for record in self._records if record["record_kind"] == "not_issued")
        settled_records = [record for record in self._records if record["record_kind"] == "settlement"]
        settled = sum((self._settlement_cost(record) for record in settled_records), start=0)
        active = self._active_unsettled_nanousd()
        terminal = self._closed_for_stage()
        return LedgerSnapshot(issued, not_issued, settled, active, active, terminal)

    def stage_counts(self) -> tuple[int, int]:
        return (
            sum(
                record["record_kind"] == "reservation" and record["stage"] == self.stage
                for record in self._records
            ),
            sum(
                record["record_kind"] == "not_issued" and record["stage"] == self.stage
                for record in self._records
            ),
        )

    def reserve(self, request: LedgerReservation, created_at: str) -> LedgerAppend:
        self._ensure_open()
        self._validate_reservation(request)
        snapshot = self.snapshot()
        stage_issued = sum(
            1
            for record in self._records
            if record["record_kind"] == "reservation" and record["stage"] == self.stage
        )
        call_cap = SCREENING_CALL_CAP if self.stage == "screening" else BCT_CALL_CAP
        stage_cap = SCREENING_CAP_NANOUSD if self.stage == "screening" else BCT_CAP_NANOUSD
        stage_accounted = self._stage_settled(self.stage) + self._active_unsettled_nanousd(
            self.stage
        )
        cumulative_accounted = snapshot.cumulative_settled_nanousd + snapshot.active_unsettled_nanousd
        if (
            stage_issued >= call_cap
            or snapshot.cumulative_issued >= CUMULATIVE_CALL_CAP
            or stage_accounted + RESERVATION_NANOUSD > stage_cap
            or cumulative_accounted + RESERVATION_NANOUSD > CUMULATIVE_CAP_NANOUSD
        ):
            raise RootlessContractError("ROOTLESS_BUDGET_CAP_EXCEEDED")
        return self._append(
            "reservation",
            created_at,
            {
                "slot_id": request.slot_id,
                "idempotency_key": request.idempotency_key,
                "compiler_sha256": request.compiler_sha256,
                "static_input_sha256": request.static_input_sha256,
                "predecessor_receipt_sha256": request.predecessor_receipt_sha256,
                "compile_status": "compiled",
                "request_sha256": request.request_sha256,
                "request_bytes": request.request_bytes,
                "compiled_input_tokens": request.compiled_input_tokens,
                "reserved_input_tokens": RESERVATION_INPUT_TOKENS,
                "reserved_output_tokens": RESERVATION_OUTPUT_TOKENS,
                "reserved_nanousd": RESERVATION_NANOUSD,
            },
        )

    def settle(
        self,
        reservation_record_sha256: str,
        archive_manifest_sha256: str,
        typed_outcome_sha256: str,
        usage: ProviderUsage,
        created_at: str,
    ) -> LedgerAppend:
        self._ensure_open()
        for value in (
            reservation_record_sha256,
            archive_manifest_sha256,
            typed_outcome_sha256,
        ):
            self._require_hash(value)
        reservation_hashes = {
            self._artifact_hash(path)
            for path, record in zip(
                self._record_paths()[: len(self._records)], self._records, strict=True
            )
            if record["record_kind"] == "reservation"
        }
        settled: set[str] = set()
        for record in self._records:
            settlement_reference = record.get("reservation_record_sha256")
            if record["record_kind"] == "settlement" and isinstance(
                settlement_reference, str
            ):
                settled.add(settlement_reference)
        if reservation_record_sha256 not in reservation_hashes or reservation_record_sha256 in settled:
            raise RootlessContractError("ROOTLESS_LEDGER_INVALID")
        cost = actual_cost_nanousd(usage)
        return self._append(
            "settlement",
            created_at,
            {
                "reservation_record_sha256": reservation_record_sha256,
                "archive_manifest_sha256": archive_manifest_sha256,
                "typed_outcome_sha256": typed_outcome_sha256,
                "usage_input_tokens": usage.input_tokens,
                "cached_input_tokens": usage.cached_input_tokens,
                "output_tokens": usage.output_tokens,
                "total_tokens": usage.total_tokens,
                "actual_nanousd": cost,
            },
        )

    def not_issued(
        self,
        request: LedgerReservation,
        reason: NotIssuedReason,
        created_at: str,
        *,
        compile_status: CompileStatus,
        include_request: bool,
    ) -> LedgerAppend:
        self._ensure_open()
        input_cap = reason == "ROOTLESS_INPUT_CAP_EXCEEDED"
        if input_cap:
            self._validate_input_cap_not_issued(request)
        else:
            self._validate_reservation(request)
        if input_cap != include_request or input_cap != (compile_status == "compiled"):
            raise RootlessContractError("ROOTLESS_LEDGER_INVALID")
        if not input_cap and request.predecessor_receipt_sha256 is None:
            raise RootlessContractError("ROOTLESS_LEDGER_INVALID")
        return self._append(
            "not_issued",
            created_at,
            {
                "slot_id": request.slot_id,
                "compiler_sha256": request.compiler_sha256,
                "static_input_sha256": request.static_input_sha256,
                "predecessor_receipt_sha256": request.predecessor_receipt_sha256,
                "compile_status": compile_status,
                "request_sha256": request.request_sha256 if include_request else None,
                "request_bytes": request.request_bytes if include_request else None,
                "compiled_input_tokens": request.compiled_input_tokens if include_request else None,
                "reserved_nanousd": 0,
                "actual_nanousd": 0,
                "reason": reason,
            },
        )

    def terminal(
        self,
        *,
        terminal_status: TerminalStatus,
        reason_code: str,
        registered_slots: int,
        created_at: str,
        terminal_scope: Literal["stage", "attempt"] = "stage",
    ) -> LedgerAppend:
        if self.snapshot().terminal:
            record_path = self._record_paths()[-1]
            head_path = self._head_paths()[-1]
            record = self._records[-1]
            if (
                record["terminal_status"] == terminal_status
                and record["reason_code"] == reason_code
                and record["registered_slots"] == registered_slots
            ):
                return self._append_value(record_path, head_path, record, self._heads[-1])
            raise RootlessContractError("ROOTLESS_LEDGER_TERMINAL")
        stage_issued = sum(
            record["record_kind"] == "reservation" and record["stage"] == self.stage
            for record in self._records
        )
        stage_not_issued = sum(
            record["record_kind"] == "not_issued" and record["stage"] == self.stage
            for record in self._records
        )
        return self._append(
            "interruption" if terminal_status == "interrupted" else "terminal",
            created_at,
            {
                "terminal_scope": terminal_scope,
                "terminal_status": terminal_status,
                "reason_code": reason_code,
                "registered_slots": registered_slots,
                "issued_slots": stage_issued,
                "not_issued_slots": stage_not_issued,
                "settled_nanousd": self._stage_settled(self.stage),
                "retained_nanousd": self._active_unsettled_nanousd(self.stage),
            },
        )

    def _append(
        self, record_kind: str, created_at: str, fields: dict[str, JsonValue]
    ) -> LedgerAppend:
        parse_timestamp(created_at)
        snapshot = self.snapshot()
        previous_record = self._artifact_hash(self._record_paths()[-1]) if self._records else None
        payload: dict[str, JsonValue] = {
            "schema_version": "rootless_ledger_record_v1",
            "profile": PROFILE,
            "kind": "ledger_record",
            "record_kind": record_kind,
            "sequence": len(self._records),
            "previous_record_sha256": previous_record,
            "attempt_id": self.attempt_id,
            "stage": self.stage,
            "created_at": created_at,
            "key_fingerprint": hashlib.sha256(public_key_from_seed(self.seed)).hexdigest(),
            **fields,
        }
        record = self._signed(payload, "ledger-record-v1")
        settlement_cost = self._settlement_cost(record) if record_kind == "settlement" else 0
        post_record_active = snapshot.active_unsettled_nanousd + (
            RESERVATION_NANOUSD
            if record_kind == "reservation"
            else -RESERVATION_NANOUSD
            if record_kind == "settlement"
            else 0
        )
        record_raw = canonical_json_file(record)
        record_hash = hashlib.sha256(record_raw).hexdigest()
        record_path = self.records_root / f"{len(self._records):06d}-{record_hash}.json"
        self._atomic_new(record_path, record_raw)
        head_payload: dict[str, JsonValue] = {
            "schema_version": "rootless_ledger_head_v1",
            "profile": PROFILE,
            "kind": "ledger_head",
            "attempt_id": self.attempt_id,
            "sequence": len(self._heads),
            "record_sha256": record_hash,
            "previous_head_sha256": self.head_sha256,
            "cumulative_issued": snapshot.cumulative_issued + (record_kind == "reservation"),
            "cumulative_not_issued": snapshot.cumulative_not_issued + (record_kind == "not_issued"),
            "cumulative_settled_nanousd": snapshot.cumulative_settled_nanousd
            + settlement_cost,
            "cumulative_retained_nanousd": post_record_active,
            "screening_settled_nanousd": self._stage_settled("screening")
            + (settlement_cost if record_kind == "settlement" and self.stage == "screening" else 0),
            "bct_settled_nanousd": self._stage_settled("bct")
            + (settlement_cost if record_kind == "settlement" and self.stage == "bct" else 0),
            "issued_at": created_at,
            "key_fingerprint": hashlib.sha256(public_key_from_seed(self.seed)).hexdigest(),
        }
        head = self._signed(head_payload, "ledger-head-v1")
        head_raw = canonical_json_file(head)
        head_hash = hashlib.sha256(head_raw).hexdigest()
        head_path = self.heads_root / f"{len(self._heads):06d}-{head_hash}.json"
        self._atomic_new(head_path, head_raw)
        self._records.append(record)
        self._heads.append(head)
        return LedgerAppend(record, head, record_hash, head_hash, record_path, head_path)

    def _recover(self) -> tuple[list[dict[str, JsonValue]], list[dict[str, JsonValue]]]:
        record_paths = self._paths(self.records_root)
        head_paths = self._paths(self.heads_root)
        records = [self._read_verified(path, "ledger-record-v1") for path in record_paths]
        heads = [self._read_verified(path, "ledger-head-v1") for path in head_paths]
        previous_record: str | None = None
        previous_head: str | None = None
        for sequence, (path, record) in enumerate(zip(record_paths, records, strict=True)):
            if (
                record.get("schema_version") != "rootless_ledger_record_v1"
                or record.get("profile") != PROFILE
                or record.get("kind") != "ledger_record"
                or record.get("record_kind")
                not in {"reservation", "settlement", "not_issued", "terminal", "interruption"}
                or record.get("sequence") != sequence
                or record.get("previous_record_sha256") != previous_record
                or record.get("attempt_id") != self.attempt_id
                or record.get("stage") not in {"screening", "bct"}
                or record.get("key_fingerprint")
                != hashlib.sha256(public_key_from_seed(self.seed)).hexdigest()
            ):
                raise RootlessContractError("ROOTLESS_LEDGER_INVALID")
            previous_record = self._artifact_hash(path)
        if len(heads) > len(records) or len(records) - len(heads) > 1:
            raise RootlessContractError("ROOTLESS_LEDGER_INVALID")
        for sequence, (path, head) in enumerate(zip(head_paths, heads, strict=True)):
            if (
                head.get("schema_version") != "rootless_ledger_head_v1"
                or head.get("profile") != PROFILE
                or head.get("kind") != "ledger_head"
                or head.get("attempt_id") != self.attempt_id
                or head.get("sequence") != sequence
                or head.get("previous_head_sha256") != previous_head
                or head.get("record_sha256") != self._artifact_hash(record_paths[sequence])
                or head.get("key_fingerprint")
                != hashlib.sha256(public_key_from_seed(self.seed)).hexdigest()
            ):
                raise RootlessContractError("ROOTLESS_LEDGER_INVALID")
            previous_head = self._artifact_hash(path)
        self._records, self._heads = records, heads
        if len(records) == len(heads) + 1:
            record = records[-1]
            self._records = records[:-1]
            rebuilt = self._rebuild_head(record, record_paths[-1])
            heads.append(rebuilt.head)
            self._records = records
        return records, heads

    def _rebuild_head(self, record: dict[str, JsonValue], record_path: Path) -> LedgerAppend:
        sequence = len(self._heads)
        snapshot = self.snapshot()
        kind = str(record["record_kind"])
        actual = self._settlement_cost(record) if kind == "settlement" else 0
        created_at = str(record["created_at"])
        payload: dict[str, JsonValue] = {
            "schema_version": "rootless_ledger_head_v1",
            "profile": PROFILE,
            "kind": "ledger_head",
            "attempt_id": self.attempt_id,
            "sequence": sequence,
            "record_sha256": self._artifact_hash(record_path),
            "previous_head_sha256": self.head_sha256,
            "cumulative_issued": snapshot.cumulative_issued + (kind == "reservation"),
            "cumulative_not_issued": snapshot.cumulative_not_issued + (kind == "not_issued"),
            "cumulative_settled_nanousd": snapshot.cumulative_settled_nanousd + actual,
            "cumulative_retained_nanousd": snapshot.active_unsettled_nanousd
            + (
                RESERVATION_NANOUSD
                if kind == "reservation"
                else -RESERVATION_NANOUSD
                if kind == "settlement"
                else 0
            ),
            "screening_settled_nanousd": self._stage_settled("screening")
            + (actual if kind == "settlement" and record["stage"] == "screening" else 0),
            "bct_settled_nanousd": self._stage_settled("bct")
            + (actual if kind == "settlement" and record["stage"] == "bct" else 0),
            "issued_at": created_at,
            "key_fingerprint": hashlib.sha256(public_key_from_seed(self.seed)).hexdigest(),
        }
        head = self._signed(payload, "ledger-head-v1")
        raw = canonical_json_file(head)
        digest = hashlib.sha256(raw).hexdigest()
        path = self.heads_root / f"{sequence:06d}-{digest}.json"
        self._atomic_new(path, raw)
        return LedgerAppend(record, head, self._artifact_hash(record_path), digest, record_path, path)

    def _stage_settled(self, stage: Stage) -> int:
        return sum(
            self._settlement_cost(record)
            for record in self._records
            if record["record_kind"] == "settlement" and record["stage"] == stage
        )

    def _active_unsettled_nanousd(self, stage: Stage | None = None) -> int:
        settled_reservations = {
            value
            for record in self._records
            if record["record_kind"] == "settlement"
            and isinstance((value := record.get("reservation_record_sha256")), str)
        }
        active_reservations = sum(
            1
            for path, record in zip(
                self._record_paths()[: len(self._records)], self._records, strict=True
            )
            if record["record_kind"] == "reservation"
            and (stage is None or record["stage"] == stage)
            and self._artifact_hash(path) not in settled_reservations
        )
        return RESERVATION_NANOUSD * active_reservations

    @staticmethod
    def _settlement_cost(record: dict[str, JsonValue]) -> int:
        value = record.get("actual_nanousd")
        if not isinstance(value, int) or isinstance(value, bool):
            raise RootlessContractError("ROOTLESS_LEDGER_INVALID")
        return value

    def _validate_reservation(self, request: LedgerReservation) -> None:
        self._validate_request_identity(request)
        if request.request_bytes > 262_144:
            raise RootlessContractError("ROOTLESS_LEDGER_INVALID")

    def _validate_input_cap_not_issued(self, request: LedgerReservation) -> None:
        self._validate_request_identity(request)

    def _validate_request_identity(self, request: LedgerReservation) -> None:
        if (
            _ID.fullmatch(request.slot_id) is None
            or _ID.fullmatch(request.idempotency_key) is None
            or request.request_bytes < 0
            or request.compiled_input_tokens < 0
        ):
            raise RootlessContractError("ROOTLESS_LEDGER_INVALID")
        for value in (
            request.compiler_sha256,
            request.static_input_sha256,
            request.request_sha256,
        ):
            self._require_hash(value)
        if request.predecessor_receipt_sha256 is not None:
            self._require_hash(request.predecessor_receipt_sha256)
        if any(
            record.get("slot_id") == request.slot_id
            or record.get("idempotency_key") == request.idempotency_key
            for record in self._records
        ):
            raise RootlessContractError("ROOTLESS_LEDGER_DUPLICATE")

    def _ensure_open(self) -> None:
        if self.snapshot().terminal:
            raise RootlessContractError("ROOTLESS_LEDGER_TERMINAL")

    def _closed_for_stage(self) -> bool:
        if not self._records or self._records[-1]["record_kind"] not in {"terminal", "interruption"}:
            return False
        terminal = self._records[-1]
        return not (
            self.stage == "bct"
            and terminal.get("stage") == "screening"
            and terminal.get("terminal_status") == "completed_estimable"
            and terminal.get("reason_code") == "SCREENING_ESTIMABLE"
        )

    def _signed(self, payload: dict[str, JsonValue], domain: str) -> dict[str, JsonValue]:
        result = dict(payload)
        result["signature"] = sign_object(self.seed, domain, payload)
        return result

    def _read_verified(self, path: Path, domain: str) -> dict[str, JsonValue]:
        raw = self._read_regular(path)
        value = parse_canonical_object(raw)
        signature = value.get("signature")
        if not isinstance(signature, str):
            raise RootlessContractError("ROOTLESS_LEDGER_INVALID")
        unsigned = dict(value)
        del unsigned["signature"]
        try:
            verify_object_signature(public_key_from_seed(self.seed), domain, unsigned, signature)
        except RootlessContractError as error:
            raise RootlessContractError("ROOTLESS_LEDGER_INVALID") from error
        return value

    @staticmethod
    def _paths(root: Path) -> list[Path]:
        paths = sorted(root.iterdir(), key=lambda path: path.name.encode())
        if len(paths) > _MAX_LEDGER_ENTRIES:
            raise RootlessContractError("ROOTLESS_LEDGER_INVALID")
        for sequence, path in enumerate(paths):
            if not path.name.startswith(f"{sequence:06d}-"):
                raise RootlessContractError("ROOTLESS_LEDGER_INVALID")
        return paths

    def _record_paths(self) -> list[Path]:
        return self._paths(self.records_root)

    def _head_paths(self) -> list[Path]:
        return self._paths(self.heads_root)

    @staticmethod
    def _artifact_hash(path: Path) -> str:
        return hashlib.sha256(GlobalLedger._read_regular(path)).hexdigest()

    @staticmethod
    def _read_regular(path: Path) -> bytes:
        try:
            descriptor = os.open(path, _FILE_FLAGS)
        except OSError as error:
            raise RootlessContractError("ROOTLESS_LEDGER_INVALID") from error
        try:
            info = os.fstat(descriptor)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid()
                or info.st_nlink != 1
                or stat.S_IMODE(info.st_mode) != 0o600
            ):
                raise RootlessContractError("ROOTLESS_LEDGER_INVALID")
            chunks: list[bytes] = []
            while chunk := os.read(descriptor, 1_048_576):
                chunks.append(chunk)
            return b"".join(chunks)
        finally:
            os.close(descriptor)

    @staticmethod
    def _atomic_new(path: Path, raw: bytes) -> None:
        temporary = path.with_name(f".{path.name}.tmp")
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
        )
        try:
            offset = 0
            while offset < len(raw):
                offset += os.write(descriptor, raw[offset:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.rename(temporary, path)
        parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)

    @staticmethod
    def _require_hash(value: str) -> None:
        if _HEX.fullmatch(value) is None:
            raise RootlessContractError("ROOTLESS_LEDGER_INVALID")

    def _append_value(
        self,
        record_path: Path,
        head_path: Path,
        record: dict[str, JsonValue],
        head: dict[str, JsonValue],
    ) -> LedgerAppend:
        return LedgerAppend(
            record,
            head,
            self._artifact_hash(record_path),
            self._artifact_hash(head_path),
            record_path,
            head_path,
        )


__all__ = (
    "BCT_CAP_NANOUSD",
    "CUMULATIVE_CAP_NANOUSD",
    "GlobalLedger",
    "LedgerAppend",
    "LedgerReservation",
    "LedgerSnapshot",
    "ProviderUsage",
    "RESERVATION_NANOUSD",
    "SCREENING_CAP_NANOUSD",
    "actual_cost_nanousd",
)
