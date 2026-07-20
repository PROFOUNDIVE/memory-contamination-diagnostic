from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from memcontam.baselines.contracts import CorpusIdentity
from memcontam.logging.schema import ContaminationClass, LineageBasis, LineageStatus
from memcontam.memory.filters import FilterTelemetry, drop_known_contaminated
from memcontam.memory.stores import MemoryEntry


LOCKED_TASKS = {"game24", "math_equation_balancer", "word_sorting"}
KNOWN_BASELINES = {
    "no_memory",
    "full_history",
    "retrieval_rag",
    "reflexion_style",
    "bot_style",
    "dynamic_cheatsheet_optional",
    "dynamic_cheatsheet_rs_optional",
    "expel_optional",
}

_FORBIDDEN_ANSWER_SUBSTRINGS = frozenset(
    [
        "final:",
        "6 / (1 - 3/4)",
        "6/(1-3/4)",
        "7 * 2 + 7 + 3",
        "7*2+7+3",
        "8 / (3 - 8/3)",
        "8/(3-8/3)",
        "5 * (5 - 1/5)",
        "5*(5-1/5)",
        "2 + 5 = 7",
        "9 - 4 = 5",
        "3 * 6 = 18",
        "apple banana pear",
        "alpha bravo charlie delta",
        "ant yak zebra",
    ]
)


class CorpusValidationError(ValueError):
    pass


class CorpusManifest(BaseModel):
    manifest_id: str
    corpus_version: str
    content_hash: str


def corpus_content_hash(records: list["CorpusRecord"]) -> str:
    payload = json.dumps(
        [record.model_dump(mode="json") for record in records],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def trusted_corpus_card(
    records: list["CorpusRecord"],
    *,
    manifest: CorpusManifest | None,
    task_family: str,
    embedding_provider_identity: str,
) -> CorpusIdentity:
    if manifest is None:
        raise ValueError("corpus manifest is required")
    _ValidatedCorpus(records)
    if manifest.content_hash != corpus_content_hash(records):
        raise ValueError("manifest content_hash does not match corpus content")
    return CorpusIdentity(
        manifest_id=manifest.manifest_id,
        corpus_version=manifest.corpus_version,
        task_family=task_family,
        embedding_provider_identity=embedding_provider_identity,
    )


def _assert_no_leakage(text: str, entry_id: str) -> None:
    lowered = text.lower()
    for substring in _FORBIDDEN_ANSWER_SUBSTRINGS:
        if substring.lower() in lowered:
            raise ValueError(
                f"record {entry_id!r} contains raw evaluation answer {substring!r}"
            )


class CorpusRecord(BaseModel):
    entry_id: str
    task: str
    target_baselines: list[str] = Field(default_factory=list)
    memory_type: str
    content: str
    output_text: str | None = None
    source: str
    clean_or_contaminated: Literal["clean", "contaminated"]
    paired_clean_entry_id: str | None = None
    contamination_class: ContaminationClass | None = None
    lineage_status: LineageStatus | None = None
    lineage_basis: LineageBasis | None = None
    direct_parent_ids: list[str] = Field(default_factory=list)
    injected_root_ids: list[str] = Field(default_factory=list)

    @field_validator("task")
    @classmethod
    def _task_is_locked(cls, value: str) -> str:
        if value not in LOCKED_TASKS:
            raise ValueError(f"task must be one of {sorted(LOCKED_TASKS)}, got {value!r}")
        return value

    @field_validator("target_baselines")
    @classmethod
    def _baselines_are_known(cls, values: list[str]) -> list[str]:
        unknown = [baseline for baseline in values if baseline not in KNOWN_BASELINES]
        if unknown:
            raise ValueError(f"unknown baselines: {unknown}")
        return values

    @field_validator("source")
    @classmethod
    def _source_is_present(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("source provenance is required")
        return value

    @field_validator("content")
    @classmethod
    def _no_raw_evaluation_answers(cls, value: str, info) -> str:
        entry_id = info.data.get("entry_id", "?")
        _assert_no_leakage(value, entry_id)
        return value

    @field_validator("output_text")
    @classmethod
    def _output_text_no_raw_evaluation_answers(cls, value: str | None, info) -> str | None:
        if value is None:
            return value
        entry_id = info.data.get("entry_id", "?")
        _assert_no_leakage(value, entry_id)
        return value

    @model_validator(mode="after")
    def _dc_rs_io_pair_has_input_and_output(self) -> "CorpusRecord":
        if self.memory_type == "dc_rs_io_pair":
            if not self.content or not self.content.strip():
                raise ValueError(
                    f"record {self.entry_id!r} dc_rs_io_pair requires non-empty content"
                )
            if self.output_text is None or not self.output_text.strip():
                raise ValueError(
                    f"record {self.entry_id!r} dc_rs_io_pair requires non-empty output_text"
                )
        self._validate_phase11_seed_provenance()
        return self

    def _validate_phase11_seed_provenance(self) -> None:
        has_phase11_fields = any(
            [
                self.contamination_class is not None,
                self.lineage_status is not None,
                self.lineage_basis is not None,
                bool(self.direct_parent_ids),
                bool(self.injected_root_ids),
            ]
        )
        if not has_phase11_fields:
            return
        if self.contamination_class is None:
            raise ValueError(f"record {self.entry_id!r} requires contamination_class")
        if self.lineage_status is None:
            raise ValueError(f"record {self.entry_id!r} requires lineage_status")
        if self.lineage_basis is None:
            raise ValueError(f"record {self.entry_id!r} requires lineage_basis")
        if self.contamination_class == "clean" and self.clean_or_contaminated != "clean":
            raise ValueError(f"record {self.entry_id!r} clean class must be clean")
        if self.contamination_class != "clean" and self.clean_or_contaminated != "contaminated":
            raise ValueError(f"record {self.entry_id!r} contaminated class must be contaminated")
        if self.lineage_status == "exact" and self.lineage_basis == "signature":
            raise ValueError(f"record {self.entry_id!r} signature basis cannot be exact")
        if self.lineage_basis != "seed":
            return
        if self.contamination_class not in {"clean", "injected"}:
            raise ValueError(f"record {self.entry_id!r} seed provenance must be clean or injected")
        if self.direct_parent_ids:
            raise ValueError(f"record {self.entry_id!r} seed provenance cannot have parents")
        if self.contamination_class == "clean" and self.injected_root_ids:
            raise ValueError(f"record {self.entry_id!r} clean seed cannot have injected_root_ids")
        if self.contamination_class == "injected" and self.injected_root_ids != [self.entry_id]:
            raise ValueError(
                f"record {self.entry_id!r} injected seed requires injected_root_ids=[entry_id]"
            )


class _ValidatedCorpus:
    def __init__(self, records: list[CorpusRecord]):
        seen: dict[str, CorpusRecord] = {}
        for record in records:
            if record.entry_id in seen:
                prior = seen[record.entry_id]
                raise CorpusValidationError(
                    f"duplicate entry_id {record.entry_id!r} "
                    f"(first task={prior.task}, second task={record.task})"
                )
            seen[record.entry_id] = record

        for record in records:
            if record.clean_or_contaminated == "contaminated":
                if not record.paired_clean_entry_id:
                    raise CorpusValidationError(
                        f"corrupted record {record.entry_id!r} is missing paired_clean_entry_id"
                    )
                clean = seen.get(record.paired_clean_entry_id)
                if clean is None:
                    raise CorpusValidationError(
                        f"corrupted record {record.entry_id!r} references missing "
                        f"paired_clean_entry_id {record.paired_clean_entry_id!r}"
                    )
                if clean.clean_or_contaminated != "clean":
                    raise CorpusValidationError(
                        f"corrupted record {record.entry_id!r} pairs with non-clean record "
                        f"{record.paired_clean_entry_id!r}"
                    )
                if clean.task != record.task:
                    raise CorpusValidationError(
                        f"corrupted record {record.entry_id!r} task {record.task!r} "
                        f"does not match paired clean record task {clean.task!r}"
                    )


def load_corpus(path: Path) -> list[CorpusRecord]:
    records: list[CorpusRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CorpusValidationError(f"line {line_number}: malformed JSON") from exc
            try:
                record = CorpusRecord.model_validate(raw)
            except ValueError as exc:
                raise CorpusValidationError(
                    f"line {line_number} (entry_id={raw.get('entry_id', '?')}): {exc}"
                ) from exc
            records.append(record)

    _ValidatedCorpus(records)
    return records


def _to_memory_entry(record: CorpusRecord) -> MemoryEntry:
    metadata: dict[str, Any] = {
        "task": record.task,
        "source": record.source,
        "target_baselines": record.target_baselines,
    }
    if record.paired_clean_entry_id is not None:
        metadata["paired_clean_entry_id"] = record.paired_clean_entry_id
    if record.memory_type == "dc_rs_io_pair" and record.output_text is not None:
        metadata["output_text"] = record.output_text
    for field_name in (
        "contamination_class",
        "lineage_status",
        "lineage_basis",
        "direct_parent_ids",
        "injected_root_ids",
    ):
        if field_name not in record.model_fields_set:
            continue
        value = getattr(record, field_name)
        if value is not None:
            metadata[field_name] = value
    return MemoryEntry(
        entry_id=record.entry_id,
        content=record.content,
        memory_type=record.memory_type,
        clean_or_contaminated=record.clean_or_contaminated,
        source_trial_id=None,
        metadata=metadata,
    )


def build_arm_corpus(
    records: list[CorpusRecord],
    task: str,
    arm: Literal["clean", "contaminated", "contaminated_filter"],
) -> tuple[list[MemoryEntry], FilterTelemetry | None]:
    task_records = [record for record in records if record.task == task]
    clean_records = [record for record in task_records if record.clean_or_contaminated == "clean"]
    clean_ids = {record.entry_id for record in clean_records}

    if arm == "clean":
        selected = clean_records
    elif arm in {"contaminated", "contaminated_filter"}:
        corrupted_records = [
            record
            for record in task_records
            if record.clean_or_contaminated == "contaminated"
            and record.paired_clean_entry_id in clean_ids
        ]
        selected = clean_records + corrupted_records
    else:
        raise CorpusValidationError(f"unknown arm: {arm}")

    selected = sorted(selected, key=lambda record: record.entry_id)
    entries = [_to_memory_entry(record) for record in selected]

    if arm == "contaminated_filter":
        return drop_known_contaminated(entries)
    return entries, None
