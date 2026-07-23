from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import TYPE_CHECKING, Any, Mapping

from memcontam.experiment.phase12.contracts import PrefixTemplateSpec, canonical_json_hash
from memcontam.memory.checkpoint_v3 import Phase12Checkpoint, Phase12CheckpointIdentity

if TYPE_CHECKING:
    from memcontam.experiment.phase12.prefix_runner import PrefixRunResult


class PrefixReuseError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class PrefixLawIdentity:
    prefix_template_key: str
    baseline_condition_id: str
    seed: int
    model_snapshot: str
    evidence_layer: str
    task_family: str
    sensitivity_cell_hash: str
    prompt_version: str
    tool_contract_hash: str
    corpus_version: str
    capacity_contract_id: str
    task_sequence_hash: str

    @classmethod
    def from_template(
        cls,
        template: PrefixTemplateSpec,
        *,
        seed: int,
        task_sequence: tuple[Mapping[str, Any], ...],
    ) -> PrefixLawIdentity:
        return cls(
            prefix_template_key=template.prefix_template_key,
            baseline_condition_id=template.baseline_condition_id,
            seed=seed,
            model_snapshot=template.model_snapshot,
            evidence_layer=template.evidence_layer,
            task_family=template.task_family,
            sensitivity_cell_hash=canonical_json_hash(dict(template.sensitivity_cell_ref)),
            prompt_version=template.prompt_version,
            tool_contract_hash=template.tool_contract_hash,
            corpus_version=template.corpus_version,
            capacity_contract_id=template.capacity_contract_id,
            task_sequence_hash=canonical_json_hash(task_sequence),
        )

    @property
    def fingerprint(self) -> str:
        return canonical_json_hash(self.to_mapping())

    @property
    def scope_key(self) -> tuple[str, str, int]:
        return (self.prefix_template_key, self.baseline_condition_id, self.seed)

    def to_mapping(self) -> dict[str, str | int]:
        return {
            "baseline_condition_id": self.baseline_condition_id,
            "capacity_contract_id": self.capacity_contract_id,
            "corpus_version": self.corpus_version,
            "evidence_layer": self.evidence_layer,
            "model_snapshot": self.model_snapshot,
            "prefix_template_key": self.prefix_template_key,
            "prompt_version": self.prompt_version,
            "seed": self.seed,
            "sensitivity_cell_hash": self.sensitivity_cell_hash,
            "task_family": self.task_family,
            "task_sequence_hash": self.task_sequence_hash,
            "tool_contract_hash": self.tool_contract_hash,
        }


@dataclass(frozen=True)
class StoredPrefix:
    identity: PrefixLawIdentity
    checkpoint: Phase12Checkpoint
    result: PrefixRunResult


class CheckpointStore:
    def __init__(self) -> None:
        self._lock = RLock()
        self._by_prefix: dict[str, StoredPrefix] = {}
        self._by_scope: dict[tuple[str, str, int], PrefixLawIdentity] = {}
        self._by_checkpoint: dict[Phase12CheckpointIdentity, Phase12Checkpoint] = {}

    def reuse(self, identity: PrefixLawIdentity) -> PrefixRunResult | None:
        with self._lock:
            known_identity = self._by_scope.get(identity.scope_key)
            if known_identity is not None and known_identity != identity:
                raise PrefixReuseError("PREFIX_IDENTITY_DRIFT")
            stored = self._by_prefix.get(identity.fingerprint)
            return None if stored is None else stored.result

    def save(self, result: PrefixRunResult) -> Phase12CheckpointIdentity:
        identity = result.prefix_identity
        checkpoint = result.checkpoint
        with self._lock:
            known_identity = self._by_scope.get(identity.scope_key)
            if known_identity is not None and known_identity != identity:
                raise PrefixReuseError("PREFIX_IDENTITY_DRIFT")
            stored = self._by_prefix.get(identity.fingerprint)
            if stored is not None:
                if stored.checkpoint != checkpoint:
                    raise PrefixReuseError("PREFIX_CHECKPOINT_CONFLICT")
                return stored.checkpoint.identity
            self._by_scope[identity.scope_key] = identity
            self._by_prefix[identity.fingerprint] = StoredPrefix(identity, checkpoint, result)
            self._by_checkpoint[checkpoint.identity] = checkpoint
            return checkpoint.identity

    def load(self, identity: Phase12CheckpointIdentity) -> Phase12Checkpoint:
        with self._lock:
            try:
                return self._by_checkpoint[identity]
            except KeyError as error:
                raise PrefixReuseError("UNKNOWN_PREFIX_CHECKPOINT") from error


_DEFAULT_STORE = CheckpointStore()


def default_checkpoint_store() -> CheckpointStore:
    return _DEFAULT_STORE


def save_checkpoint(result: PrefixRunResult) -> Phase12CheckpointIdentity:
    return _DEFAULT_STORE.save(result)


def load_checkpoint(identity: Phase12CheckpointIdentity) -> Phase12Checkpoint:
    return _DEFAULT_STORE.load(identity)
