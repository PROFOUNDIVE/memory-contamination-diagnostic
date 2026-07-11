from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal

from memcontam.memory.bot_buffer import BotBufferIdentity, BotBufferRegistry, ThoughtTemplate


WARMUP_METADATA = {"phase": "warmup", "exclude_from_aggregate": True}


@dataclass(frozen=True)
class WarmupSnapshot:
    entries: tuple[ThoughtTemplate, ...]
    metadata: dict[str, Any]


class RunState:
    def __init__(
        self,
        run_id: str,
        config_hash: str,
        mode: Literal["fresh", "resume"] = "fresh",
        evaluation_sample_ids: list[str] | tuple[str, ...] = (),
        state_config_hash: str | None = None,
    ):
        if mode not in {"fresh", "resume"}:
            raise ValueError("mode must be fresh or resume")
        if mode == "resume" and state_config_hash != config_hash:
            raise ValueError("state config hash mismatch")

        self.run_id = run_id
        self.config_hash = config_hash
        self.mode = mode
        self.sample_order = tuple(evaluation_sample_ids)
        self._evaluation_sample_ids = set(evaluation_sample_ids)
        self._registry = BotBufferRegistry()
        self._warmup_sample_ids: dict[BotBufferIdentity, list[str]] = {}
        self._accepted_template_ids: dict[BotBufferIdentity, list[str]] = {}
        self._clean_snapshots: dict[tuple[str, str, str, str], WarmupSnapshot] = {}
        self._arm_metadata: dict[BotBufferIdentity, dict[str, Any]] = {}

    def register_warmup_result(
        self, identity: BotBufferIdentity, update_event: dict[str, Any]
    ) -> dict[str, Any]:
        if identity.run_id != self.run_id:
            raise ValueError("identity run_id must match RunState run_id")

        sample_id = str(update_event.get("sample_id", ""))
        if not sample_id:
            raise ValueError("warm-up update_event requires sample_id")
        if sample_id in self._evaluation_sample_ids:
            raise ValueError(f"evaluation sample cannot be used for warm-up: {sample_id}")

        self._warmup_sample_ids.setdefault(identity, []).append(sample_id)
        if update_event.get("status") != "accepted":
            return {**WARMUP_METADATA, "sample_id": sample_id, "accepted": False}

        entry = ThoughtTemplate(
            entry_id=str(update_event["new_entry_id"]),
            content=str(update_event.get("distilled_content") or update_event.get("candidate_content") or ""),
            source_trial_id=str(update_event.get("source_trial_id") or update_event.get("parent_trial_id") or sample_id),
            source_entry_ids=list(update_event.get("source_entry_ids", [])),
            metadata={**WARMUP_METADATA, "warmup_sample_id": sample_id},
        )
        stored = self._registry.insert(identity, entry)
        self._accepted_template_ids.setdefault(identity, []).append(stored.entry_id)
        return {
            **WARMUP_METADATA,
            "sample_id": sample_id,
            "accepted": True,
            "new_entry_id": stored.entry_id,
        }

    def snapshot_clean_warmup(self, identity: BotBufferIdentity) -> WarmupSnapshot:
        entries = self._registry.snapshot(identity)
        metadata = {
            "warmup_sample_ids": list(self._warmup_sample_ids.get(identity, [])),
            "accepted_template_ids": list(self._accepted_template_ids.get(identity, [])),
            "snapshot_hash": _snapshot_hash(entries),
        }
        snapshot = WarmupSnapshot(entries=entries, metadata=metadata)
        self._clean_snapshots[_snapshot_key(identity)] = snapshot
        return snapshot

    def clone_for_arm(
        self, identity: BotBufferIdentity, injection_entries: list[ThoughtTemplate]
    ) -> list[ThoughtTemplate]:
        key = _snapshot_key(identity)
        snapshot = self._clean_snapshots.get(key)
        if snapshot is None:
            clean = BotBufferIdentity(
                identity.run_id, identity.task_name, identity.baseline, "clean", identity.backbone
            )
            snapshot = self.snapshot_clean_warmup(clean)

        clone = copy.deepcopy(list(snapshot.entries))
        clone.extend(copy.deepcopy(injection_entries))
        self._arm_metadata[identity] = {
            **snapshot.metadata,
            "injection_ids": [entry.entry_id for entry in injection_entries],
        }
        return clone

    def arm_metadata(self, identity: BotBufferIdentity) -> dict[str, Any]:
        return copy.deepcopy(self._arm_metadata.get(identity, {}))


def _snapshot_key(identity: BotBufferIdentity) -> tuple[str, str, str, str]:
    return identity.run_id, identity.task_name, identity.baseline, identity.backbone


def _snapshot_hash(entries: tuple[ThoughtTemplate, ...]) -> str:
    payload = [
        {
            "entry_id": entry.entry_id,
            "content": entry.content,
            "source_trial_id": entry.source_trial_id,
            "source_entry_ids": entry.source_entry_ids,
            "metadata": entry.metadata,
        }
        for entry in entries
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
