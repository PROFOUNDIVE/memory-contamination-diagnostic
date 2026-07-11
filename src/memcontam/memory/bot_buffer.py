from __future__ import annotations

import copy
import hashlib
from typing import Any
from dataclasses import dataclass, field
from datetime import datetime, timezone

from memcontam.clients.base import LLMClient
from memcontam.memory.embeddings import EmbeddingProvider, FakeEmbeddingProvider, normalized_dot_top_k


@dataclass(frozen=True)
class BotBufferIdentity:
    run_id: str
    task_name: str
    baseline: str
    arm: str
    backbone: str


@dataclass
class ThoughtTemplate:
    entry_id: str
    content: str
    source_trial_id: str
    source_entry_ids: list[str] = field(default_factory=list)
    accepted_at: datetime | None = None
    metadata: dict = field(default_factory=dict)


class BotBufferRegistry:
    def __init__(self):
        self._buffers: dict[BotBufferIdentity, list[ThoughtTemplate]] = {}

    def insert(self, identity: BotBufferIdentity, entry: ThoughtTemplate) -> ThoughtTemplate:
        stored = copy.deepcopy(entry)
        if stored.accepted_at is None:
            stored.accepted_at = datetime.now(timezone.utc)
        self._buffers.setdefault(identity, []).append(stored)
        return stored

    def snapshot(self, identity: BotBufferIdentity) -> tuple[ThoughtTemplate, ...]:
        return tuple(self._buffers.get(identity, []))

    def clone(self, identity: BotBufferIdentity) -> list[ThoughtTemplate]:
        return copy.deepcopy(self._buffers.get(identity, []))


def propose_template(
    candidate: ThoughtTemplate,
    client: LLMClient,
    model: str,
    config: dict[str, Any] | None = None,
) -> tuple[ThoughtTemplate, str]:
    config = dict(config or {})
    messages = [
        {
            "role": "user",
            "content": "Distill this solved trial into one reusable high-level thought template.\n"
            f"{candidate.content}",
        }
    ]
    response = client.chat(
        messages,
        model=model,
        config={**config, "method_stage": "bot_thought_distill"},
    )
    distilled = response.content.strip()
    proposed = copy.deepcopy(candidate)
    proposed.content = distilled
    proposed.entry_id = f"bot_template:{_short_hash(candidate.source_trial_id + ':' + distilled)}"
    proposed.metadata = {**candidate.metadata, "distillation_stage": "bot_thought_distill"}
    return proposed, response.content


def maybe_update(
    registry: BotBufferRegistry,
    identity: BotBufferIdentity,
    candidate: ThoughtTemplate,
    retrieved_template: ThoughtTemplate | None,
    client: LLMClient,
    model: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = dict(config or {})
    if not _is_verified(config.get("verifier_result", True)):
        return _event(candidate, "rejected", retrieved_template, reject_reason="verifier_failed")

    proposed, distill_response = propose_template(candidate, client, model, config)
    raw_response = str(candidate.metadata.get("raw_response", "")).strip()
    if _malformed_template(proposed.content, raw_response):
        return _event(
            candidate,
            "rejected",
            retrieved_template,
            distilled_content=proposed.content,
            distill_response=distill_response,
            reject_reason="malformed_template",
        )

    existing = list(registry.snapshot(identity))
    if not existing:
        stored = registry.insert(identity, proposed)
        return _event(
            candidate,
            "accepted",
            retrieved_template,
            distilled_content=stored.content,
            distill_response=distill_response,
            new_entry_id=stored.entry_id,
            accept_reason="empty_buffer",
        )

    provider = _embedding_provider(config)
    top_entry, top_similarity = _top_existing_template(proposed, existing, provider)
    decision_response = client.chat(
        [
            {
                "role": "user",
                "content": "Return True only if the candidate thought template is fundamentally novel.\n"
                f"Existing: {top_entry.content}\nCandidate: {proposed.content}",
            }
        ],
        model=model,
        config={**config, "method_stage": "bot_novelty_decide"},
    ).content
    accepted = _parse_novelty_decision(decision_response)
    if accepted:
        stored = registry.insert(identity, proposed)
        return _event(
            candidate,
            "accepted",
            top_entry,
            distilled_content=stored.content,
            distill_response=distill_response,
            novelty_decision_response=decision_response,
            top_similarity=top_similarity,
            new_entry_id=stored.entry_id,
            accept_reason="novelty_accepted",
        )
    return _event(
        candidate,
        "rejected",
        top_entry,
        distilled_content=proposed.content,
        distill_response=distill_response,
        novelty_decision_response=decision_response,
        top_similarity=top_similarity,
        reject_reason="novelty_rejected",
    )


def _parse_novelty_decision(response: str) -> bool:
    lowered = response.lower()
    if "false" in lowered:
        return False
    if "true" in lowered:
        return True
    raise ValueError("novelty decision must contain True or False")


def _is_verified(verifier_result: Any) -> bool:
    if isinstance(verifier_result, bool):
        return verifier_result
    if isinstance(verifier_result, dict):
        return bool(verifier_result.get("is_correct"))
    return bool(getattr(verifier_result, "is_correct", verifier_result))


def _malformed_template(content: str, raw_response: str) -> bool:
    stripped = content.strip()
    return not stripped or stripped == raw_response or stripped.lower().startswith("final:")


def _embedding_provider(config: dict[str, Any]) -> EmbeddingProvider:
    provider = config.get("embedding_provider")
    return provider if provider is not None else FakeEmbeddingProvider()


def _top_existing_template(
    candidate: ThoughtTemplate,
    existing: list[ThoughtTemplate],
    provider: EmbeddingProvider,
) -> tuple[ThoughtTemplate, float]:
    top_id, score = normalized_dot_top_k(
        provider.encode_query(candidate.content),
        [provider.encode_document(entry.content) for entry in existing],
        [entry.entry_id for entry in existing],
        k=1,
    )[0]
    return next(entry for entry in existing if entry.entry_id == top_id), score


def _event(
    candidate: ThoughtTemplate,
    status: str,
    compared_template: ThoughtTemplate | None,
    **extra: Any,
) -> dict[str, Any]:
    event = {
        "event_type": "bot_write" if status == "accepted" else "bot_write_rejected",
        "baseline": "bot_style",
        "status": status,
        "accepted": status == "accepted",
        "parent_trial_id": candidate.source_trial_id,
        "source_trial_id": candidate.source_trial_id,
        "source_entry_ids": list(candidate.source_entry_ids),
        "candidate_entry_id": candidate.entry_id,
        "candidate_content": candidate.content,
        "top_existing_entry_id": compared_template.entry_id if compared_template else None,
        "top_existing_content": compared_template.content if compared_template else None,
        "top_similarity": None,
        "novelty_decision_response": None,
        "distilled_content": None,
        "distill_response": None,
        "new_entry_id": None,
    }
    event.update(extra)
    return event


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
