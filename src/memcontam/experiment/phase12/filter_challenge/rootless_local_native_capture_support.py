from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal, TypeAlias

from memcontam.clients.base import LLMResponse
from memcontam.experiment.phase12.filter_challenge.contracts import ChallengeCandidate
from memcontam.experiment.phase12.filter_challenge.registry_calibration import ScheduledCall
from memcontam.experiment.phase12.filter_challenge.rootless_local_contract import (
    JsonValue,
    RootlessContractError,
    parse_canonical_object,
)
from memcontam.memory.checkpoint_v3 import NativeState, Phase12Checkpoint, serialize_checkpoint
from memcontam.tasks.base import TaskInstance

MODEL: Final = "gpt-4o-2024-11-20"
_ROOT: Final = Path(__file__).resolve().parents[5]
MessageRole: TypeAlias = Literal["system", "user"]
CapturedMessages: TypeAlias = tuple[tuple[MessageRole, str], ...]


@dataclass(slots=True)
class CaptureClient:
    answer: str
    predecessor_text: str | None = None
    calls: list[tuple[str, CapturedMessages]] = field(default_factory=list)

    def chat(self, messages: list[dict[str, str]], model: str, config: dict) -> LLMResponse:
        del model
        stage = config.get("method_stage")
        if not isinstance(stage, str):
            raise RootlessContractError("ROOTLESS_COMPILATION_INVALID")
        captured: CapturedMessages = tuple(
            (_message_role(message["role"]), message["content"]) for message in messages
        )
        self.calls.append((stage, captured))
        match stage:
            case "bot_problem_distill":
                content = self.predecessor_text or json.dumps(
                    {
                        "key_information": "registered probe input",
                        "restrictions": "preserve the registered task contract",
                        "distilled_task": "solve the registered probe",
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            case "bot_instantiate_solve":
                content = json.dumps(
                    {
                        "selected_structure": "registered-template",
                        "solution_trace": "apply the selected template",
                        "final_answer": f"final: {self.answer}",
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            case _:
                content = f"final: {self.answer}"
        return LLMResponse(content, {}, {}, 0)


@dataclass(frozen=True, slots=True)
class CaptureEmbedder:
    candidate_content: str

    def encode_document(self, text: str) -> list[float]:
        return [1.0, 0.0] if text == self.candidate_content else [0.0, 1.0]

    def encode_query(self, text: str) -> list[float]:
        del text
        return [1.0, 0.0]


def captured(client: CaptureClient, stage: str) -> CapturedMessages:
    matches = [messages for recorded_stage, messages in client.calls if recorded_stage == stage]
    if len(matches) != 1:
        raise RootlessContractError("ROOTLESS_COMPILATION_INVALID")
    return matches[0]


def task_fixture(call: ScheduledCall) -> tuple[TaskInstance, str]:
    manifest = _load("probe_construction_manifest_v1.json")
    probes = manifest.get("probes")
    if not isinstance(probes, dict):
        raise RootlessContractError("ROOTLESS_COMPILATION_INVALID")
    rows = probes.get(call.task)
    if not isinstance(rows, list):
        raise RootlessContractError("ROOTLESS_COMPILATION_INVALID")
    for row in rows:
        if not isinstance(row, dict) or row.get("probe_id") != call.probe_id:
            continue
        certificate = row.get("certificate")
        if not isinstance(certificate, dict):
            break
        match call.task:
            case "game24" | "math_equation_balancer":
                task_input = {
                    "numbers": certificate.get("numbers"),
                    "target": certificate.get("target"),
                }
                answer = certificate.get("expression")
            case "word_sorting":
                task_input = {"words": certificate.get("input_words")}
                words = certificate.get("correct_order")
                if not isinstance(words, list) or not all(
                    isinstance(word, str) for word in words
                ):
                    break
                answer = " ".join(word for word in words if isinstance(word, str))
        if isinstance(answer, str):
            return TaskInstance(
                sample_id=call.probe_id, task_name=call.task, input=task_input
            ), answer
        break
    raise RootlessContractError("ROOTLESS_COMPILATION_INVALID")


def checkpoint_fixture(call: ScheduledCall) -> Phase12Checkpoint:
    manifest = _load("checkpoint_manifest_v1.json")
    rows = manifest.get("checkpoints")
    if not isinstance(rows, list):
        raise RootlessContractError("ROOTLESS_COMPILATION_INVALID")
    for row in rows:
        if not isinstance(row, dict):
            continue
        state = row.get("state")
        if not isinstance(state, dict) or state.get("baseline") != call.baseline:
            continue
        entries = state.get("entries")
        if not isinstance(entries, list) or not any(
            isinstance(entry, dict) and call.task in str(entry.get("entry_id"))
            for entry in entries
        ):
            continue
        native_state = NativeState.from_mapping(state)
        if call.baseline == "full_history":
            native_state = NativeState(
                "fh_bounded",
                native_state.entries,
                native_state.native_state,
                native_state.schema_version,
            )
        return serialize_checkpoint(native_state)
    raise RootlessContractError("ROOTLESS_COMPILATION_INVALID")


def candidate_fixture(
    call: ScheduledCall, source_hash: str, checkpoint_id: str
) -> ChallengeCandidate:
    manifest = _load("candidate_triplets_v1.json")
    rows = manifest.get("renders")
    if not isinstance(rows, list):
        raise RootlessContractError("ROOTLESS_COMPILATION_INVALID")
    match call.candidate_class:
        case None | "certified_false" | "ordinary_false":
            variant = "false"
        case "correct":
            variant = "correct"
        case "irrelevant":
            variant = "irrelevant"
    for row in rows:
        if (
            not isinstance(row, dict)
            or row.get("task") != call.task
            or row.get("baseline") != call.baseline
        ):
            continue
        entry = row.get("entry")
        if not isinstance(entry, dict) or variant not in str(entry.get("render_id")):
            continue
        return ChallengeCandidate.model_validate(
            {
                "candidate_entry_id": entry.get("entry_id"),
                "candidate_native_content": entry.get("content"),
                "candidate_native_kind": entry.get("semantic_kind"),
                "baseline_family": call.baseline,
                "rag_mode": "frozen" if call.baseline == "rag_frozen" else "not_applicable",
                "source_checkpoint_id": checkpoint_id,
                "source_active_state_hash": source_hash,
                "routability": {
                    "routability": "challenge_routable_v1",
                    "challenge_suite_key": "rootless-native-capture",
                },
            }
        )
    raise RootlessContractError("ROOTLESS_COMPILATION_INVALID")


def _load(name: str) -> dict[str, JsonValue]:
    path = _ROOT / "data/phase12/filter_v5_bct_v1" / name
    return parse_canonical_object(path.read_bytes())


def _message_role(value: str) -> MessageRole:
    match value:
        case "system":
            return "system"
        case "user":
            return "user"
        case _:
            raise RootlessContractError("ROOTLESS_COMPILATION_INVALID")
