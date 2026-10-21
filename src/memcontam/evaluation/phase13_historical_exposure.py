"""Source-bound descriptive reconstruction; never execute a historical driver."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Final, TypedDict

from pydantic import JsonValue

from memcontam.evaluation.phase13_observability import reconstruct_phase13_trial
from memcontam.evaluation.phase13_observability_lineage import validate_evidence_joins, writer_parents
from memcontam.evaluation.phase13_observability_models import Phase13TrialEvidence
from memcontam.logging.schema import MethodCall
from memcontam.memory.cards_v3 import canonical_content_hash

STAGES: Final = ("stored", "retrieved", "curator", "derivation", "strategy_visible", "actual_answer_context", "registered_z", "operational_use")


class HistoricalIntegrityError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class Binding(TypedDict):
    path: str
    sha256: str


class Observation(TypedDict):
    status: str
    value: bool | None
    reason: str
    missing_artifact: str | None
    eligible: bool


def observation(value: bool | None, reason: str, missing: str | None = None, *, eligible: bool = True) -> Observation:
    return {"status": "unknown" if value is None else "known", "value": value, "reason": reason,
            "missing_artifact": missing, "eligible": eligible}


def checked(binding: Binding) -> bytes:
    path = Path(binding["path"])
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != binding["sha256"]:
        raise HistoricalIntegrityError("SOURCE_HASH_MISMATCH:" + str(path))
    return raw


def validate_output_path(output: Path, inputs: tuple[Path, ...], verify: bool) -> None:
    if any(part.is_symlink() for part in (output, *output.parents)):
        raise HistoricalIntegrityError("OUTPUT_ALIAS")
    resolved = output.resolve()
    if any(resolved == root.resolve() or resolved.is_relative_to(root.resolve()) or root.resolve().is_relative_to(resolved) for root in inputs):
        raise HistoricalIntegrityError("OUTPUT_INPUT_COLLISION")
    if not verify and output.exists():
        raise FileExistsError(output)
    if verify and not output.is_dir():
        raise FileNotFoundError(output)


def validate_population(inventory) -> None:
    rows, cells = inventory["rows"], inventory["cells"]
    keys = [row["population_key"] for row in rows]
    if len(keys) != len(set(keys)) or len(keys) != inventory["expected_scientific_slots"]:
        raise HistoricalIntegrityError("DUPLICATE_OR_TRUNCATED_POPULATION")
    cell_keys = [cell["cell_key"] for cell in cells]
    if len(cell_keys) != len(set(cell_keys)) or set(cell_keys) != {row["cell_key"] for row in rows}:
        raise HistoricalIntegrityError("CELL_IDENTITY_MISMATCH")
    for cell in cells:
        selected = sorted((row for row in rows if row["cell_key"] == cell["cell_key"]), key=lambda row: row["position"])
        if len(selected) != cell["expected"] or [row["sample_id"] for row in selected] != cell["registered_suffix_order"] or [row["position"] for row in selected] != list(range(1, cell["expected"] + 1)):
            raise HistoricalIntegrityError("REGISTERED_SUFFIX_MISMATCH:" + cell["cell_key"])


def _attribution(entry_ids: list[str], evidence: Phase13TrialEvidence, targets: set[str]) -> bool | None:
    nodes = {node.entry_id: node for node in evidence.lineage}
    parents = writer_parents(evidence)

    def visit(entry: str, seen: frozenset[str]) -> bool | None:
        if entry in seen:
            raise HistoricalIntegrityError("LINEAGE_CYCLE")
        if entry in targets:
            return True
        node = nodes.get(entry)
        if node is None or node.lineage_status != "exact":
            return None
        references = parents.get(entry, (*node.direct_parent_ids, *((node.version_predecessor_id,) if node.version_predecessor_id else ())))
        values = [visit(parent, seen | {entry}) for parent in references]
        return True if True in values else None if None in values else False

    values = [visit(entry, frozenset()) for entry in entry_ids]
    return True if True in values else None if None in values else False


def reconstruct_stages(evidence: Phase13TrialEvidence, calls: tuple[MethodCall, ...], injected_ids: tuple[str, ...]) -> dict[str, Observation]:
    validate_evidence_joins(evidence)
    call_ids = [call.call_id for call in calls]
    if None in call_ids or len(call_ids) != len(set(call_ids)):
        raise HistoricalIntegrityError("DUPLICATE_OR_MISSING_CALL_ID")
    for call in calls:
        digest = hashlib.sha256(json.dumps(call.messages, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
        if call.provider_request_contract is None or digest != call.provider_request_contract.get("input_sha256"):
            raise HistoricalIntegrityError("REQUEST_HASH_MISMATCH:" + str(call.call_id))
        for span in call.source_spans:
            if span.message_index >= len(call.messages):
                raise HistoricalIntegrityError("SOURCE_SPAN_CALL_JOIN")
            content = call.messages[span.message_index]["content"]
            if span.start < 0 or span.end > len(content) or hashlib.sha256(content[span.start:span.end].encode()).hexdigest() != span.rendered_hash:
                raise HistoricalIntegrityError("SOURCE_SPAN_HASH_MISMATCH")
    targets = set(injected_ids)
    registered = set(evidence.target_set.target_entry_ids)
    if evidence.trial.execution_key.arm == "contam" and targets != registered:
        raise HistoricalIntegrityError("REGISTERED_TARGET_MISMATCH")
    stages = {stage: observation(None, "NOT_APPLICABLE", eligible=False) for stage in STAGES}
    stages["operational_use"] = observation(None, "NOT_REGISTERED", "separately registered operational attribution", eligible=False)
    stages["stored"] = observation(bool(targets.intersection(evidence.memory_before_ids)), "RECORDED_NATIVE_IDENTITY")
    stages["retrieved"] = observation(bool(targets.intersection(entry for event in evidence.retrievals for entry in event.retrieved_entry_ids)), "RECORDED_RETRIEVAL") if evidence.retrievals else observation(None, "MISSING_RETRIEVAL_EVENT", evidence.trial_id + ":retrieval")
    answer = next((call for call in calls if call.call_id == evidence.target_set.answer_call_id), None)
    if evidence.baseline == "dc_rs":
        curator = next((call for call in calls if call.stage == "dc_rs_synthesize"), None)
        stages["curator"] = observation(bool(targets.intersection(span.entry_id for span in curator.source_spans)), "ACTUAL_CURATOR_REQUEST") if curator else observation(None, "MISSING_CURATOR_REQUEST", evidence.trial_id + ":dc_rs_synthesize")
        final_ids = evidence.context.final_entry_ids if evidence.context else []
        derived = _attribution(final_ids, evidence, targets)
        stages["derivation"] = observation(derived, "EXPLICIT_PARENTAGE" if derived is not None else "UNAVAILABLE_STRATEGY_PARENTAGE", evidence.trial_id + ":lineage" if derived is None else None)
        stages["strategy_visible"] = observation(bool(answer and answer.source_spans), "ACTUAL_STRATEGY_REQUEST_SPAN") if answer else observation(None, "MISSING_ACTUAL_ANSWER_CALL", evidence.target_set.answer_call_id)
    if answer is None:
        stages["actual_answer_context"] = observation(None, "MISSING_ACTUAL_ANSWER_CALL", evidence.target_set.answer_call_id)
    else:
        final_ids = [span.entry_id for span in answer.source_spans]
        if evidence.context is None or set(final_ids) != set(evidence.context.final_entry_ids):
            raise HistoricalIntegrityError("ACTUAL_REQUEST_CONTEXT_JOIN")
        if any(span not in answer.source_spans for span in evidence.target_set.answer_call_spans):
            raise HistoricalIntegrityError("TARGET_ANSWER_SPAN_JOIN")
        included = _attribution(final_ids, evidence, targets)
        stages["actual_answer_context"] = observation(included, "ACTUAL_REQUEST_AND_EXACT_LINEAGE" if included is not None else "UNAVAILABLE_STRATEGY_PARENTAGE", evidence.trial_id + ":lineage" if included is None else None)
    if evidence.trial.execution_key.arm == "contam":
        stages["registered_z"] = stages["actual_answer_context"].copy()
        if answer is not None and evidence.verified_outcome is not None and stages["registered_z"]["value"] is not None:
            reconstructed = reconstruct_phase13_trial(evidence)
            if reconstructed.theory_exposure.value != stages["registered_z"]["value"]:
                raise HistoricalIntegrityError("REGISTERED_Z_RECONSTRUCTION_MISMATCH")
    return stages


def join_original(slot, archive, trial) -> tuple[Phase13TrialEvidence, tuple[MethodCall, ...], tuple[str, ...]]:
    """Authenticate standalone/native archive identity before deriving any stage."""
    matches = [record for record in archive["records"] if record["evidence"]["trial_id"] == slot["trial_id"]]
    if len(matches) != 1:
        raise HistoricalIntegrityError("ARCHIVE_TRIAL_CARDINALITY")
    record = matches[0]
    evidence = Phase13TrialEvidence.model_validate_json(json.dumps(record["evidence"]))
    if (evidence.task, evidence.baseline, evidence.trial.execution_key.arm, evidence.order_key) != (slot["task"], slot["baseline"], slot["arm"], slot["order_key"]) or record["task_instance"]["sample_id"] != slot["sample_id"]:
        raise HistoricalIntegrityError("TRIAL_CELL_JOIN")
    if trial["outcome"]["method_calls"] != record["method_calls"] or trial["context"] != record["evidence"]["context"] or [trial["retrieval"]] != record["evidence"]["retrievals"] or trial["outcome"]["answer_call_id"] != slot["answer_call_id"] or evidence.target_set.answer_call_id != slot["answer_call_id"]:
        raise HistoricalIntegrityError("NATIVE_ARCHIVE_JOIN")
    for state, ids in ((trial["state_before"], evidence.memory_before_ids), (trial["state_after"], evidence.memory_after_ids)):
        if tuple(entry["entry_id"] for entry in state["entries"]) != ids:
            raise HistoricalIntegrityError("NATIVE_STATE_JOIN")
        for entry in state["entries"]:
            if canonical_content_hash(entry["content"]) != entry["content_hash"]:
                raise HistoricalIntegrityError("NATIVE_ENTRY_HASH")
    calls = tuple(MethodCall.model_validate(call) for call in record["method_calls"])
    for call in calls:
        if call.provider_authority_contract is None or call.provider_authority_contract.get("execution_envelope_sha256") != slot["registration"]["sha256"]:
            raise HistoricalIntegrityError("CALL_REGISTRATION_JOIN")
    injected = tuple(entry["entry_id"] for entry in trial["state_before"]["entries"] if entry.get("render_id") is not None)
    return evidence, calls, injected


def summarize(rows) -> list[dict[str, JsonValue]]:
    groups = {}
    for row in rows:
        for stage, value in row["stages"].items():
            key = tuple(row[name] for name in ("task", "baseline", "arm", "cohort", "segment")) + (stage,)
            counts = groups.setdefault(key, {"expected": 0, "completed": 0, "known": 0, "unknown": 0, "true": 0, "false": 0, "eligible": 0})
            counts["expected"] += 1
            counts["completed"] += row["observation_status"] == "completed"
            counts["eligible"] += value["eligible"]
            counts["known"] += value["status"] == "known"
            counts["unknown"] += value["status"] == "unknown"
            counts["true"] += value["value"] is True
            counts["false"] += value["value"] is False
    result = []
    for key, counts in sorted(groups.items()):
        result.append(dict(zip(("task", "baseline", "arm", "cohort", "segment", "stage"), key, strict=True), **counts,
                           true_over_known=counts["true"] / counts["known"] if counts["known"] else None,
                           known_over_expected=counts["known"] / counts["expected"] if counts["eligible"] else None))
    return result
