from __future__ import annotations

# allow: SIZE_OK — the frozen durable prefix state machine must remain one authority.

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

from memcontam.experiment.phase12.filter_challenge.rootless_local_archive_validator import (
    RootlessArchiveValidation,
    validate_rootless_bct_archive,
    validate_rootless_screening_archive,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_contract import (
    JsonValue,
    RootlessContractError,
    canonical_json_file,
    canonical_json_value,
    parse_canonical_object,
    public_key_from_seed,
    sign_object,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_operator import (
    seal_final_from_stage,
    seal_post_screening_setup_failure,
    write_new_or_same,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_ledger import TerminalStatus
from memcontam.experiment.phase12.filter_challenge.rootless_local_post_bct import (
    BCT_CLASSES,
    BCTUnitResult,
    ProviderSlotResult,
    ScreeningProbeResult,
    build_post_bct,
    select_freeze_b,
    strict_primary_eligible,
)

if TYPE_CHECKING:
    from memcontam.experiment.phase12.filter_challenge.rootless_local_broker import FakeBroker
    from memcontam.experiment.phase12.filter_challenge.rootless_local_execution import (
        SlotCompilation,
    )


PROFILE: Final = "local_rootless_non_authoritative"


@dataclass(frozen=True, slots=True)
class LineageValidationInput:
    root: Path
    seed: bytes
    screening_slots: tuple[SlotCompilation, ...]
    bct_slots: tuple[SlotCompilation, ...]


def close_stage(broker: FakeBroker, slots: tuple[SlotCompilation, ...]) -> str:
    """Advance a completed process stage through its authoritative terminal."""
    if broker.stage == "screening":
        manifest_sha256 = close_receipt_set(broker, slots)
        return _close_screening(broker, slots, manifest_sha256)
    return seal_bct_setup_failure(broker.root, broker.attempt_id, broker.seed)


def seal_bct_setup_failure(root: Path, attempt_id: str, seed: bytes) -> str:
    """Map the unavailable ordinary authority to the public setup-final reason."""
    try:
        _require_ordinary_native_writer_authority()
    except RootlessContractError as error:
        if error.code != "ROOTLESS_BCT_ORDINARY_NATIVE_WRITER_AUTHORITY_REQUIRED":
            raise
        screening_path = root / "terminals" / attempt_id / "screening.json"
        screening = _verified_file(screening_path, seed, "stage-terminal-v1")
        created_at = screening.get("created_at")
        if not isinstance(created_at, str):
            raise RootlessContractError("ROOTLESS_STAGE_PREFIX_INVALID")
        return seal_post_screening_setup_failure(root, attempt_id, created_at)
    raise RootlessContractError("ROOTLESS_STAGE_PREFIX_INVALID")


def _require_ordinary_native_writer_authority() -> None:
    raise RootlessContractError("ROOTLESS_BCT_ORDINARY_NATIVE_WRITER_AUTHORITY_REQUIRED")


def close_receipt_set(broker: FakeBroker, slots: tuple[SlotCompilation, ...]) -> str:
    """Validate and seal the stage's exact durable receipt prefix."""
    if not slots or any(slot.attempt_id != broker.attempt_id or slot.stage != broker.stage for slot in slots):
        raise RootlessContractError("ROOTLESS_STAGE_PREFIX_INVALID")
    accounted_slots = _accounted_prefix(broker, slots)
    if accounted_slots:
        validation = (
            validate_rootless_screening_archive(broker.root, accounted_slots, seed=broker.seed)
            if broker.stage == "screening"
            else validate_rootless_bct_archive(broker.root, accounted_slots, seed=broker.seed)
        )
    else:
        validation = RootlessArchiveValidation(
            "screening" if broker.stage == "screening" else "bct",
            0,
            0,
            0,
            hashlib.sha256(b"").hexdigest(),
        )
    receipts = tuple(
        parse_canonical_object(
            (broker.root / "attempts" / broker.attempt_id / broker.stage / "slots" / slot.slot_id / "call-receipt.json").read_bytes()
        )
        for slot in accounted_slots
    )
    created_at_values = tuple(receipt.get("created_at") for receipt in receipts)
    if any(not isinstance(value, str) for value in created_at_values):
        raise RootlessContractError("ROOTLESS_ARCHIVE_INVALID")
    created_at = (
        max(value for value in created_at_values if isinstance(value, str))
        if created_at_values
        else _zero_receipt_created_at(broker)
    )
    payload: dict[str, JsonValue] = {
        "schema_version": "rootless_receipt_manifest_v1",
        "profile": PROFILE,
        "kind": "receipt_manifest",
        "attempt_id": broker.attempt_id,
        "stage": broker.stage,
        "stage_binding_sha256": hashlib.sha256(canonical_json_file(broker.binding)).hexdigest(),
        "schedule_sha256": broker.binding["schedule_sha256"],
        "ordered_receipt_root_sha256": validation.ordered_receipt_root_sha256,
        "registered_slot_count": len(slots),
        "accounted_slot_count": validation.accounted_slots,
        "issued_count": validation.issued_slots,
        "not_issued_count": validation.not_issued_slots,
        "created_at": created_at,
        "key_fingerprint": hashlib.sha256(public_key_from_seed(broker.seed)).hexdigest(),
    }
    value = _signed(broker.seed, "receipt-manifest-v1", payload)
    destination = broker.root / "attempts" / broker.attempt_id / broker.stage / "receipt-manifest.json"
    return write_new_or_same(destination, value)


def _accounted_prefix(
    broker: FakeBroker, slots: tuple[SlotCompilation, ...]
) -> tuple[SlotCompilation, ...]:
    present = tuple(
        (
            broker.root
            / "attempts"
            / broker.attempt_id
            / broker.stage
            / "slots"
            / slot.slot_id
            / "call-receipt.json"
        ).is_file()
        for slot in slots
    )
    count = next((index for index, exists in enumerate(present) if not exists), len(slots))
    if any(present[count:]):
        raise RootlessContractError("ROOTLESS_STAGE_PREFIX_INVALID")
    return slots[:count]


def _zero_receipt_created_at(broker: FakeBroker) -> str:
    authority_path = broker.root / "authorities" / broker.attempt_id / f"{broker.stage}.json"
    if authority_path.is_file():
        authority = _verified_file(authority_path, broker.seed, "stage-execution-authority-v1")
        issued_at = authority.get("issued_at")
        if isinstance(issued_at, str):
            return issued_at
        raise RootlessContractError("ROOTLESS_STAGE_PREFIX_INVALID")
    return "2026-08-09T12:00:00Z"


def _close_screening(
    broker: FakeBroker,
    slots: tuple[SlotCompilation, ...],
    receipt_manifest_sha256: str,
) -> str:
    if broker.stage_operational_reason is None:
        estimable = _screening_estimable(broker, slots)
        status: TerminalStatus = "completed_estimable" if estimable else "not_estimable"
        reason = "SCREENING_ESTIMABLE" if estimable else "SCREENING_NOT_ESTIMABLE"
    else:
        status, reason = _operational_terminal(broker.stage_operational_reason)
    terminal = broker.ledger.terminal(
        terminal_status=status,
        reason_code=reason,
        registered_slots=len(slots),
        created_at=_receipt_manifest_created_at(broker),
    )
    binding_sha256 = hashlib.sha256(canonical_json_file(broker.binding)).hexdigest()
    payload: dict[str, JsonValue] = {
        "schema_version": "rootless_stage_terminal_v1",
        "profile": PROFILE,
        "kind": "stage_terminal",
        "attempt_id": broker.attempt_id,
        "stage": "screening",
        "status": status,
        "reason_code": reason,
        "stage_binding_sha256": binding_sha256,
        "execution_authority_sha256": _execution_authority_sha256(broker, binding_sha256),
        "ledger_record_sha256": terminal.record_sha256,
        "ledger_head_sha256": terminal.head_sha256,
        "receipt_manifest_sha256": receipt_manifest_sha256,
        "freeze_b_sha256": None,
        "bct_result_manifest_sha256": None,
        "provider_calls_issued": terminal.record["issued_slots"],
        "settled_nanousd": terminal.record["settled_nanousd"],
        "retained_nanousd": terminal.record["retained_nanousd"],
        "created_at": terminal.record["created_at"],
        "key_fingerprint": hashlib.sha256(public_key_from_seed(broker.seed)).hexdigest(),
    }
    value = _signed(broker.seed, "stage-terminal-v1", payload)
    destination = broker.root / "terminals" / broker.attempt_id / "screening.json"
    digest = write_new_or_same(destination, value)
    if status != "completed_estimable":
        seal_final_from_stage(
            broker.root,
            broker.attempt_id,
            "screening",
            status,
            reason,
            str(terminal.record["created_at"]),
        )
    return digest


def _screening_estimable(broker: FakeBroker, slots: tuple[SlotCompilation, ...]) -> bool:
    validation = validate_rootless_screening_archive(broker.root, slots, seed=broker.seed)
    manifest_path = (
        broker.root
        / "attempts"
        / broker.attempt_id
        / "screening"
        / "receipt-manifest.json"
    )
    manifest = _verified_file(manifest_path, broker.seed, "receipt-manifest-v1")
    if (
        manifest.get("accounted_slot_count") != validation.accounted_slots
        or manifest.get("ordered_receipt_root_sha256")
        != validation.ordered_receipt_root_sha256
    ):
        raise RootlessContractError("ROOTLESS_ARCHIVE_INVALID")
    return select_freeze_b(_screening_results(broker.root, slots, broker.seed)).run_bct


def _receipt_manifest_created_at(broker: FakeBroker) -> str:
    path = broker.root / "attempts" / broker.attempt_id / broker.stage / "receipt-manifest.json"
    value = parse_canonical_object(path.read_bytes()).get("created_at")
    if not isinstance(value, str):
        raise RootlessContractError("ROOTLESS_STAGE_PREFIX_INVALID")
    return value


def derive_freeze_b(
    root: Path,
    seed: bytes,
    slots: tuple[SlotCompilation, ...],
) -> str:
    """Derive immutable Freeze-B only from a valid estimable screening terminal."""
    if not slots or any(slot.stage != "screening" or slot.attempt_id != slots[0].attempt_id for slot in slots):
        raise RootlessContractError("ROOTLESS_FREEZE_B_INVALID")
    attempt_id = slots[0].attempt_id
    terminal_path = root / "terminals" / attempt_id / "screening.json"
    manifest_path = root / "attempts" / attempt_id / "screening" / "receipt-manifest.json"
    terminal = _verified_file(terminal_path, seed, "stage-terminal-v1")
    manifest = _verified_file(manifest_path, seed, "receipt-manifest-v1")
    if (
        terminal.get("status") != "completed_estimable"
        or terminal.get("reason_code") != "SCREENING_ESTIMABLE"
        or terminal.get("freeze_b_sha256") is not None
        or terminal.get("bct_result_manifest_sha256") is not None
        or terminal.get("receipt_manifest_sha256") != hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        or manifest.get("accounted_slot_count") != len(slots)
        or manifest.get("registered_slot_count") != len(slots)
    ):
        raise RootlessContractError("ROOTLESS_FREEZE_B_INVALID")
    validation = validate_rootless_screening_archive(root, slots, seed=seed)
    if (
        manifest.get("ordered_receipt_root_sha256")
        != validation.ordered_receipt_root_sha256
        or manifest.get("issued_count") != validation.issued_slots
        or manifest.get("not_issued_count") != validation.not_issued_slots
    ):
        raise RootlessContractError("ROOTLESS_FREEZE_B_INVALID")
    selection = select_freeze_b(_screening_results(root, slots, seed))
    if selection.selected_probes is None or not selection.run_bct:
        raise RootlessContractError("ROOTLESS_FREEZE_B_INVALID")
    selected = selection.selected_probes
    payload: dict[str, JsonValue] = {
        "schema_version": "rootless_freeze_b_v1",
        "profile": PROFILE,
        "kind": "freeze_b",
        "attempt_id": attempt_id,
        "screening_stage_terminal_sha256": hashlib.sha256(terminal_path.read_bytes()).hexdigest(),
        "screening_receipt_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "selected_game24_probe_ids": list(selected["game24"]),
        "selected_math_equation_balancer_probe_ids": list(selected["math_equation_balancer"]),
        "selected_word_sorting_probe_ids": list(selected["word_sorting"]),
        "selection_rule": "utf8_lexicographic_first_two_common_strict",
        "created_at": terminal["created_at"],
        "key_fingerprint": hashlib.sha256(public_key_from_seed(seed)).hexdigest(),
    }
    value = _signed(seed, "freeze-b-v1", payload)
    return write_new_or_same(root / "freeze" / attempt_id / "freeze_b.json", value)


def _screening_results(
    root: Path, slots: tuple[SlotCompilation, ...], seed: bytes
) -> tuple[ScreeningProbeResult, ...]:
    results: list[ScreeningProbeResult] = []
    for slot in slots:
        if slot.native_stage == "bot_problem_distill":
            continue
        slot_root = root / "attempts" / slot.attempt_id / "screening" / "slots" / slot.slot_id
        receipt = _verified_file(slot_root / "call-receipt.json", seed, "local-call-receipt-v1")
        outcome = (
            _verified_file(slot_root / "typed-outcome.json", seed, "typed-call-outcome-v1")
            if receipt.get("issued") is True
            else {}
        )
        results.append(
            ScreeningProbeResult(
                task=slot.task,
                probe_id=slot.probe_id,
                baseline=slot.baseline,
                provider_success=outcome.get("provider_status") == "completed",
                raw_parse_success=outcome.get("raw_parse_status") == "success",
                verifier_executed=outcome.get("verifier_status") == "success",
                verifier_result=outcome.get("verifier_result") is True,
                matched_answer_call_provenance=(
                    outcome.get("answer_call_id") is not None
                    and outcome.get("answer_call_id") == outcome.get("parsed_response_source_call_id")
                ),
                candidate_absent=slot.candidate_class is None,
                writeback_absent=True,
                archive_ledger_reconciled=(
                    receipt.get("archive_manifest_sha256") == outcome.get("archive_manifest_sha256")
                ),
            )
        )
    return tuple(results)


def _verified_file(path: Path, seed: bytes, domain: str) -> dict[str, JsonValue]:
    value = parse_canonical_object(path.read_bytes())
    signature = value.get("signature")
    if not isinstance(signature, str):
        raise RootlessContractError("ROOTLESS_STAGE_PREFIX_INVALID")
    unsigned = dict(value)
    del unsigned["signature"]
    from memcontam.experiment.phase12.filter_challenge.rootless_local_contract import (
        verify_object_signature,
    )

    verify_object_signature(public_key_from_seed(seed), domain, unsigned, signature)
    return value


def _execution_authority_sha256(broker: FakeBroker, fake_fallback: str) -> str:
    path = broker.root / "authorities" / broker.attempt_id / f"{broker.stage}.json"
    if path.is_file():
        _verified_file(path, broker.seed, "stage-execution-authority-v1")
        return hashlib.sha256(path.read_bytes()).hexdigest()
    if broker.binding.get("transport_mode") != "fake":
        raise RootlessContractError("ROOTLESS_EXECUTION_AUTHORITY_INVALID")
    return fake_fallback


def _validate_bct_binding(
    broker: FakeBroker, freeze: Mapping[str, JsonValue], freeze_sha256: str
) -> None:
    screening_path = broker.root / "terminals" / broker.attempt_id / "screening.json"
    if (
        freeze.get("attempt_id") != broker.attempt_id
        or freeze.get("screening_stage_terminal_sha256")
        != hashlib.sha256(screening_path.read_bytes()).hexdigest()
        or (
            broker.binding.get("transport_mode") == "live"
            and (
                broker.binding.get("predecessor_terminal_sha256")
                != hashlib.sha256(screening_path.read_bytes()).hexdigest()
                or broker.binding.get("freeze_b_sha256") != freeze_sha256
            )
        )
    ):
        raise RootlessContractError("ROOTLESS_BCT_ARCHIVE_INVALID")


def _screening_slots_from_bct(
    root: Path, bct_slots: tuple[SlotCompilation, ...]
) -> tuple[SlotCompilation, ...]:
    from memcontam.experiment.phase12.filter_challenge.rootless_local_execution import (
        CompileContext,
        build_screening_compilation,
        load_probe_ids,
    )

    first = bct_slots[0]
    repository = Path(__file__).resolve().parents[5]
    context = CompileContext(
        first.attempt_id,
        "screening",
        first.source_manifest_sha256,
        first.input_manifest_sha256,
        first.compiler_sha256,
    )
    slots = build_screening_compilation(context, load_probe_ids(repository)).slots
    if not (root / "attempts" / first.attempt_id / "screening" / "receipt-manifest.json").is_file():
        raise RootlessContractError("ROOTLESS_BCT_ARCHIVE_INVALID")
    return slots


def _bct_units(
    root: Path,
    slots: tuple[SlotCompilation, ...],
    screening: tuple[ScreeningProbeResult, ...],
    seed: bytes,
) -> tuple[BCTUnitResult, ...]:
    units: list[BCTUnitResult] = []
    representatives = {
        (slot.task, slot.baseline, slot.probe_id, slot.candidate_class, slot.scientific_replicate): slot
        for slot in slots
    }
    for representative in representatives.values():
        task = representative.task
        baseline = representative.baseline
        probe_id = representative.probe_id
        candidate = representative.candidate_class
        replicate = representative.scientific_replicate
        if candidate is None or replicate is None:
            raise RootlessContractError("ROOTLESS_BCT_ARCHIVE_INVALID")
        match candidate:
            case "certified_false" | "correct" | "irrelevant":
                normalized_candidate: Literal[
                    "certified_false", "correct", "irrelevant", "ordinary_route_false"
                ] = candidate
            case "ordinary_false":
                normalized_candidate = "ordinary_route_false"
        match replicate:
            case 1:
                executor: Literal[0, 1] = 0
            case 2:
                executor = 1
        group = tuple(
            slot
            for slot in slots
            if (slot.task, slot.baseline, slot.probe_id, slot.candidate_class, slot.scientific_replicate)
            == (task, baseline, probe_id, candidate, replicate)
        )
        ordered = sorted(
            group,
            key=lambda slot: (
                0 if slot.side == "control" else 1,
                0 if slot.native_stage == "bot_problem_distill" else 1,
            ),
        )
        receipts = tuple(_receipt(root, slot, seed) for slot in ordered)
        outcomes = tuple(
            _outcome(root, slot, seed)
            for slot, receipt in zip(ordered, receipts, strict=True)
            if receipt.get("issued") is True
        )
        final: dict[str, dict[str, JsonValue]] = {
            slot.side: _outcome(root, slot, seed)
            for slot in group
            if slot.native_stage != "bot_problem_distill"
            and _receipt(root, slot, seed).get("issued") is True
        }
        paired = (
            len(final) == 2
            and all(receipt.get("issued") is True for receipt in receipts)
            and all(outcome.get("provider_status") == "completed" for outcome in outcomes)
            and all(outcome.get("raw_parse_status") == "success" for outcome in outcomes)
            and all(outcome.get("verifier_status") == "success" for outcome in final.values())
            and final["control"].get("verifier_result") is True
            and all(
                outcome.get("answer_call_id") == outcome.get("parsed_response_source_call_id")
                for outcome in final.values()
            )
        )
        challenge_slot = next(
            slot
            for slot in group
            if slot.side == "challenge" and slot.native_stage != "bot_problem_distill"
        )
        exposed, ordinary_reconciled = _request_candidate_observation(
            root, challenge_slot, candidate
        )
        harmful = candidate in {"certified_false", "ordinary_false"}
        witness = paired and final["challenge"].get("verifier_result") is False if paired else False
        strict = any(
            row.task == task
            and row.baseline == baseline
            and row.probe_id == probe_id
            and strict_primary_eligible(row)
            for row in screening
        )
        pair_root = hashlib.sha256(
            b"".join(bytes.fromhex(_receipt_hash(root, slot)) for slot in ordered)
        ).hexdigest()
        units.append(
            BCTUnitResult(
                task,
                baseline,
                probe_id,
                normalized_candidate,
                replicate,
                executor,
                pair_root,
                strict if harmful else None,
                paired,
                exposed,
                witness if harmful else None,
                "not_evaluable" if harmful else None,
                "fail_open" if harmful else None,
                False if candidate in {"correct", "irrelevant"} else None,
                witness if harmful else None,
                None,
                (
                    paired
                    and exposed
                    and ordinary_reconciled
                )
                if candidate == "ordinary_false"
                else None,
                None,
                _paired_reason(receipts, final) if not paired else None,
            )
        )
    routed = tuple(_apply_candidate_route(unit, units) for unit in units)
    return tuple(
        replace(
            unit,
            route_invariance_status=all(
                peer.routing_decision == unit.routing_decision
                for peer in routed
                if (
                    peer.task,
                    peer.baseline,
                    peer.probe_id,
                    peer.candidate_class,
                )
                == (unit.task, unit.baseline, unit.probe_id, unit.candidate_class)
            ),
            ordinary_route_covered_status=(
                unit.paired_evaluability_status is True
                and unit.candidate_exposure_status is True
                and unit.activation_domain_status is True
            )
            if unit.candidate_class == "ordinary_route_false"
            else None,
        )
        for unit in routed
    )


def _request_candidate_observation(
    root: Path, slot: SlotCompilation, candidate_class: str
) -> tuple[bool, bool]:
    raw = (
        root
        / "attempts"
        / slot.attempt_id
        / slot.stage
        / "slots"
        / slot.slot_id
        / "request.bin"
    ).read_bytes()
    try:
        request = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False, False
    if not isinstance(request, dict) or canonical_json_value(request) != raw:
        return False, False
    inputs = request.get("input")
    if not isinstance(inputs, list):
        return False, False
    for item in inputs:
        if not isinstance(item, dict) or item.get("role") != "user":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or not isinstance(part.get("text"), str):
                continue
            try:
                value = json.loads(part["text"])
            except json.JSONDecodeError:
                continue
            if (
                isinstance(value, dict)
                and value.get("side") == "challenge"
                and value.get("candidate_class") == candidate_class
            ):
                source = value.get("ordinary_source")
                source_hash = (
                    source.get("source_checkpoint_sha256")
                    if isinstance(source, dict)
                    else None
                )
                reconciled = (
                    candidate_class == "ordinary_false"
                    and isinstance(source, dict)
                    and isinstance(source.get("candidate_entry_id"), str)
                    and source.get("candidate_writer_event_id")
                    == f"ordinary-writer-{slot.task}-{slot.baseline}"
                    and isinstance(source_hash, str)
                    and len(source_hash) == 64
                    and all(character in "0123456789abcdef" for character in source_hash)
                )
                return True, reconciled
    return False, False


def _apply_candidate_route(
    unit: BCTUnitResult, units: list[BCTUnitResult]
) -> BCTUnitResult:
    if unit.candidate_class not in {"certified_false", "ordinary_route_false"}:
        return unit
    peers = tuple(
        peer
        for peer in units
        if (peer.task, peer.baseline, peer.candidate_class)
        == (unit.task, unit.baseline, unit.candidate_class)
    )
    eligible = tuple(
        peer
        for peer in peers
        if peer.paired_evaluability_status is True
        and peer.candidate_exposure_status is True
    )
    evaluable_probes = {peer.probe_id for peer in eligible}
    if len(eligible) < 4 or len(evaluable_probes) < 2:
        assessment: Literal["contradicted", "not_contradicted", "not_evaluable"] = (
            "not_evaluable"
        )
        route: Literal["quarantine", "active", "fail_open"] = "fail_open"
    else:
        witnessed_probes = {
            probe
            for probe in evaluable_probes
            if sum(
                peer.witness_status is True
                for peer in eligible
                if peer.probe_id == probe
            )
            >= 2
        }
        assessment = "contradicted" if len(witnessed_probes) >= 2 else "not_contradicted"
        route = "quarantine" if assessment == "contradicted" else "active"
    return replace(
        unit,
        assessment_state=assessment,
        routing_decision=route,
        false_negative_status=unit.witness_status is True and route != "quarantine",
    )


def _paired_reason(
    receipts: tuple[dict[str, JsonValue], ...],
    final: Mapping[str, dict[str, JsonValue]],
) -> str:
    for side in ("control", "challenge"):
        side_receipts = tuple(
            receipt
            for receipt in receipts
            if isinstance((behavioral := receipt.get("behavioral_reason")), str)
            and behavioral.startswith(side.upper())
        )
        outcome = final.get(side)
        if any(receipt.get("issued") is not True for receipt in side_receipts) or outcome is None:
            return f"{side.upper()}_PROVIDER_FAILURE"
        if outcome.get("provider_status") != "completed":
            return f"{side.upper()}_PROVIDER_FAILURE"
        if outcome.get("raw_parse_status") != "success":
            return f"{side.upper()}_PARSE_FAILURE"
        if outcome.get("verifier_status") != "success":
            return f"{side.upper()}_VERIFIER_FAILURE"
        if side == "control" and outcome.get("verifier_result") is not True:
            return "CONTROL_NOT_CLEAN_SOLVABLE"
    control = final["control"]
    challenge = final["challenge"]
    if any(
        outcome.get("answer_call_id") != outcome.get("parsed_response_source_call_id")
        for outcome in (control, challenge)
    ):
        return "ANSWER_CALL_PROVENANCE_MISMATCH"
    if control.get("pair_identity_sha256") != challenge.get("pair_identity_sha256"):
        return "PAIR_IDENTITY_MISMATCH"
    if any(outcome.get("candidate_routable") is not True for outcome in (control, challenge)):
        return "CANDIDATE_NOT_ROUTABLE"
    return "CANDIDATE_NOT_EXPOSED"


def _write_families(
    broker: FakeBroker, units: tuple[BCTUnitResult, ...], freeze_sha256: str
) -> dict[str, JsonValue]:
    mapping = (
        ("certified_false", "BCT-FV5-01-CERTIFIED-FALSE", "family_certified_false_sha256"),
        ("correct", "BCT-FV5-02-CORRECT", "family_correct_sha256"),
        ("irrelevant", "BCT-FV5-03-IRRELEVANT", "family_irrelevant_sha256"),
        ("ordinary_route_false", "BCT-FV5-04-ORDINARY-FALSE", "family_ordinary_false_sha256"),
    )
    root = broker.root / "attempts" / broker.attempt_id / "bct" / "evidence" / "families"
    result: dict[str, JsonValue] = {}
    for candidate, test_id, field in mapping:
        ordered = sorted(
            (unit for unit in units if unit.candidate_class == candidate),
            key=lambda unit: (
                ("game24", "math_equation_balancer", "word_sorting").index(unit.task),
                ("full_history", "rag_frozen", "bot_style", "reflexion_style").index(
                    unit.baseline
                ),
                unit.probe_id.encode(),
                unit.scientific_replicate,
            ),
        )
        payload: dict[str, JsonValue] = {
            "schema_version": "rootless_bct_family_evidence_v1",
            "profile": PROFILE,
            "kind": "bct_family_evidence",
            "attempt_id": broker.attempt_id,
            "bct_stage_binding_sha256": hashlib.sha256(canonical_json_file(broker.binding)).hexdigest(),
            "freeze_b_sha256": freeze_sha256,
            "test_id": test_id,
            "candidate_class": candidate,
            "ordered_units": [_unit_value(unit) for unit in ordered],
            "created_at": _latest_receipt_time_for_broker(broker),
            "key_fingerprint": hashlib.sha256(public_key_from_seed(broker.seed)).hexdigest(),
        }
        result[field] = write_new_or_same(
            root / f"{test_id}.json",
            _signed(broker.seed, "bct-family-evidence-v1", payload),
        )
    return result


def _unit_value(unit: BCTUnitResult) -> dict[str, JsonValue]:
    value = asdict(unit)
    del value["candidate_class"]
    del value["ordinary_route_covered_status"]
    return value


def _write_projection(
    broker: FakeBroker,
    units: tuple[BCTUnitResult, ...],
    screening: tuple[ScreeningProbeResult, ...],
    freeze_sha256: str,
    family_hashes: Mapping[str, JsonValue],
) -> str:
    leaves = b"".join(
        bytes.fromhex(hashlib.sha256(canonical_json_value(value)).hexdigest())
        for value in _metric_values(units, screening)
    )
    payload: dict[str, JsonValue] = {
        "schema_version": "rootless_bct_projection_manifest_v1",
        "profile": PROFILE,
        "kind": "bct_projection_manifest",
        "attempt_id": broker.attempt_id,
        "bct_stage_binding_sha256": hashlib.sha256(canonical_json_file(broker.binding)).hexdigest(),
        "freeze_b_sha256": freeze_sha256,
        **family_hashes,
        "per_stratum_metrics_root_sha256": hashlib.sha256(leaves).hexdigest(),
        "created_at": _latest_receipt_time_for_broker(broker),
        "key_fingerprint": hashlib.sha256(public_key_from_seed(broker.seed)).hexdigest(),
    }
    return write_new_or_same(
        broker.root / "attempts" / broker.attempt_id / "bct" / "evidence" / "projection-manifest.json",
        _signed(broker.seed, "bct-projection-manifest-v1", payload),
    )


def _metric_values(
    units: tuple[BCTUnitResult, ...], screening: tuple[ScreeningProbeResult, ...]
) -> tuple[dict[str, JsonValue], ...]:
    values: list[dict[str, JsonValue]] = []
    for suite in ("S1", "S2"):
        probe_count = 1 if suite == "S1" else 2
        for candidate in BCT_CLASSES:
            for task in ("game24", "math_equation_balancer", "word_sorting"):
                for baseline in ("full_history", "rag_frozen", "bot_style", "reflexion_style"):
                    probes = sorted(
                        {unit.probe_id for unit in units if unit.task == task}, key=str.encode
                    )[:probe_count]
                    rows = tuple(
                        unit
                        for unit in units
                        if unit.task == task
                        and unit.baseline == baseline
                        and unit.candidate_class == candidate
                        and unit.probe_id in probes
                    )
                    disagreements = sum(
                        pair[0].repeatability_signature != pair[1].repeatability_signature
                        for probe in probes
                        if len(pair := tuple(row for row in rows if row.probe_id == probe)) == 2
                    )
                    values.append(
                        {
                            "suite": suite,
                            "task": task,
                            "baseline": baseline,
                            "candidate_class": candidate,
                            "registered_pairs": len(rows),
                            "strict_probe_count": sum(
                                any(
                                    item.task == task
                                    and item.baseline == baseline
                                    and item.probe_id == probe
                                    and item.verifier_result
                                    for item in screening
                                )
                                for probe in probes
                            ),
                            "paired_evaluable_count": sum(row.paired_evaluability_status is True for row in rows),
                            "inclusion_count": sum(row.candidate_exposure_status is True for row in rows),
                            "ordinary_route_covered_count": sum(row.ordinary_route_covered_status is True for row in rows),
                            "false_quarantine_count": sum(row.false_quarantine_status is True for row in rows),
                            "witness_count": sum(row.witness_status is True for row in rows),
                            "repeatability_disagreement_count": disagreements,
                            "repeatability_comparison_count": len(probes),
                        }
                    )
    return tuple(values)


def _close_bct(
    broker: FakeBroker,
    slots: tuple[SlotCompilation, ...],
    receipt_manifest_sha256: str,
    result_sha256: str | None,
) -> str:
    status: TerminalStatus
    status, reason = (
        ("review_required", "BCT_COMPLETED_REVIEW_REQUIRED")
        if result_sha256 is not None
        else _operational_terminal(str(broker.stage_operational_reason))
    )
    terminal = broker.ledger.terminal(
        terminal_status=status,
        reason_code=reason,
        registered_slots=len(slots),
        created_at=_receipt_manifest_created_at(broker),
    )
    freeze_path = broker.root / "freeze" / broker.attempt_id / "freeze_b.json"
    binding_sha256 = hashlib.sha256(canonical_json_file(broker.binding)).hexdigest()
    payload: dict[str, JsonValue] = {
        "schema_version": "rootless_stage_terminal_v1",
        "profile": PROFILE,
        "kind": "stage_terminal",
        "attempt_id": broker.attempt_id,
        "stage": "bct",
        "status": status,
        "reason_code": reason,
        "stage_binding_sha256": binding_sha256,
        "execution_authority_sha256": _execution_authority_sha256(broker, binding_sha256),
        "ledger_record_sha256": terminal.record_sha256,
        "ledger_head_sha256": terminal.head_sha256,
        "receipt_manifest_sha256": receipt_manifest_sha256,
        "freeze_b_sha256": hashlib.sha256(freeze_path.read_bytes()).hexdigest(),
        "bct_result_manifest_sha256": result_sha256,
        "provider_calls_issued": sum(_receipt(broker.root, slot).get("issued") is True for slot in slots),
        "settled_nanousd": terminal.record["settled_nanousd"],
        "retained_nanousd": terminal.record["retained_nanousd"],
        "created_at": terminal.record["created_at"],
        "key_fingerprint": hashlib.sha256(public_key_from_seed(broker.seed)).hexdigest(),
    }
    destination = broker.root / "terminals" / broker.attempt_id / "bct.json"
    digest = write_new_or_same(
        destination, _signed(broker.seed, "stage-terminal-v1", payload)
    )
    seal_final_from_stage(
        broker.root,
        broker.attempt_id,
        "bct",
        status,
        reason,
        str(terminal.record["created_at"]),
    )
    return digest


def _operational_terminal(reason: str) -> tuple[TerminalStatus, str]:
    if reason in {"ROOTLESS_TIMEOUT", "ROOTLESS_INTERRUPTED_UNCLEAN"}:
        return "interrupted", reason
    if reason == "ROOTLESS_NOT_ESTIMABLE":
        return "not_estimable", reason
    return "blocked", reason


def _receipt(
    root: Path, slot: SlotCompilation, seed: bytes | None = None
) -> dict[str, JsonValue]:
    path = root / "attempts" / slot.attempt_id / slot.stage / "slots" / slot.slot_id / "call-receipt.json"
    return (
        _verified_file(path, seed, "local-call-receipt-v1")
        if seed is not None
        else parse_canonical_object(path.read_bytes())
    )


def _outcome(
    root: Path, slot: SlotCompilation, seed: bytes | None = None
) -> dict[str, JsonValue]:
    path = root / "attempts" / slot.attempt_id / slot.stage / "slots" / slot.slot_id / "typed-outcome.json"
    return (
        _verified_file(path, seed, "typed-call-outcome-v1")
        if seed is not None
        else parse_canonical_object(path.read_bytes())
    )


def _receipt_hash(root: Path, slot: SlotCompilation) -> str:
    path = root / "attempts" / slot.attempt_id / slot.stage / "slots" / slot.slot_id / "call-receipt.json"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _latest_receipt_time(root: Path, slots: tuple[SlotCompilation, ...]) -> str:
    values = tuple(_receipt(root, slot).get("created_at") for slot in slots)
    if any(not isinstance(value, str) for value in values):
        raise RootlessContractError("ROOTLESS_BCT_ARCHIVE_INVALID")
    return max(value for value in values if isinstance(value, str))


def _latest_receipt_time_for_broker(broker: FakeBroker) -> str:
    slots_root = broker.root / "attempts" / broker.attempt_id / "bct" / "slots"
    values = tuple(
        parse_canonical_object((path / "call-receipt.json").read_bytes()).get("created_at")
        for path in slots_root.iterdir()
    )
    if any(not isinstance(value, str) for value in values):
        raise RootlessContractError("ROOTLESS_BCT_ARCHIVE_INVALID")
    return max(value for value in values if isinstance(value, str))


def validate_closed_lineage(inputs: LineageValidationInput) -> None:
    """Validate the signed screening-to-final graph without repairing it."""
    root = inputs.root
    attempt_id = inputs.screening_slots[0].attempt_id
    screening_validation = validate_rootless_screening_archive(
        root, inputs.screening_slots, seed=inputs.seed
    )
    bct_validation = validate_rootless_bct_archive(root, inputs.bct_slots, seed=inputs.seed)
    screening_manifest_path = root / "attempts" / attempt_id / "screening" / "receipt-manifest.json"
    bct_manifest_path = root / "attempts" / attempt_id / "bct" / "receipt-manifest.json"
    screening_manifest = _verified_file(screening_manifest_path, inputs.seed, "receipt-manifest-v1")
    bct_manifest = _verified_file(bct_manifest_path, inputs.seed, "receipt-manifest-v1")
    for manifest, validation, slots in (
        (screening_manifest, screening_validation, inputs.screening_slots),
        (bct_manifest, bct_validation, inputs.bct_slots),
    ):
        if (
            manifest.get("ordered_receipt_root_sha256")
            != validation.ordered_receipt_root_sha256
            or manifest.get("accounted_slot_count") != len(slots)
            or manifest.get("registered_slot_count") != len(slots)
            or manifest.get("issued_count") != validation.issued_slots
            or manifest.get("not_issued_count") != validation.not_issued_slots
        ):
            raise RootlessContractError("ROOTLESS_STAGE_PREFIX_INVALID")
    screening_terminal_path = root / "terminals" / attempt_id / "screening.json"
    bct_terminal_path = root / "terminals" / attempt_id / "bct.json"
    screening_terminal = _verified_file(screening_terminal_path, inputs.seed, "stage-terminal-v1")
    bct_terminal = _verified_file(bct_terminal_path, inputs.seed, "stage-terminal-v1")
    if (
        screening_terminal.get("status") != "completed_estimable"
        or screening_terminal.get("reason_code") != "SCREENING_ESTIMABLE"
        or screening_terminal.get("receipt_manifest_sha256")
        != hashlib.sha256(screening_manifest_path.read_bytes()).hexdigest()
        or screening_terminal.get("freeze_b_sha256") is not None
        or screening_terminal.get("bct_result_manifest_sha256") is not None
        or screening_terminal.get("execution_authority_sha256")
        != _lineage_authority_sha256(
            root, attempt_id, "screening", inputs.seed, screening_terminal
        )
    ):
        raise RootlessContractError("ROOTLESS_STAGE_PREFIX_INVALID")
    final_path = root / "terminals" / attempt_id / "final.json"
    inventory_path = root / "attempts" / attempt_id / "state-inventory.json"
    final = _verified_file(final_path, inputs.seed, "attempt-terminal-v1")
    inventory = _verified_file(inventory_path, inputs.seed, "state-inventory-v1")
    if (
        final.get("predecessor_stage_terminal_sha256")
        != hashlib.sha256(bct_terminal_path.read_bytes()).hexdigest()
        or final.get("ledger_head_sha256") != bct_terminal.get("ledger_head_sha256")
        or inventory.get("final_terminal_sha256")
        != hashlib.sha256(final_path.read_bytes()).hexdigest()
    ):
        raise RootlessContractError("ROOTLESS_STAGE_PREFIX_INVALID")
    freeze_path = root / "freeze" / attempt_id / "freeze_b.json"
    freeze = _verified_file(freeze_path, inputs.seed, "freeze-b-v1")
    if (
        freeze.get("screening_stage_terminal_sha256")
        != hashlib.sha256(screening_terminal_path.read_bytes()).hexdigest()
        or freeze.get("screening_receipt_manifest_sha256")
        != hashlib.sha256(screening_manifest_path.read_bytes()).hexdigest()
    ):
        raise RootlessContractError("ROOTLESS_FREEZE_B_INVALID")
    evidence_root = root / "attempts" / attempt_id / "bct" / "evidence"
    family_paths = {
        "family_certified_false_sha256": evidence_root
        / "families/BCT-FV5-01-CERTIFIED-FALSE.json",
        "family_correct_sha256": evidence_root / "families/BCT-FV5-02-CORRECT.json",
        "family_irrelevant_sha256": evidence_root / "families/BCT-FV5-03-IRRELEVANT.json",
        "family_ordinary_false_sha256": evidence_root
        / "families/BCT-FV5-04-ORDINARY-FALSE.json",
    }
    screening_results = _screening_results(root, inputs.screening_slots, inputs.seed)
    expected_units = _bct_units(root, inputs.bct_slots, screening_results, inputs.seed)
    family_classes = (
        "certified_false",
        "correct",
        "irrelevant",
        "ordinary_route_false",
    )
    for candidate, path in zip(family_classes, family_paths.values(), strict=True):
        family = _verified_file(path, inputs.seed, "bct-family-evidence-v1")
        expected_order = sorted(
            (unit for unit in expected_units if unit.candidate_class == candidate),
            key=lambda unit: (
                ("game24", "math_equation_balancer", "word_sorting").index(unit.task),
                ("full_history", "rag_frozen", "bot_style", "reflexion_style").index(
                    unit.baseline
                ),
                unit.probe_id.encode(),
                unit.scientific_replicate,
            ),
        )
        if (
            family.get("freeze_b_sha256")
            != hashlib.sha256(freeze_path.read_bytes()).hexdigest()
            or family.get("ordered_units") != [_unit_value(unit) for unit in expected_order]
        ):
            raise RootlessContractError("ROOTLESS_BCT_ARCHIVE_INVALID")
    projection_path = evidence_root / "projection-manifest.json"
    rows_path = evidence_root / "search-config-rows.json"
    result_path = evidence_root / "bct-result-manifest.json"
    projection = _verified_file(projection_path, inputs.seed, "bct-projection-manifest-v1")
    rows = _verified_file(rows_path, inputs.seed, "search-config-rows-v1")
    result = _verified_file(result_path, inputs.seed, "bct-result-manifest-v1")
    metric_leaves = b"".join(
        hashlib.sha256(canonical_json_value(value)).digest()
        for value in _metric_values(expected_units, screening_results)
    )
    provider_slots = tuple(
        ProviderSlotResult(
            slot.task,
            slot.probe_id,
            _receipt(root, slot, inputs.seed).get("issued") is True,
        )
        for slot in inputs.bct_slots
    )
    expected_rows = build_post_bct(
        screening_results,
        expected_units,
        provider_calls_issued=sum(slot.issued for slot in provider_slots),
        projection_manifest_sha256=hashlib.sha256(projection_path.read_bytes()).hexdigest(),
        provider_slots=provider_slots,
    ).rows
    expected_families = {
        field: hashlib.sha256(path.read_bytes()).hexdigest() for field, path in family_paths.items()
    }
    if (
        any(projection.get(field) != digest for field, digest in expected_families.items())
        or projection.get("freeze_b_sha256") != hashlib.sha256(freeze_path.read_bytes()).hexdigest()
        or projection.get("per_stratum_metrics_root_sha256")
        != hashlib.sha256(metric_leaves).hexdigest()
        or rows.get("ordered_rows") != [asdict(row) for row in expected_rows]
        or rows.get("projection_manifest_sha256")
        != hashlib.sha256(projection_path.read_bytes()).hexdigest()
        or result.get("projection_manifest_sha256")
        != hashlib.sha256(projection_path.read_bytes()).hexdigest()
        or result.get("search_config_rows_sha256") != hashlib.sha256(rows_path.read_bytes()).hexdigest()
        or any(result.get(field) != digest for field, digest in expected_families.items())
    ):
        raise RootlessContractError("ROOTLESS_BCT_ARCHIVE_INVALID")
    if (
        bct_terminal.get("status") != "review_required"
        or bct_terminal.get("reason_code") != "BCT_COMPLETED_REVIEW_REQUIRED"
        or bct_terminal.get("receipt_manifest_sha256")
        != hashlib.sha256(bct_manifest_path.read_bytes()).hexdigest()
        or bct_terminal.get("freeze_b_sha256") != hashlib.sha256(freeze_path.read_bytes()).hexdigest()
        or bct_terminal.get("bct_result_manifest_sha256")
        != hashlib.sha256(result_path.read_bytes()).hexdigest()
        or bct_terminal.get("execution_authority_sha256")
        != _lineage_authority_sha256(root, attempt_id, "bct", inputs.seed, bct_terminal)
    ):
        raise RootlessContractError("ROOTLESS_STAGE_PREFIX_INVALID")


def _lineage_authority_sha256(
    root: Path,
    attempt_id: str,
    stage: str,
    seed: bytes,
    terminal: Mapping[str, JsonValue],
) -> JsonValue:
    path = root / "authorities" / attempt_id / f"{stage}.json"
    if not path.is_file():
        return terminal.get("stage_binding_sha256")
    _verified_file(path, seed, "stage-execution-authority-v1")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _signed(seed: bytes, domain: str, payload: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    value = dict(payload)
    value["signature"] = sign_object(seed, domain, value)
    return value


__all__ = (
    "LineageValidationInput",
    "close_receipt_set",
    "close_stage",
    "derive_freeze_b",
    "validate_closed_lineage",
)
