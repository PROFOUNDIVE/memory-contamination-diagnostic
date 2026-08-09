from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import anyio
import pytest

from memcontam.experiment.phase12.filter_challenge.registry_calibration import BASELINES
from memcontam.experiment.phase12.filter_challenge.rootless_local_contract import (
    RootlessContractError,
    canonical_json_file,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_archive_validator import (
    reject_rootless_at_legacy_seam,
    validate_rootless_bct_archive,
    validate_rootless_screening_archive,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_execution import (
    RENDERER_VERSIONS,
    CompileContext,
    FakeResponse,
    build_bct_compilation,
    build_screening_compilation,
    compile_request_goldens,
    execute_fake_stage,
    validate_compilation,
)

ROOT = Path(__file__).resolve().parents[1]
PROBES = ROOT / "data/phase12/filter_v5_bct_v1/probe_construction_manifest_v1.json"
PROFILE = "local_rootless_non_authoritative"


def _probes() -> dict[str, tuple[str, ...]]:
    payload = json.loads(PROBES.read_bytes())
    return {
        task: tuple(probe["probe_id"] for probe in task_probes)
        for task, task_probes in payload["probes"].items()
    }


def _context(stage: str) -> CompileContext:
    return CompileContext(
        attempt_id="fixture-t5",
        stage=stage,
        source_manifest_sha256="1" * 64,
        input_manifest_sha256="2" * 64,
        compiler_sha256="3" * 64,
    )


def test_compiler_covers_exact_schedules_and_is_deterministic() -> None:
    # Given: the historical six probes and a valid Freeze-B projection.
    probes = _probes()
    selected = {task: tuple(sorted(values, key=str.encode)[:2]) for task, values in probes.items()}

    # When: both schedules are independently compiled twice.
    screening_a = build_screening_compilation(_context("screening"), probes)
    screening_b = build_screening_compilation(_context("screening"), probes)
    bct_a = build_bct_compilation(_context("bct"), selected)
    bct_b = build_bct_compilation(_context("bct"), selected)

    # Then: cardinalities, bytes, hashes, and renderer identity are frozen.
    assert len(screening_a.slots) == len(screening_b.slots) == 90
    assert len({(s.task, s.probe_id, s.baseline) for s in screening_a.slots}) == 72
    assert len(bct_a.slots) == len(bct_b.slots) == 480
    assert len({(s.task, s.probe_id, s.candidate_class, s.scientific_replicate, s.baseline)
                for s in bct_a.slots}) == 192
    assert len({(s.task, s.probe_id, s.candidate_class, s.scientific_replicate, s.side,
                 s.baseline) for s in bct_a.slots}) == 384
    assert canonical_json_file(screening_a.to_json()) == canonical_json_file(screening_b.to_json())
    assert canonical_json_file(bct_a.to_json()) == canonical_json_file(bct_b.to_json())
    assert RENDERER_VERSIONS == (
        "full-history-generate=rootless-adapter-v1",
        "rag-generate=rootless-adapter-v1",
        "bot-problem-distill=rootless-adapter-v1",
        "bot-instantiate-solve=rootless-adapter-v1",
        "reflexion-generate=rootless-adapter-v1",
        "responses-text-extractor=rootless-responses-text-v1",
    )


def test_compiler_maps_replicates_and_bot_predecessors() -> None:
    # Given/When: the selected BCT schedule is compiled in frozen order.
    probes = _probes()
    selected = {task: values[:2] for task, values in probes.items()}
    compiled = build_bct_compilation(_context("bct"), selected)

    # Then: scientific order maps to the historical executor and BoT has two receipts per side.
    assert {(slot.scientific_replicate, slot.executor_replicate_id, slot.execution_order)
            for slot in compiled.slots} == {
                (1, 0, "control_first"),
                (2, 1, "challenge_first"),
            }
    bot = [slot for slot in compiled.slots if slot.baseline == "bot_style"]
    assert all(slot.predecessor_slot_ids for slot in bot if slot.native_stage == "bot_instantiate_solve")
    groups: dict[tuple[str, str, str, int, str], list[str]] = {}
    for slot in bot:
        key = (slot.task, slot.probe_id, str(slot.candidate_class),
               int(slot.scientific_replicate), slot.side)
        groups.setdefault(key, []).append(slot.native_stage)
    assert set(map(tuple, groups.values())) == {("bot_problem_distill", "bot_instantiate_solve")}


def test_static_hash_rejects_field_and_predecessor_drift() -> None:
    # Given: one valid compiled slot.
    slot = build_screening_compilation(_context("screening"), _probes()).slots[0]

    # When/Then: static identity and source-role drift cannot retain authority.
    validate_compilation(slot)
    with pytest.raises(RootlessContractError, match="ROOTLESS_COMPILATION_INVALID"):
        validate_compilation(replace(slot, source_manifest_sha256="9" * 64))
    with pytest.raises(RootlessContractError, match="ROOTLESS_COMPILATION_INVALID"):
        validate_compilation(replace(slot, predecessor_slot_ids=(slot.slot_id,)))
    with pytest.raises(RootlessContractError, match="ROOTLESS_COMPILATION_INVALID"):
        validate_compilation(
            replace(slot, messages=(replace(slot.messages[0], content="drift"), *slot.messages[1:]))
        )
    with pytest.raises(RootlessContractError, match="ROOTLESS_COMPILATION_INVALID"):
        validate_compilation(replace(slot, request=b"{}"))


def test_goldens_cover_all_baselines_and_dynamic_bot_solve() -> None:
    # Given/When: the complete attempt-independent universe is compiled twice.
    first = compile_request_goldens(_probes())
    second = compile_request_goldens(_probes())

    # Then: all 1,530 entries are stable and dynamic solves defer predecessor injection.
    assert canonical_json_file(first) == canonical_json_file(second)
    entries = first["ordered_entries"]
    assert isinstance(entries, list) and len(entries) == 1530
    assert {entry["baseline"] for entry in entries} == set(BASELINES)
    dynamic = [entry for entry in entries if entry["native_stage"] == "bot_instantiate_solve"]
    assert dynamic and all(
        entry["message_roles"] == ["system", "user"]
        and entry["message_content_sha256s"] is None
        and entry["input_items_sha256"] is None
        and entry["request_sha256"] is None
        for entry in dynamic
    )


@pytest.mark.parametrize(
    ("response", "provider_status", "raw_parse_status"),
    [
        (FakeResponse.completed(("first", "second")), "completed", "success"),
        (FakeResponse.completed(()), "completed", "failure"),
        (FakeResponse.refusal(), "completed", "failure"),
        (FakeResponse.malformed(), "archive_error", "not_run"),
        (FakeResponse.status("failed"), "failed", "not_run"),
        (FakeResponse.status("cancelled"), "cancelled", "not_run"),
        (FakeResponse.status("incomplete"), "incomplete", "not_run"),
        (FakeResponse.status("queued"), "nonterminal", "not_run"),
        (FakeResponse.status("in_progress"), "nonterminal", "not_run"),
    ],
)
def test_fake_execution_uses_broker_response_matrix(
    tmp_path: Path,
    response: FakeResponse,
    provider_status: str,
    raw_parse_status: str,
) -> None:
    # Given: one compiled slot and a schema-disjoint fixture response.
    slot = build_screening_compilation(_context("screening"), _probes()).slots[0]

    # When: execution crosses the T4 fake broker.
    result = anyio.run(execute_fake_stage, (slot,), response, tmp_path)

    # Then: the broker classification and archive are preserved without Pilot-B calls.
    assert result.provider_calls_issued == 1
    assert result.pilot_b_calls == 0
    assert result.receipts[0]["provider_status"] == provider_status
    assert result.outcomes[0]["raw_parse_status"] == raw_parse_status


def test_archive_failure_blocks_descendants_with_zero_pilot_b_calls(tmp_path: Path) -> None:
    # Given: one BoT distill/solve chain and a malformed predecessor response.
    compiled = build_screening_compilation(_context("screening"), _probes())
    pair = tuple(slot for slot in compiled.slots if slot.baseline == "bot_style")[:2]

    # When: the predecessor archive fails.
    result = anyio.run(execute_fake_stage, pair, FakeResponse.malformed(), tmp_path)

    # Then: solve is accounted as not-issued and no Pilot-B route is reachable.
    assert result.provider_calls_issued == 1
    assert result.not_issued_count == 1
    assert result.pilot_b_calls == 0
    assert result.receipts[1]["issued"] is False
    assert result.receipts[1]["compile_status"] == "blocked_predecessor"


def test_bct_receipt_records_scientific_and_executor_replicates(tmp_path: Path) -> None:
    # Given: one static R0 BCT provider slot.
    selected = {task: probes[:2] for task, probes in _probes().items()}
    slot = build_bct_compilation(_context("bct"), selected).slots[0]

    # When: it is issued through the fake broker.
    result = anyio.run(execute_fake_stage, (slot,), FakeResponse.completed(("ok",)), tmp_path)

    # Then: receipt identity preserves the scientific-to-executor bijection.
    assert result.receipts[0]["scientific_replicate"] == 1
    assert result.receipts[0]["executor_replicate_id"] == 0


def test_full_fake_screening_and_bct_reconcile_exact_provider_slots(tmp_path: Path) -> None:
    # Given: the full screening schedule and selected two-probe R0 BCT schedule.
    probes = _probes()
    selected = {task: values[:2] for task, values in probes.items()}
    screening = build_screening_compilation(_context("screening"), probes)
    bct = build_bct_compilation(_context("bct"), selected)
    response = FakeResponse.completed(("fixture answer",))

    # When: both stages run through independent disposable T4 fake brokers.
    screening_result = anyio.run(execute_fake_stage, screening.slots, response, tmp_path / "s")
    bct_result = anyio.run(execute_fake_stage, bct.slots, response, tmp_path / "b")

    # Then: all 90 and 480 provider slots are issued, archived, and ledger-reconciled.
    assert screening_result.provider_calls_issued == 90
    assert bct_result.provider_calls_issued == 480
    screening_archive = tmp_path / "s/basetemp/t5/tmp/fake-state/fixture-t5"
    bct_archive = tmp_path / "b/basetemp/t5/tmp/fake-state/fixture-t5"
    assert validate_rootless_screening_archive(screening_archive, screening.slots).issued_slots == 90
    assert validate_rootless_bct_archive(bct_archive, bct.slots).issued_slots == 480
    bot_receipts = [
        receipt
        for slot, receipt in zip(bct.slots, bct_result.receipts, strict=True)
        if slot.baseline == "bot_style"
    ]
    assert len(bot_receipts) == 192


def test_rootless_archive_validates_separately_and_legacy_seam_rejects(tmp_path: Path) -> None:
    # Given: a complete fake screening slot and its rootless-only archive.
    compilation = build_screening_compilation(_context("screening"), _probes())
    slot = compilation.slots[0]
    result = anyio.run(execute_fake_stage, (slot,), FakeResponse.completed(("ok",)), tmp_path)
    archive_root = tmp_path / "basetemp/t5/tmp/fake-state/fixture-t5"

    # When: the separate rootless validator reads the archive.
    validated = validate_rootless_screening_archive(archive_root, (slot,))

    # Then: reconciliation succeeds while the same profile is forbidden at legacy authority.
    assert validated.accounted_slots == 1
    assert validated.issued_slots == result.provider_calls_issued
    with pytest.raises(RootlessContractError, match="ROOTLESS_PROFILE_FORBIDDEN"):
        reject_rootless_at_legacy_seam(
            canonical_json_file({"profile": PROFILE, "kind": "rootless_screening_archive"})
        )


def test_rootless_archive_rejects_provenance_and_receipt_drift(tmp_path: Path) -> None:
    # Given: one valid fake archive whose receipt is then replaced with noncanonical provenance.
    slot = build_screening_compilation(_context("screening"), _probes()).slots[0]
    anyio.run(execute_fake_stage, (slot,), FakeResponse.completed(("ok",)), tmp_path)
    archive_root = tmp_path / "basetemp/t5/tmp/fake-state/fixture-t5"
    receipt_path = archive_root / f"attempts/fixture-t5/screening/slots/{slot.slot_id}/call-receipt.json"
    receipt = json.loads(receipt_path.read_bytes())
    receipt["parsed_response_source_call_id"] = "other-answer"
    receipt_path.write_bytes(canonical_json_file(receipt))

    # When/Then: archive failure blocks acceptance rather than reaching any Pilot-B route.
    with pytest.raises(RootlessContractError, match="ROOTLESS_ARCHIVE_INVALID"):
        validate_rootless_screening_archive(archive_root, (slot,))


def test_rootless_archive_rejects_response_blob_substitution(tmp_path: Path) -> None:
    # Given: a valid signed archive whose persisted response body is substituted.
    slot = build_screening_compilation(_context("screening"), _probes()).slots[0]
    anyio.run(execute_fake_stage, (slot,), FakeResponse.completed(("ok",)), tmp_path)
    archive_root = tmp_path / "basetemp/t5/tmp/fake-state/fixture-t5"
    body = archive_root / f"attempts/fixture-t5/screening/slots/{slot.slot_id}/response.body"
    body.write_bytes(b"substituted")

    # When/Then: signed manifest-to-blob reconciliation rejects the archive.
    with pytest.raises(RootlessContractError, match="ROOTLESS_ARCHIVE_INVALID"):
        validate_rootless_screening_archive(archive_root, (slot,))
