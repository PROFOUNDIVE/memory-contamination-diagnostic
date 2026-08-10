from __future__ import annotations

# allow: SIZE_OK — one full lifecycle fixture supports the signed corruption matrix.

import base64
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import anyio

from memcontam.experiment.phase12.filter_challenge.rootless_local_binding import (
    build_fake_stage_binding,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_broker import (
    build_fake_broker_for_tests,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_contract import (
    JsonValue,
    RootlessContractError,
    canonical_json_file,
    parse_canonical_object,
    public_key_from_seed,
    sign_object,
)
from memcontam.experiment.phase12.filter_challenge import rootless_local_closure
from memcontam.experiment.phase12.filter_challenge import rootless_local_operator
from memcontam.experiment.phase12.filter_challenge import rootless_local_runtime
from memcontam.experiment.phase12.filter_challenge.rootless_local_execution import (
    CompileContext,
    FakeResponse,
    SlotCompilation,
    StageCompilation,
    build_bct_compilation,
    build_screening_compilation,
    execute_fake_stage,
    load_probe_ids,
    _StaticTransport,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_runtime import (
    run_stage_process,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_operator import (
    publish_from_root,
    seal_final_from_stage,
    write_new_or_same,
    write_anchor,
)


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class _AnswerTransport:
    fixture_id: str
    answers: dict[str, str]

    async def exchange(self, slot_id: str, request: bytes) -> dict[str, str | int | bool | None]:
        del request
        body = FakeResponse.completed((self.answers[slot_id],)).body
        return {
            "schema_version": "rootless_fake_http_exchange_v1",
            "profile": "local_rootless_non_authoritative",
            "kind": "fake_http_exchange",
            "fixture_id": self.fixture_id,
            "slot_id": slot_id,
            "lifecycle_marker": "streaming_body",
            "response_surfaced": True,
            "http_status": 200,
            "headers_base64": base64.b64encode(b"\0\0\0\0").decode("ascii"),
            "body_base64": base64.b64encode(body).decode("ascii"),
            "raised_exception": None,
        }


class _CountingTransport:
    def __init__(self, fixture_id: str, response: FakeResponse) -> None:
        self.fixture_id = fixture_id
        self.response = response
        self.calls = 0

    async def exchange(self, slot_id: str, request: bytes) -> dict[str, str | int | bool | None]:
        self.calls += 1
        return await _StaticTransport(self.fixture_id, self.response).exchange(slot_id, request)


def _answers(compilation: StageCompilation) -> dict[str, str]:
    manifest = parse_canonical_object(
        (ROOT / "data/phase12/filter_v5_bct_v1/probe_construction_manifest_v1.json").read_bytes()
    )
    probes = manifest["probes"]
    assert isinstance(probes, dict)
    certificates: dict[str, dict[str, JsonValue]] = {}
    for rows in probes.values():
        assert isinstance(rows, list)
        for row in rows:
            assert isinstance(row, dict)
            probe_id = row["probe_id"]
            certificate = row["certificate"]
            assert isinstance(probe_id, str) and isinstance(certificate, dict)
            certificates[probe_id] = certificate
    answers: dict[str, str] = {}
    for slot in compilation.slots:
        certificate = certificates[slot.probe_id]
        if slot.native_stage == "bot_problem_distill":
            answers[slot.slot_id] = json.dumps(
                {
                    "key_information": "registered probe input",
                    "restrictions": "preserve the registered task contract",
                    "distilled_task": "solve the registered probe",
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        elif slot.task == "game24":
            answers[slot.slot_id] = (
                "0"
                if slot.side == "challenge"
                and slot.candidate_class in {"certified_false", "ordinary_false"}
                else str(certificate["expression"])
            )
        elif slot.task == "math_equation_balancer":
            answers[slot.slot_id] = (
                "wrong"
                if slot.side == "challenge"
                and slot.candidate_class in {"certified_false", "ordinary_false"}
                else str(certificate["target"])
            )
        else:
            words = certificate["correct_order"]
            assert isinstance(words, list) and all(isinstance(word, str) for word in words)
            ordered = [word for word in words if isinstance(word, str)]
            if slot.side == "challenge" and slot.candidate_class in {
                "certified_false",
                "ordinary_false",
            }:
                ordered.reverse()
            answers[slot.slot_id] = " ".join(ordered)
    return answers


def test_valid_fake_screening_closes_completed_estimable(tmp_path: Path) -> None:
    # Given: all 90 screening slots return answers accepted by the production task verifiers.
    context = CompileContext("closure-screening", "screening", "1" * 64, "2" * 64, "3" * 64)
    compilation = build_screening_compilation(context, load_probe_ids(ROOT))
    binding = build_fake_stage_binding(
        fixture_id=context.attempt_id,
        stage="screening",
        source_manifest_sha256=context.source_manifest_sha256,
        input_manifest_sha256=context.input_manifest_sha256,
        compiler_sha256=context.compiler_sha256,
        schedule_sha256=compilation.ordered_slot_root_sha256,
        fake_scenario_sha256="4" * 64,
    )
    fake_root = tmp_path / "basetemp/closure/tmp/fake-state" / context.attempt_id
    fake_root.mkdir(mode=0o700, parents=True)
    (fake_root / "plan-bind.md").write_bytes(b"fake plan binding\n")
    (fake_root / "plan-bind.md").chmod(0o600)

    # When: screening executes through the fork/socket process seam.
    exit_code = run_stage_process(
        compilation.slots,
        lambda: build_fake_broker_for_tests(
            binding,
            _AnswerTransport(context.attempt_id, _answers(compilation)),
            fake_root,
        ),
    )

    # Then: persisted verifier evidence supports the completed-estimable terminal.
    terminal = parse_canonical_object(
        (fake_root / f"terminals/{context.attempt_id}/screening.json").read_bytes()
    )
    outcomes = tuple(
        parse_canonical_object(
            (fake_root / f"attempts/{context.attempt_id}/screening/slots/{slot.slot_id}/typed-outcome.json").read_bytes()
        )
        for slot in compilation.slots
        if slot.native_stage != "bot_problem_distill"
    )
    assert exit_code == 0
    assert terminal["status"] == "completed_estimable"
    assert terminal["reason_code"] == "SCREENING_ESTIMABLE"
    assert all(outcome["verifier_status"] == "success" for outcome in outcomes)
    assert all(outcome["verifier_result"] is True for outcome in outcomes)

    freeze_sha256 = rootless_local_closure.derive_freeze_b(
        fake_root,
        hashlib.sha256(f"rootless-fixture:{context.attempt_id}".encode()).digest(),
        compilation.slots,
    )
    freeze = parse_canonical_object(
        (fake_root / f"freeze/{context.attempt_id}/freeze_b.json").read_bytes()
    )
    assert freeze_sha256 == hashlib.sha256(
        (fake_root / f"freeze/{context.attempt_id}/freeze_b.json").read_bytes()
    ).hexdigest()
    assert freeze["screening_stage_terminal_sha256"] == hashlib.sha256(
        (fake_root / f"terminals/{context.attempt_id}/screening.json").read_bytes()
    ).hexdigest()
    assert freeze["selected_game24_probe_ids"] == list(load_probe_ids(ROOT)["game24"][:2])

    selected: dict[str, tuple[str, ...]] = {}
    for task, field in (
        ("game24", "selected_game24_probe_ids"),
        ("math_equation_balancer", "selected_math_equation_balancer_probe_ids"),
        ("word_sorting", "selected_word_sorting_probe_ids"),
    ):
        values = freeze[field]
        assert isinstance(values, list) and all(isinstance(probe, str) for probe in values)
        selected[task] = tuple(probe for probe in values if isinstance(probe, str))
    assert all(all(isinstance(probe, str) for probe in probes) for probes in selected.values())
    bct_context = CompileContext(context.attempt_id, "bct", "1" * 64, "2" * 64, "3" * 64)
    bct = build_bct_compilation(bct_context, selected)
    bct_binding = build_fake_stage_binding(
        fixture_id=context.attempt_id,
        stage="bct",
        source_manifest_sha256=bct_context.source_manifest_sha256,
        input_manifest_sha256=bct_context.input_manifest_sha256,
        compiler_sha256=bct_context.compiler_sha256,
        schedule_sha256=bct.ordered_slot_root_sha256,
        fake_scenario_sha256="5" * 64,
    )

    # When: BCT setup reaches the absent ordinary native-writer authority.
    bct_exit_code = run_stage_process(
        bct.slots,
        lambda: build_fake_broker_for_tests(
            bct_binding,
            _AnswerTransport(context.attempt_id, _answers(bct)),
            fake_root,
        ),
    )

    evidence_root = fake_root / f"attempts/{context.attempt_id}/bct/evidence"
    ledger_records = tuple(
        parse_canonical_object(path.read_bytes())
        for path in (fake_root / "ledger/global/records").glob("*.json")
    )
    final = parse_canonical_object(
        (fake_root / f"terminals/{context.attempt_id}/final.json").read_bytes()
    )
    assert bct_exit_code == 69
    assert final["status"] == "blocked"
    assert final["reason_code"] == "ROOTLESS_BCT_SETUP_FAILED"
    assert not evidence_root.exists()
    assert not (fake_root / f"attempts/{context.attempt_id}/bct").exists()
    assert not any(
        record.get("record_kind") == "terminal" and record.get("stage") == "bct"
        for record in ledger_records
    )
    assert not (fake_root / f"terminals/{context.attempt_id}/bct.json").exists()


def _rewrite_signed(
    path: Path,
    seed: bytes,
    domain: str,
    field: str,
    replacement: JsonValue,
) -> None:
    value = parse_canonical_object(path.read_bytes())
    value[field] = replacement
    del value["signature"]
    value["signature"] = sign_object(seed, domain, value)
    path.write_bytes(canonical_json_file(value))


def test_screening_is_estimable_at_two_common_strict_probes_per_task(tmp_path: Path) -> None:
    # Given: exactly two probes per task succeed across all four baselines.
    context = CompileContext("closure-minimum", "screening", "1" * 64, "2" * 64, "3" * 64)
    compilation = build_screening_compilation(context, load_probe_ids(ROOT))
    selected = {
        task: set(probes[:2]) for task, probes in load_probe_ids(ROOT).items()
    }
    minimum_slots = tuple(
        slot for slot in compilation.slots if slot.probe_id in selected[slot.task]
    )
    binding = build_fake_stage_binding(
        fixture_id=context.attempt_id,
        stage="screening",
        source_manifest_sha256=context.source_manifest_sha256,
        input_manifest_sha256=context.input_manifest_sha256,
        compiler_sha256=context.compiler_sha256,
        schedule_sha256=compilation.ordered_slot_root_sha256,
        fake_scenario_sha256="4" * 64,
    )
    fake_root = tmp_path / "basetemp/minimum/tmp/fake-state" / context.attempt_id
    fake_root.mkdir(mode=0o700, parents=True)

    # When: only that minimum frozen screening subset crosses the process seam.
    exit_code = run_stage_process(
        minimum_slots,
        lambda: build_fake_broker_for_tests(
            binding,
            _AnswerTransport(context.attempt_id, _answers(compilation)),
            fake_root,
        ),
    )

    # Then: all unneeded controls are irrelevant to the estimability threshold.
    terminal = parse_canonical_object(
        (fake_root / f"terminals/{context.attempt_id}/screening.json").read_bytes()
    )
    assert exit_code == 0
    assert len(minimum_slots) == 30
    assert terminal["status"] == "completed_estimable"


def test_operational_stop_seals_partial_manifest_before_terminal(tmp_path: Path) -> None:
    # Given: the first screening response is operationally archive-invalid.
    context = CompileContext("closure-partial", "screening", "1" * 64, "2" * 64, "3" * 64)
    compilation = build_screening_compilation(context, load_probe_ids(ROOT))
    response = FakeResponse.malformed()
    binding = build_fake_stage_binding(
        fixture_id=context.attempt_id,
        stage="screening",
        source_manifest_sha256=context.source_manifest_sha256,
        input_manifest_sha256=context.input_manifest_sha256,
        compiler_sha256=context.compiler_sha256,
        schedule_sha256=compilation.ordered_slot_root_sha256,
        fake_scenario_sha256=hashlib.sha256(response.body).hexdigest(),
    )
    fake_root = tmp_path / "basetemp/partial/tmp/fake-state" / context.attempt_id
    fake_root.mkdir(mode=0o700, parents=True)

    # When: the process seam stops on that signed call receipt.
    exit_code = run_stage_process(
        compilation.slots,
        lambda: build_fake_broker_for_tests(
            binding, _StaticTransport(context.attempt_id, response), fake_root
        ),
    )

    # Then: the exact admitted five-slot batch precedes blocked stage/final terminals.
    manifest = parse_canonical_object(
        (fake_root / f"attempts/{context.attempt_id}/screening/receipt-manifest.json").read_bytes()
    )
    terminal = parse_canonical_object(
        (fake_root / f"terminals/{context.attempt_id}/screening.json").read_bytes()
    )
    assert exit_code == 69
    assert manifest["registered_slot_count"] == 90
    assert manifest["accounted_slot_count"] == 5
    assert terminal["status"] == "blocked"
    assert terminal["reason_code"] == "ROOTLESS_ARCHIVE_INVALID"
    assert (fake_root / f"terminals/{context.attempt_id}/final.json").is_file()


def test_operational_stop_before_dispatch_seals_zero_receipt_manifest(tmp_path: Path) -> None:
    # Given: a screening broker has acquired the lock but dispatch is stopped before slot one.
    context = CompileContext("closure-zero", "screening", "1" * 64, "2" * 64, "3" * 64)
    compilation = build_screening_compilation(context, load_probe_ids(ROOT))
    response = FakeResponse.completed(("unused",))
    binding = build_fake_stage_binding(
        fixture_id=context.attempt_id,
        stage="screening",
        source_manifest_sha256=context.source_manifest_sha256,
        input_manifest_sha256=context.input_manifest_sha256,
        compiler_sha256=context.compiler_sha256,
        schedule_sha256=compilation.ordered_slot_root_sha256,
        fake_scenario_sha256=hashlib.sha256(response.body).hexdigest(),
    )
    fake_root = tmp_path / "basetemp/zero/tmp/fake-state" / context.attempt_id
    fake_root.mkdir(mode=0o700, parents=True)
    (fake_root / "plan-bind.md").write_bytes(b"fake plan binding\n")
    (fake_root / "plan-bind.md").chmod(0o600)
    broker = build_fake_broker_for_tests(
        binding, _StaticTransport(context.attempt_id, response), fake_root
    )
    broker.stage_operational_reason = "ROOTLESS_MISSING_SECRET"

    # When: the parent advances the empty durable prefix while holding runtime.lock.
    try:
        rootless_local_closure.close_stage(broker, compilation.slots)
    finally:
        broker.close()

    # Then: SHA256(empty) and zero accounting precede blocked stage/final terminals.
    manifest = parse_canonical_object(
        (fake_root / f"attempts/{context.attempt_id}/screening/receipt-manifest.json").read_bytes()
    )
    assert manifest["accounted_slot_count"] == 0
    assert manifest["ordered_receipt_root_sha256"] == hashlib.sha256(b"").hexdigest()
    assert (fake_root / f"terminals/{context.attempt_id}/screening.json").is_file()
    assert (fake_root / f"terminals/{context.attempt_id}/final.json").is_file()

    final_path = fake_root / f"terminals/{context.attempt_id}/final.json"
    _rewrite_signed(
        final_path,
        hashlib.sha256(f"rootless-fixture:{context.attempt_id}".encode()).digest(),
        "attempt-terminal-v1",
        "predecessor_stage_terminal_sha256",
        "f" * 64,
    )
    publication_repository = tmp_path / "screening-publication"
    publication_repository.mkdir(mode=0o700)
    write_anchor(publication_repository, "a" * 40)
    with pytest.raises(RootlessContractError):
        publish_from_root(
            publication_repository,
            fake_root,
            context.attempt_id,
            "2026-08-09T12:00:00Z",
        )


def test_atomic_writer_recovers_identical_orphan_temp_without_overwrite(tmp_path: Path) -> None:
    # Given: a fully fsynced temp left before its atomic installation.
    destination = tmp_path / "state/artifact.json"
    destination.parent.mkdir(mode=0o700)
    value: dict[str, JsonValue] = {"schema_version": "fixture_v1", "value": 1}
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_bytes(canonical_json_file(value))
    temporary.chmod(0o600)

    # When: reconciliation repeats the same immutable write.
    digest = write_new_or_same(destination, value)

    # Then: the exact temp becomes authoritative and no orphan remains.
    assert digest == hashlib.sha256(destination.read_bytes()).hexdigest()
    assert not temporary.exists()
    original = destination.read_bytes()
    with pytest.raises(RootlessContractError, match="ROOTLESS_REPORTING_CONFLICT"):
        write_new_or_same(destination, {"schema_version": "fixture_v1", "value": 2})
    assert destination.read_bytes() == original


def test_reconcile_replays_durable_prefix_without_duplicate_transport(tmp_path: Path) -> None:
    # Given: two durable issued receipts left before receipt-manifest closure.
    context = CompileContext("reconcile-prefix", "screening", "1" * 64, "2" * 64, "3" * 64)
    compilation = build_screening_compilation(context, load_probe_ids(ROOT))
    slots = tuple(slot for slot in compilation.slots if not slot.predecessor_slot_ids)[:2]
    assert len(slots) == 2
    response = FakeResponse.completed(("fixture answer",))
    anyio.run(execute_fake_stage, slots, response, tmp_path)
    fake_root = tmp_path / "basetemp/t5/tmp/fake-state" / context.attempt_id
    receipt_paths = tuple(
        fake_root / f"attempts/{context.attempt_id}/screening/slots/{slot.slot_id}/call-receipt.json"
        for slot in slots
    )
    original_receipts = tuple(path.read_bytes() for path in receipt_paths)
    binding = build_fake_stage_binding(
        fixture_id=context.attempt_id,
        stage="screening",
        source_manifest_sha256=context.source_manifest_sha256,
        input_manifest_sha256=context.input_manifest_sha256,
        compiler_sha256=context.compiler_sha256,
        schedule_sha256="4" * 64,
        fake_scenario_sha256=hashlib.sha256(response.body).hexdigest(),
    )
    transport = _CountingTransport(context.attempt_id, FakeResponse.malformed())

    # When: process reconciliation encounters that exact durable prefix.
    exit_code = run_stage_process(
        slots,
        lambda: build_fake_broker_for_tests(binding, transport, fake_root),
    )

    # Then: it replays the durable branch, closes once, and never mutates the receipt.
    assert exit_code == 0
    assert tuple(path.read_bytes() for path in receipt_paths) == original_receipts
    assert transport.calls == 0
    assert (fake_root / f"attempts/{context.attempt_id}/screening/receipt-manifest.json").is_file()
    assert (fake_root / f"terminals/{context.attempt_id}/screening.json").is_file()


def test_reconcile_validates_maximal_durable_prefix_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given: two independently durable receipts and a validator that records each admitted prefix.
    context = CompileContext("reconcile-prefix-once", "screening", "1" * 64, "2" * 64, "3" * 64)
    compilation = build_screening_compilation(context, load_probe_ids(ROOT))
    slots = tuple(slot for slot in compilation.slots if not slot.predecessor_slot_ids)[:2]
    assert len(slots) == 2
    response = FakeResponse.completed(("fixture answer",))
    anyio.run(execute_fake_stage, slots, response, tmp_path)
    fake_root = tmp_path / "basetemp/t5/tmp/fake-state" / context.attempt_id
    binding = build_fake_stage_binding(
        fixture_id=context.attempt_id,
        stage="screening",
        source_manifest_sha256=context.source_manifest_sha256,
        input_manifest_sha256=context.input_manifest_sha256,
        compiler_sha256=context.compiler_sha256,
        schedule_sha256="4" * 64,
        fake_scenario_sha256=hashlib.sha256(response.body).hexdigest(),
    )
    validations: list[tuple[SlotCompilation, ...]] = []

    def validate(
        _root: Path, durable_slots: tuple[SlotCompilation, ...], *, seed: bytes | None = None
    ) -> None:
        del seed
        validations.append(durable_slots)

    monkeypatch.setattr(rootless_local_runtime, "validate_rootless_screening_archive", validate)
    transport = _CountingTransport(context.attempt_id, FakeResponse.malformed())

    # When: reconciliation loads the maximal durable prefix under the broker-held lock.
    exit_code = run_stage_process(
        slots,
        lambda: build_fake_broker_for_tests(binding, transport, fake_root),
    )

    # Then: exactly one collective validation exposes the two already verified receipts.
    assert exit_code == 0
    assert validations == [slots]
    assert transport.calls == 0


def test_tampered_recovered_predecessor_blocks_dependent_dispatch(tmp_path: Path) -> None:
    # Given: a parseable, re-signed but receipt-inconsistent durable predecessor outcome.
    context = CompileContext("reconcile-tampered", "screening", "1" * 64, "2" * 64, "3" * 64)
    compilation = build_screening_compilation(context, load_probe_ids(ROOT))
    predecessor = next(slot for slot in compilation.slots if slot.native_stage == "bot_problem_distill")
    descendant = next(
        slot
        for slot in compilation.slots
        if slot.predecessor_slot_ids == (predecessor.slot_id,)
    )
    response = FakeResponse.completed(
        (
            '{"distilled_task":"solve the registered probe",'
            '"key_information":"registered probe input",'
            '"restrictions":"preserve the registered task contract"}',
        )
    )
    anyio.run(execute_fake_stage, (predecessor,), response, tmp_path)
    fake_root = tmp_path / "basetemp/t5/tmp/fake-state" / context.attempt_id
    outcome_path = (
        fake_root
        / f"attempts/{context.attempt_id}/screening/slots/{predecessor.slot_id}/typed-outcome.json"
    )
    _rewrite_signed(
        outcome_path,
        hashlib.sha256(f"rootless-fixture:{context.attempt_id}".encode()).digest(),
        "typed-call-outcome-v1",
        "parsed_output",
        (
            '{"distilled_task":"tampered task",'
            '"key_information":"registered probe input",'
            '"restrictions":"preserve the registered task contract"}'
        ),
    )

    transport = _CountingTransport(context.attempt_id, response)
    binding = build_fake_stage_binding(
        fixture_id=context.attempt_id,
        stage=context.stage,
        source_manifest_sha256=context.source_manifest_sha256,
        input_manifest_sha256=context.input_manifest_sha256,
        compiler_sha256=context.compiler_sha256,
        schedule_sha256=compilation.ordered_slot_root_sha256,
        fake_scenario_sha256=hashlib.sha256(response.body).hexdigest(),
    )

    # When: reconciliation encounters the altered predecessor before its dependent slot.
    with pytest.raises(RootlessContractError):
        run_stage_process(
            (predecessor, descendant),
            lambda: build_fake_broker_for_tests(binding, transport, fake_root),
        )

    # Then: no provider transport receives a descendant request from unvalidated output.
    assert transport.calls == 0


def test_state_inventory_rejects_symlink_without_reading_target(tmp_path: Path) -> None:
    # Given: a valid stage terminal plus an immutable-state symlink to outside bytes.
    root = tmp_path / "state"
    seed = bytes(range(32))
    (root / "keys").mkdir(mode=0o700, parents=True)
    (root / "keys/ed25519-private.key").write_bytes(seed)
    (root / "keys/ed25519-private.key").chmod(0o600)
    terminal_path = root / "terminals/attempt-001/screening.json"
    terminal_path.parent.mkdir(mode=0o700, parents=True)
    terminal: dict[str, JsonValue] = {
        "schema_version": "rootless_stage_terminal_v1",
        "profile": "local_rootless_non_authoritative",
        "kind": "stage_terminal",
        "attempt_id": "attempt-001",
        "stage": "screening",
        "status": "not_estimable",
        "reason_code": "SCREENING_NOT_ESTIMABLE",
        "ledger_head_sha256": "1" * 64,
        "created_at": "2026-08-09T12:00:00Z",
        "key_fingerprint": hashlib.sha256(public_key_from_seed(seed)).hexdigest(),
    }
    terminal["signature"] = sign_object(seed, "stage-terminal-v1", terminal)
    terminal_path.write_bytes(canonical_json_file(terminal))
    terminal_path.chmod(0o600)
    outside = tmp_path / "outside"
    outside.write_bytes(b"must-not-be-read")
    (root / "unsafe-link").symlink_to(outside)

    # When/Then: finalization rejects the inventory graph without following the link.
    with pytest.raises(RootlessContractError, match="ROOTLESS_REPORTING_CONFLICT"):
        seal_final_from_stage(
            root,
            "attempt-001",
            "screening",
            "not_estimable",
            "SCREENING_NOT_ESTIMABLE",
            "2026-08-09T12:00:00Z",
        )
    assert outside.read_bytes() == b"must-not-be-read"


def test_publication_clock_validator_rejects_signed_wrong_stage_checkpoint(tmp_path: Path) -> None:
    root = tmp_path / "state"
    attempt_id = "clock-chain"
    seed = bytes(range(32))
    authority_path = root / f"authorities/{attempt_id}/screening.json"
    authority_path.parent.mkdir(mode=0o700, parents=True)
    authority: dict[str, JsonValue] = {"schema_version": "fixture_v1", "attempt_id": attempt_id}
    authority["signature"] = sign_object(seed, "stage-execution-authority-v1", authority)
    authority_raw = canonical_json_file(authority)
    authority_path.write_bytes(authority_raw)
    authority_path.chmod(0o600)
    first: dict[str, JsonValue] = {
        "schema_version": "rootless_runtime_clock_checkpoint_v1",
        "profile": "local_rootless_non_authoritative",
        "kind": "runtime_clock_checkpoint",
        "sequence": 0,
        "previous_checkpoint_sha256": None,
        "attempt_id": attempt_id,
        "stage": "screening",
        "boot_id_sha256": "a" * 64,
        "realtime_at_claim": "2026-08-09T12:00:00Z",
        "monotonic_ns_at_claim": 1,
        "stage_started_at": None,
        "stage_monotonic_ns": None,
        "checkpoint_realtime": "2026-08-09T12:00:00Z",
        "checkpoint_monotonic_ns": 1,
        "key_fingerprint": hashlib.sha256(public_key_from_seed(seed)).hexdigest(),
    }
    first["signature"] = sign_object(seed, "runtime-clock-checkpoint-v1", first)
    first_raw = canonical_json_file(first)
    second: dict[str, JsonValue] = {
        **first,
        "sequence": 1,
        "previous_checkpoint_sha256": hashlib.sha256(first_raw).hexdigest(),
        "stage": "screening",
        "stage_started_at": "2026-08-09T12:00:01Z",
        "stage_monotonic_ns": 2,
        "checkpoint_realtime": "2026-08-09T12:00:01Z",
        "checkpoint_monotonic_ns": 2,
    }
    del second["signature"]
    second["signature"] = sign_object(seed, "runtime-clock-checkpoint-v1", second)
    for sequence, raw in enumerate((first_raw, canonical_json_file(second))):
        path = root / "runtime-clock" / f"{sequence:08d}-{hashlib.sha256(raw).hexdigest()}.json"
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.write_bytes(raw)
        path.chmod(0o600)
    binding: dict[str, JsonValue] = {
        "attempt_id": attempt_id,
        "execution_commit": "b" * 40,
        "plan_binding_sha256": "c" * 64,
    }
    claim: dict[str, JsonValue] = {
        "attempt_id": attempt_id,
        "execution_commit": "b" * 40,
        "plan_binding_sha256": "c" * 64,
        "created_at": "2026-08-09T12:00:00Z",
        "key_fingerprint": hashlib.sha256(public_key_from_seed(seed)).hexdigest(),
    }
    predecessor: dict[str, JsonValue] = {
        "execution_authority_sha256": hashlib.sha256(authority_raw).hexdigest(),
    }

    with pytest.raises(RootlessContractError, match="ROOTLESS_REPORTING_CONFLICT"):
        rootless_local_operator._validate_runtime_clock_chain(
            root, seed, binding, claim, predecessor, attempt_id, "screening"
        )
