from __future__ import annotations

import hashlib
import importlib
import json
import os
from argparse import Namespace
from pathlib import Path
import subprocess
import sys

import anyio
import pytest
from memcontam.experiment.phase12.filter_challenge.rootless_local_binding import (
    build_fake_stage_binding,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_broker import (
    build_fake_broker_for_tests,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_contract import (
    canonical_json_file,
    parse_canonical_object,
    public_key_from_seed,
    sign_object,
    verify_object_signature,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_execution import (
    CompileContext,
    FakeResponse,
    build_screening_compilation,
    load_probe_ids,
)


ROOT = Path(__file__).resolve().parents[1]
LOCK_HELPER = ROOT / "scripts" / "open_phase12_filter_v5_rootless_orchestration_lock.py"


def _run(qa_root: Path) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        (sys.executable, "-B", "-I", "-S", os.fspath(LOCK_HELPER), "--qa-root", os.fspath(qa_root)),
        check=False,
        capture_output=True,
        env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"},
        close_fds=True,
    )


def test_lock_helper_creates_and_reopens_private_lock(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    qa_root = repository / "runs" / "phase12-filter-v5-rootless-qa"
    qa_root.mkdir(parents=True)
    for path in (repository, repository / "runs", qa_root):
        path.chmod(0o700)

    first = _run(qa_root)
    second = _run(qa_root)

    lock = qa_root / "orchestration.lock"
    assert first.returncode == second.returncode == 0
    assert first.stdout == first.stderr == second.stdout == second.stderr == b""
    assert lock.stat().st_mode & 0o777 == 0o600
    assert lock.stat().st_nlink == 1


def test_lock_helper_rejects_symlink_without_touching_target(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    qa_root = repository / "runs" / "phase12-filter-v5-rootless-qa"
    qa_root.mkdir(parents=True)
    for path in (repository, repository / "runs", qa_root):
        path.chmod(0o700)
    target = tmp_path / "target"
    target.write_bytes(b"unchanged")
    (qa_root / "orchestration.lock").symlink_to(target)

    completed = _run(qa_root)

    assert completed.returncode == 64
    assert completed.stdout == completed.stderr == b""
    assert target.read_bytes() == b"unchanged"


def test_lock_helper_rejects_wrong_mode_and_multiple_links(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    qa_root = repository / "runs" / "phase12-filter-v5-rootless-qa"
    qa_root.mkdir(parents=True)
    for path in (repository, repository / "runs", qa_root):
        path.chmod(0o700)
    lock = qa_root / "orchestration.lock"
    lock.write_bytes(b"")
    lock.chmod(0o644)

    wrong_mode = _run(qa_root)
    lock.chmod(0o600)
    os.link(lock, qa_root / "second-link")
    multiple_links = _run(qa_root)

    assert wrong_mode.returncode == multiple_links.returncode == 64
    assert wrong_mode.stdout == wrong_mode.stderr == multiple_links.stdout == multiple_links.stderr == b""


def test_runtime_closes_every_inherited_descriptor_discovered_under_proc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from memcontam.experiment.phase12.filter_challenge import rootless_local_runtime

    # Given: inherited descriptors include an arbitrary descriptor beyond a fixed range.
    closed: list[int] = []
    monkeypatch.setattr(os, "listdir", lambda path: ["0", "1", "2", "3", "257", "invalid"])
    monkeypatch.setattr(os, "close", closed.append)

    # When: the fork child applies its capability descriptor allowlist.
    rootless_local_runtime._close_inherited_descriptors()

    # Then: only standard IO and the capability survive independent of descriptor number.
    assert closed == [257]


def test_broker_runtime_process_dispatches_fake_slot_through_worker_fd(tmp_path: Path) -> None:
    # Given: a complete screening compilation and a fake broker kept behind the production factory seam.
    from memcontam.experiment.phase12.filter_challenge.rootless_local_execution import (
        _StaticTransport,
    )

    rootless_local_runtime = importlib.import_module(
        "memcontam.experiment.phase12.filter_challenge.rootless_local_runtime"
    )

    context = CompileContext("runtime-smoke", "screening", "1" * 64, "2" * 64, "3" * 64)
    compilation = build_screening_compilation(context, load_probe_ids(ROOT))
    response = FakeResponse.completed(
        (
            '{"distilled_task":"solve the registered probe",'
            '"key_information":"registered probe input",'
            '"restrictions":"preserve the registered task contract"}',
        )
    )
    binding = build_fake_stage_binding(
        fixture_id=context.attempt_id,
        stage=context.stage,
        source_manifest_sha256=context.source_manifest_sha256,
        input_manifest_sha256=context.input_manifest_sha256,
        compiler_sha256=context.compiler_sha256,
        schedule_sha256=compilation.ordered_slot_root_sha256,
        fake_scenario_sha256="5" * 64,
    )
    fake_root = tmp_path / "basetemp" / "runtime" / "tmp" / "fake-state" / context.attempt_id
    fake_root.mkdir(mode=0o700, parents=True)

    # When: the same process/socket runtime used by broker-runtime executes with fake transport.
    exit_code = rootless_local_runtime.run_stage_process(
        compilation.slots,
        lambda: build_fake_broker_for_tests(
            binding,
            _StaticTransport(context.attempt_id, response),
            fake_root,
        ),
    )

    # Then: every slot is durable before the process closes the authoritative receipt set.
    stage_root = fake_root / f"attempts/{context.attempt_id}/screening"
    assert exit_code == 0
    receipt_leaves: list[bytes] = []
    receipt_created_at: list[str] = []
    issued_count = 0
    for slot in compilation.slots:
        slot_root = stage_root / "slots" / slot.slot_id
        archive = parse_canonical_object((slot_root / "archive-manifest.json").read_bytes())
        receipt_raw = (slot_root / "call-receipt.json").read_bytes()
        receipt = parse_canonical_object(receipt_raw)
        assert (archive["schema_version"], archive["attempt_id"], archive["stage"], archive["slot_id"]) == (
            "rootless_raw_archive_manifest_v1",
            context.attempt_id,
            "screening",
            slot.slot_id,
        )
        assert (receipt["schema_version"], receipt["attempt_id"], receipt["stage"], receipt["slot_id"]) == (
            "rootless_local_call_receipt_v1",
            context.attempt_id,
            "screening",
            slot.slot_id,
        )
        assert isinstance(receipt["created_at"], str)
        assert isinstance(receipt["issued"], bool)
        receipt_created_at.append(receipt["created_at"])
        issued_count += receipt["issued"]
        receipt_leaves.append(
            hashlib.sha256(
                slot.slot_id.encode("utf-8")
                + b"\x00"
                + hashlib.sha256(receipt_raw).digest()
            ).digest()
        )

    receipt_manifest_path = stage_root / "receipt-manifest.json"
    assert receipt_manifest_path.is_file(), "process stage must close its durable receipt set"
    receipt_manifest = parse_canonical_object(receipt_manifest_path.read_bytes())
    assert set(receipt_manifest) == {
        "schema_version",
        "profile",
        "kind",
        "attempt_id",
        "stage",
        "stage_binding_sha256",
        "schedule_sha256",
        "ordered_receipt_root_sha256",
        "registered_slot_count",
        "accounted_slot_count",
        "issued_count",
        "not_issued_count",
        "created_at",
        "key_fingerprint",
        "signature",
    }
    assert receipt_manifest["schema_version"] == "rootless_receipt_manifest_v1"
    assert receipt_manifest["profile"] == "local_rootless_non_authoritative"
    assert receipt_manifest["kind"] == "receipt_manifest"
    assert receipt_manifest["attempt_id"] == context.attempt_id
    assert receipt_manifest["stage"] == "screening"
    assert receipt_manifest["stage_binding_sha256"] == hashlib.sha256(
        canonical_json_file(binding)
    ).hexdigest()
    assert receipt_manifest["schedule_sha256"] == compilation.ordered_slot_root_sha256
    assert receipt_manifest["ordered_receipt_root_sha256"] == hashlib.sha256(
        b"".join(receipt_leaves)
    ).hexdigest()
    assert receipt_manifest["registered_slot_count"] == len(compilation.slots)
    assert receipt_manifest["accounted_slot_count"] == len(receipt_leaves)
    assert receipt_manifest["issued_count"] == issued_count
    assert receipt_manifest["not_issued_count"] == len(receipt_leaves) - issued_count
    assert receipt_manifest["created_at"] == max(receipt_created_at)
    seed = hashlib.sha256(f"rootless-fixture:{context.attempt_id}".encode()).digest()
    public_key = public_key_from_seed(seed)
    assert receipt_manifest["key_fingerprint"] == hashlib.sha256(public_key).hexdigest()
    signature = receipt_manifest.pop("signature")
    assert isinstance(signature, str)
    verify_object_signature(public_key, "receipt-manifest-v1", receipt_manifest, signature)

    stage_terminal_path = fake_root / f"terminals/{context.attempt_id}/screening.json"
    assert stage_terminal_path.is_file(), "process stage must close after its receipt manifest"
    stage_terminal = parse_canonical_object(stage_terminal_path.read_bytes())
    assert stage_terminal["schema_version"] == "rootless_stage_terminal_v1"
    assert stage_terminal["status"] == "not_estimable"
    assert stage_terminal["reason_code"] == "SCREENING_NOT_ESTIMABLE"
    assert stage_terminal["receipt_manifest_sha256"] == hashlib.sha256(
        receipt_manifest_path.read_bytes()
    ).hexdigest()
    assert stage_terminal["freeze_b_sha256"] is None
    assert stage_terminal["bct_result_manifest_sha256"] is None
    terminal_signature = stage_terminal.pop("signature")
    assert isinstance(terminal_signature, str)
    verify_object_signature(public_key, "stage-terminal-v1", stage_terminal, terminal_signature)


def test_runtime_fills_five_ready_provider_slots_concurrently(tmp_path: Path) -> None:
    # Given: five independent screening slots and a transport that exposes overlap.
    from memcontam.experiment.phase12.filter_challenge.rootless_local_execution import (
        _StaticTransport,
    )
    from memcontam.experiment.phase12.filter_challenge.rootless_local_runtime import (
        run_stage_process,
    )

    context = CompileContext("runtime-concurrency", "screening", "1" * 64, "2" * 64, "3" * 64)
    compilation = build_screening_compilation(context, load_probe_ids(ROOT))
    slots = tuple(slot for slot in compilation.slots if not slot.predecessor_slot_ids)[:5]
    response = FakeResponse.completed(("fixture answer",))
    delegate = _StaticTransport(context.attempt_id, response)

    class ConcurrentTransport:
        active = 0
        maximum = 0

        async def exchange(self, slot_id: str, request: bytes):
            self.active += 1
            self.maximum = max(self.maximum, self.active)
            try:
                await anyio.sleep(0.05)
                return await delegate.exchange(slot_id, request)
            finally:
                self.active -= 1

    transport = ConcurrentTransport()
    binding = build_fake_stage_binding(
        fixture_id=context.attempt_id,
        stage="screening",
        source_manifest_sha256=context.source_manifest_sha256,
        input_manifest_sha256=context.input_manifest_sha256,
        compiler_sha256=context.compiler_sha256,
        schedule_sha256=compilation.ordered_slot_root_sha256,
        fake_scenario_sha256=hashlib.sha256(response.body).hexdigest(),
    )
    fake_root = tmp_path / "basetemp/concurrency/tmp/fake-state" / context.attempt_id
    fake_root.mkdir(mode=0o700, parents=True)

    # When: the fork/FD3 runtime advances the ready set.
    exit_code = run_stage_process(
        slots,
        lambda: build_fake_broker_for_tests(binding, transport, fake_root),
    )

    # Then: all five legal workers are occupied without changing wire variants.
    assert exit_code == 0
    assert transport.maximum == 5


def test_runtime_channel_loss_seals_interrupted_terminal(tmp_path: Path) -> None:
    # Given: one ready slot whose provider channel disappears after reservation.
    from memcontam.experiment.phase12.filter_challenge.rootless_local_runtime import (
        run_stage_process,
    )

    context = CompileContext("runtime-channel-loss", "screening", "1" * 64, "2" * 64, "3" * 64)
    compilation = build_screening_compilation(context, load_probe_ids(ROOT))
    slot = next(item for item in compilation.slots if not item.predecessor_slot_ids)
    response = FakeResponse.completed(("unused",))

    class ChannelLossTransport:
        async def exchange(self, slot_id: str, request: bytes):
            del slot_id, request
            raise EOFError("provider channel lost")

    binding = build_fake_stage_binding(
        fixture_id=context.attempt_id,
        stage="screening",
        source_manifest_sha256=context.source_manifest_sha256,
        input_manifest_sha256=context.input_manifest_sha256,
        compiler_sha256=context.compiler_sha256,
        schedule_sha256=compilation.ordered_slot_root_sha256,
        fake_scenario_sha256=hashlib.sha256(response.body).hexdigest(),
    )
    fake_root = tmp_path / "basetemp/channel-loss/tmp/fake-state" / context.attempt_id
    fake_root.mkdir(mode=0o700, parents=True)

    # When: the parent broker loses that channel while the FD3 worker is waiting.
    exit_code = run_stage_process(
        (slot,),
        lambda: build_fake_broker_for_tests(binding, ChannelLossTransport(), fake_root),
    )

    # Then: zero inferred receipts are sealed and the attempt is durably interrupted.
    manifest = parse_canonical_object(
        (fake_root / f"attempts/{context.attempt_id}/screening/receipt-manifest.json").read_bytes()
    )
    terminal = parse_canonical_object(
        (fake_root / f"terminals/{context.attempt_id}/screening.json").read_bytes()
    )
    assert exit_code == 69
    assert manifest["accounted_slot_count"] == 0
    assert terminal["status"] == "interrupted"
    assert terminal["reason_code"] == "ROOTLESS_INTERRUPTED_UNCLEAN"
    assert (fake_root / f"terminals/{context.attempt_id}/final.json").is_file()


def test_continue_after_screening_rejects_valid_ledger_fork_without_mutation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Given: two independently valid signed heads claim the same authoritative sequence.
    from memcontam.experiment.phase12.filter_challenge.rootless_local_bootstrap_cli import (
        _administrative,
    )

    state_home = tmp_path / "state-home"
    root = state_home / "memcontam/phase12-filter-v5-rootless-local"
    heads = root / "ledger/global/heads"
    records = root / "ledger/global/records"
    keys = root / "keys"
    terminal_root = root / "terminals/forked"
    heads.mkdir(mode=0o700, parents=True)
    records.mkdir(mode=0o700, parents=True)
    keys.mkdir(mode=0o700, parents=True)
    terminal_root.mkdir(mode=0o700, parents=True)
    for path in (state_home, state_home / "memcontam", root, root / "ledger", root / "ledger/global"):
        path.chmod(0o700)
    (root / "runtime.lock").write_bytes(b"")
    (root / "runtime.lock").chmod(0o600)
    (terminal_root / "screening.json").write_bytes(
        canonical_json_file({"status": "completed_estimable"})
    )
    seed = hashlib.sha256(b"valid-ledger-fork").digest()
    (keys / "ed25519-private.key").write_bytes(seed)
    (keys / "ed25519-private.key").chmod(0o600)
    fingerprint = hashlib.sha256(public_key_from_seed(seed)).hexdigest()
    common = {
        "schema_version": "rootless_ledger_head_v1",
        "profile": "local_rootless_non_authoritative",
        "kind": "ledger_head",
        "attempt_id": "forked",
        "sequence": 0,
        "previous_head_sha256": None,
        "cumulative_issued": 1,
        "cumulative_not_issued": 0,
        "cumulative_settled_nanousd": 0,
        "cumulative_retained_nanousd": 16_640_000,
        "screening_settled_nanousd": 0,
        "bct_settled_nanousd": 0,
        "issued_at": "2026-08-10T00:00:00Z",
        "key_fingerprint": fingerprint,
    }
    record_hashes = []
    for index in (1, 2):
        record = {
            "schema_version": "rootless_ledger_record_v1",
            "profile": "local_rootless_non_authoritative",
            "kind": "ledger_record",
            "record_kind": "reservation",
            "sequence": 0,
            "previous_record_sha256": None,
            "attempt_id": "forked",
            "stage": "screening",
            "created_at": "2026-08-10T00:00:00Z",
            "key_fingerprint": fingerprint,
            "slot_id": f"slot-fork-{index}",
            "idempotency_key": f"idempotency-fork-{index}",
            "compiler_sha256": "3" * 64,
            "static_input_sha256": "4" * 64,
            "predecessor_receipt_sha256": None,
            "compile_status": "compiled",
            "request_sha256": str(index) * 64,
            "request_bytes": 2,
            "compiled_input_tokens": 1,
            "reserved_input_tokens": 4096,
            "reserved_output_tokens": 640,
            "reserved_nanousd": 16_640_000,
        }
        record["signature"] = sign_object(seed, "ledger-record-v1", record)
        raw = canonical_json_file(record)
        digest = hashlib.sha256(raw).hexdigest()
        record_path = records / f"000000-{digest}.json"
        record_path.write_bytes(raw)
        record_path.chmod(0o600)
        record_hashes.append(digest)
    for record_sha256 in record_hashes:
        head = {**common, "record_sha256": record_sha256}
        head["signature"] = sign_object(seed, "ledger-head-v1", head)
        raw = canonical_json_file(head)
        head_path = heads / f"000000-{hashlib.sha256(raw).hexdigest()}.json"
        head_path.write_bytes(raw)
        head_path.chmod(0o600)
    before = {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }

    # When: continuation inspects the otherwise completed screening lineage.
    arguments = Namespace(
        rootless_command="continue-after-screening",
        repo_root=tmp_path / "repo",
        state_home=state_home,
        attempt_id="forked",
    )
    with pytest.raises(SystemExit) as raised:
        _administrative(arguments)

    # Then: authoritative ambiguity returns 67 and writes no sealing artifacts.
    assert raised.value.code == 67
    assert json.loads(capsys.readouterr().out)["exit_code"] == 67
    assert {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    } == before
