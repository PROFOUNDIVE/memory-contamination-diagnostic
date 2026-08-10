from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import importlib

from memcontam.experiment.phase12.filter_challenge.rootless_local_binding import (
    build_fake_stage_binding,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_broker import (
    build_fake_broker_for_tests,
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


def test_broker_runtime_process_dispatches_fake_slot_through_worker_fd(tmp_path: Path) -> None:
    # Given: one compiled screening slot and a fake broker kept behind the production factory seam.
    from memcontam.experiment.phase12.filter_challenge.rootless_local_execution import (
        _StaticTransport,
    )

    rootless_local_runtime = importlib.import_module(
        "memcontam.experiment.phase12.filter_challenge.rootless_local_runtime"
    )

    context = CompileContext("runtime-smoke", "screening", "1" * 64, "2" * 64, "3" * 64)
    slot = build_screening_compilation(context, load_probe_ids(ROOT)).slots[0]
    response = FakeResponse.completed(("fixture answer",))
    binding = build_fake_stage_binding(
        fixture_id=context.attempt_id,
        stage=context.stage,
        source_manifest_sha256=context.source_manifest_sha256,
        input_manifest_sha256=context.input_manifest_sha256,
        compiler_sha256=context.compiler_sha256,
        schedule_sha256="4" * 64,
        fake_scenario_sha256="5" * 64,
    )
    fake_root = tmp_path / "basetemp" / "runtime" / "tmp" / "fake-state" / context.attempt_id
    fake_root.mkdir(mode=0o700, parents=True)

    # When: the same process/socket runtime used by broker-runtime executes with fake transport.
    exit_code = rootless_local_runtime.run_stage_process(
        (slot,),
        lambda: build_fake_broker_for_tests(
            binding,
            _StaticTransport(context.attempt_id, response),
            fake_root,
        ),
    )

    # Then: the worker exits cleanly and the broker durably archives the slot.
    receipt = fake_root / f"attempts/{context.attempt_id}/screening/slots/{slot.slot_id}/call-receipt.json"
    assert exit_code == 0
    assert receipt.is_file()
