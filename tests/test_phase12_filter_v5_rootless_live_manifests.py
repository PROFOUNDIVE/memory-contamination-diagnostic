from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import pytest

from memcontam.experiment.phase12.filter_challenge import (
    rootless_local_bootstrap_cli as rootless_local_manifests,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_binding import (
    RuntimeInstallationEvidence,
    build_stage_binding,
    validate_rootless_configs,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_compilation import (
    load_live_stage_compilation,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_contract import (
    JsonValue,
    RootlessContractError,
    canonical_json_file,
    parse_canonical_object,
    public_key_from_seed,
    sign_object,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_execution import load_probe_ids

ROOT = Path(__file__).resolve().parents[1]


def _parse(state_home: Path, command: str, *options: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="phase12_command", required=True)
    rootless_local_manifests.add_parser(commands)
    return parser.parse_args((
        "filter-v5-rootless", "--repo-root", os.fspath(ROOT), "--state-home",
        os.fspath(state_home), command, *options,
    ))


def _private_state(tmp_path: Path, attempt_id: str) -> tuple[Path, bytes]:
    root = tmp_path / "state/memcontam/phase12-filter-v5-rootless-local"
    seed = bytes(range(32))
    for relative in ("keys", "tokenizer/cache", "manifests"):
        (root / relative).mkdir(mode=0o700, parents=True, exist_ok=True)
    (root / "keys/ed25519-private.key").write_bytes(seed)
    (root / "tokenizer/cache/tokenizer").write_bytes(b"synthetic tokenizer\n")
    for path in (root / "keys/ed25519-private.key", root / "tokenizer/cache/tokenizer"):
        path.chmod(0o600)
    return root, seed


def _runtime_evidence(repository: Path, tokenizer_hash: str) -> RuntimeInstallationEvidence:
    return RuntimeInstallationEvidence(
        "/bound/python", "3.11.15", "26.1.2", f"{repository}/src/memcontam/__init__.py",
        0o755, "1" * 64, "2" * 64, ("3" * 64,), "4" * 64, "5" * 64,
        "0.13.0", tokenizer_hash,
    )


def test_fresh_manifest_materialization_loads_production_screening_compilation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: fresh private state and deterministic runtime/external observations.
    attempt = "fresh-live-manifests"
    root, seed = _private_state(tmp_path, attempt)
    source_entry: JsonValue = {
        "role": "src-fixture", "repo_relative_path": "src/memcontam/__init__.py",
        "size_bytes": 1, "sha256": "6" * 64,
    }
    monkeypatch.setattr(
        rootless_local_manifests, "source_files",
        lambda _repository, _commit: ([source_entry], ["src-fixture"], ["6" * 64]),
    )
    monkeypatch.setattr(
        rootless_local_manifests, "collect_runtime_installation_evidence",
        _runtime_evidence,
    )
    roles = (
        "phase13-theory", "phase13-baseline-filter",
        "phase13-contamination-protocol", "phase13-experiment-design",
    )
    monkeypatch.setattr(
        rootless_local_manifests, "observe_external_authorities",
        lambda _decoding: [{"role": role} for role in roles],
    )

    # When: Task-7 initialization materializes the prerequisite authority set.
    rootless_local_manifests.materialize_screening_prerequisites(
        ROOT, root, attempt, "b" * 40, "2026-08-11T12:00:00Z"
    )

    # Then: the production loader reconstructs the exact 90-slot screening schedule.
    manifests = root / "manifests" / attempt
    names = ("source", "runtime", "input", "compiler", "screening-schedule")
    hashes = {name: hashlib.sha256((manifests / f"{name}.json").read_bytes()).hexdigest() for name in names}
    configs = validate_rootless_configs(ROOT)
    binding = build_stage_binding(
        attempt_id=attempt, stage="screening", plan_binding_sha256="7" * 64,
        trusted_base_commit="a" * 40, execution_commit="b" * 40,
        decoding_authority_sha256=configs["decoding_authority"],
        rate_card_sha256=configs["rate_card"], source_manifest_sha256=hashes["source"],
        runtime_manifest_sha256=hashes["runtime"], input_manifest_sha256=hashes["input"],
        compiler_sha256=hashes["compiler"], schedule_sha256=hashes["screening-schedule"],
        registered_slots=90, stage_cap_nanousd=2_000_000_000,
        created_at="2026-08-11T12:00:00Z",
    )
    compilation = load_live_stage_compilation(binding, root, ROOT, public_key_from_seed(seed))
    assert len(compilation.slots) == 90
    assert not (manifests / "bct-schedule.json").exists()


def test_valid_screening_freeze_materializes_signed_480_slot_bct_schedule(
    tmp_path: Path,
) -> None:
    # Given: signed estimable Screening and Freeze-B lineage over two probes per task.
    attempt = "positive-bct-schedule"
    root, seed = _private_state(tmp_path, attempt)
    manifests = root / "manifests" / attempt
    manifests.mkdir(mode=0o700)
    for name in ("source", "input", "compiler"):
        (manifests / f"{name}.json").write_bytes(b"{}\n")
        (manifests / f"{name}.json").chmod(0o600)
    terminal: dict[str, JsonValue] = {
        "attempt_id": attempt, "status": "completed_estimable",
        "reason_code": "SCREENING_ESTIMABLE",
    }
    terminal["signature"] = sign_object(seed, "stage-terminal-v1", terminal)
    terminal_raw = canonical_json_file(terminal)
    probes = load_probe_ids(ROOT)
    freeze: dict[str, JsonValue] = {
        "attempt_id": attempt,
        "screening_stage_terminal_sha256": hashlib.sha256(terminal_raw).hexdigest(),
        **{f"selected_{task}_probe_ids": list(values[:2]) for task, values in probes.items()},
    }
    freeze["signature"] = sign_object(seed, "freeze-b-v1", freeze)
    for relative, raw in (
        (f"terminals/{attempt}/screening.json", terminal_raw),
        (f"freeze/{attempt}/freeze_b.json", canonical_json_file(freeze)),
    ):
        path = root / relative
        path.parent.mkdir(mode=0o700, parents=True)
        path.write_bytes(raw)
        path.chmod(0o600)

    # When: production BCT binding preparation consumes that lineage.
    rootless_local_manifests.materialize_bct_schedule(
        ROOT, root, attempt, "2026-08-11T12:01:00Z"
    )

    # Then: the immutable signed schedule carries the frozen 480-slot authority.
    schedule = parse_canonical_object((manifests / "bct-schedule.json").read_bytes())
    assert schedule["slot_count"] == 480
    assert isinstance(schedule["signature"], str)


def test_bct_schedule_fails_closed_without_screening_or_freeze_authority(
    tmp_path: Path,
) -> None:
    # Given: fresh state has no Screening terminal or Freeze-B authority.
    attempt = "invalid-bct-lineage"
    root, _seed = _private_state(tmp_path, attempt)

    # When/Then: BCT preparation blocks before creating schedule or binding authority.
    with pytest.raises(RootlessContractError, match="ROOTLESS_BINDING_INVALID"):
        rootless_local_manifests.materialize_bct_schedule(
            ROOT, root, attempt, "2026-08-11T12:01:00Z"
        )
    assert not (root / f"manifests/{attempt}/bct-schedule.json").exists()


def test_init_state_materializes_screening_prerequisite_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: reviewed plan inputs and a final execution commit for a fresh attempt.
    state_home = tmp_path / "state"
    state_home.mkdir(mode=0o700)
    plan = tmp_path / "plan.md"
    plan.write_bytes(b"reviewed plan\n")
    descriptor = tmp_path / "plan.sha256"
    descriptor.write_text(
        f"{hashlib.sha256(plan.read_bytes()).hexdigest()}  "
        "phase12-filter-v5-rootless-local-execution.md\n", encoding="ascii",
    )
    metadata = tmp_path / "review.json"
    metadata.write_bytes(b"{}\n")
    for path in (plan, descriptor, metadata):
        path.chmod(0o600)
    calls: list[tuple[Path, Path, str, str]] = []
    monkeypatch.setattr(
        rootless_local_manifests,
        "materialize_screening_prerequisites",
        lambda repository, state, attempt, commit, _created_at: calls.append(
            (repository, state, attempt, commit)
        ),
    )
    monkeypatch.setattr(rootless_local_manifests, "_status", lambda *_args: None)

    # When: the production administrative initializer completes.
    rootless_local_manifests.run(_parse(
        state_home, "init-state", "--attempt-id", "fresh-task7",
        "--plan-source", os.fspath(plan), "--plan-descriptor", os.fspath(descriptor),
        "--review-metadata", os.fspath(metadata), "--execution-commit", "b" * 40,
    ))

    # Then: manifest orchestration receives the new immutable state and commit identity.
    assert calls == [(
        ROOT, state_home / "memcontam/phase12-filter-v5-rootless-local",
        "fresh-task7", "b" * 40,
    )]
