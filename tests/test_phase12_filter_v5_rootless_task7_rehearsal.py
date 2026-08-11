from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
from typing import assert_never

import pytest

from memcontam.experiment.phase12.filter_challenge import rootless_local_bootstrap_cli
from memcontam.experiment.phase12.filter_challenge.rootless_local_binding import (
    RuntimeInstallationEvidence,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_broker import (
    FakeBroker,
    acquire_runtime_lock,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_compilation import (
    load_live_stage_compilation,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_contract import (
    JsonValue,
    canonical_json_file,
    parse_canonical_object,
    public_key_from_seed,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_execution import (
    FakeResponse,
    StageCompilation,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_state import (
    read_private_file,
)

ROOT = Path(__file__).resolve().parents[1]
manifests = rootless_local_bootstrap_cli


class _Task7Transport:
    def __init__(self, answers: dict[str, str]) -> None:
        self.answers = answers
        self.calls = 0

    async def exchange(self, slot_id: str, request: bytes) -> dict[str, str | int | bool | None]:
        del request
        self.calls += 1
        body = FakeResponse.completed((self.answers[slot_id],)).body
        return {
            "schema_version": "rootless_fake_http_exchange_v1",
            "profile": "local_rootless_non_authoritative",
            "kind": "fake_http_exchange",
            "fixture_id": "live",
            "slot_id": slot_id,
            "lifecycle_marker": "streaming_body",
            "response_surfaced": True,
            "http_status": 200,
            "headers_base64": base64.b64encode(b"\0\0\0\0").decode("ascii"),
            "body_base64": base64.b64encode(body).decode("ascii"),
            "raised_exception": None,
        }


def _answers(compilation: StageCompilation) -> dict[str, str]:
    probe_manifest = parse_canonical_object(
        (ROOT / "data/phase12/filter_v5_bct_v1/probe_construction_manifest_v1.json").read_bytes()
    )
    raw_probes = probe_manifest["probes"]
    assert isinstance(raw_probes, dict)
    certificates: dict[str, dict[str, JsonValue]] = {}
    for rows in raw_probes.values():
        assert isinstance(rows, list)
        for row in rows:
            assert isinstance(row, dict) and isinstance(row["certificate"], dict)
            certificates[str(row["probe_id"])] = row["certificate"]
    answers: dict[str, str] = {}
    for slot in compilation.slots:
        certificate = certificates[slot.probe_id]
        match slot.native_stage:
            case "bot_problem_distill":
                answers[slot.slot_id] = json.dumps(
                    {
                        "key_information": "registered probe input",
                        "restrictions": "preserve the registered task contract",
                        "distilled_task": "solve the registered probe",
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            case "answer" | "bot_instantiate_solve":
                match slot.task:
                    case "game24":
                        answers[slot.slot_id] = (
                            "0" if slot.side == "challenge"
                            and slot.candidate_class in {"certified_false", "ordinary_false"}
                            else str(certificate["expression"])
                        )
                    case "math_equation_balancer":
                        answers[slot.slot_id] = (
                            "wrong" if slot.side == "challenge"
                            and slot.candidate_class in {"certified_false", "ordinary_false"}
                            else str(certificate["target"])
                        )
                    case "word_sorting":
                        words = certificate["correct_order"]
                        assert isinstance(words, list) and all(isinstance(word, str) for word in words)
                        ordered = [word for word in words if isinstance(word, str)]
                        if slot.side == "challenge" and slot.candidate_class in {
                            "certified_false", "ordinary_false",
                        }:
                            ordered.reverse()
                        answers[slot.slot_id] = " ".join(ordered)
                    case unreachable:
                        assert_never(unreachable)
            case unreachable:
                assert_never(unreachable)
    return answers


def _parse(state_home: Path, command: str, *options: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="phase12_command", required=True)
    rootless_local_bootstrap_cli.add_parser(commands)
    return parser.parse_args((
        "filter-v5-rootless", "--repo-root", os.fspath(ROOT), "--state-home",
        os.fspath(state_home), command, *options,
    ))


def _run(state_home: Path, command: str, *options: str) -> None:
    rootless_local_bootstrap_cli.run(_parse(state_home, command, *options))


def test_fresh_task7_cli_rehearsal_reaches_review_required_without_provider_egress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: frozen inputs, fresh private state, and only a test transport at the live broker seam.
    state_home = tmp_path / "state"
    state_home.mkdir(mode=0o700)
    plan = tmp_path / "plan.md"
    plan.write_bytes(b"reviewed rootless plan\n")
    descriptor = tmp_path / "plan.sha256"
    descriptor.write_text(
        f"{hashlib.sha256(plan.read_bytes()).hexdigest()}  "
        "phase12-filter-v5-rootless-local-execution.md\n", encoding="ascii",
    )
    metadata = tmp_path / "review.json"
    metadata.write_bytes(b"{}\n")
    tokenizer = tmp_path / "tokenizer"
    tokenizer.write_bytes(b"synthetic tokenizer\n")
    for path in (plan, descriptor, metadata, tokenizer):
        path.chmod(0o600)
    execution_commit = "b" * 40
    source_entry: JsonValue = {
        "role": "src-fixture", "repo_relative_path": "src/memcontam/__init__.py",
        "size_bytes": 1, "sha256": "6" * 64,
    }
    monkeypatch.setattr(
        manifests, "source_files",
        lambda _repository, _commit: ([source_entry], ["src-fixture"], ["6" * 64]),
    )
    monkeypatch.setattr(
        manifests,
        "collect_runtime_installation_evidence",
        lambda repository, tokenizer_hash: RuntimeInstallationEvidence(
            "/bound/python", "3.11.15", "26.1.2", f"{repository}/src/memcontam/__init__.py",
            0o755, "1" * 64, "2" * 64, ("3" * 64,), "4" * 64, "5" * 64,
            "0.13.0", tokenizer_hash,
        ),
    )
    roles = (
        "phase13-theory", "phase13-baseline-filter",
        "phase13-contamination-protocol", "phase13-experiment-design",
    )
    monkeypatch.setattr(
        manifests, "observe_external_authorities",
        lambda _decoding: [{"role": role} for role in roles],
    )

    def cache(source: Path, cache_directory: Path, **_kwargs: int) -> Path:
        target = cache_directory / "tokenizer"
        target.write_bytes(source.read_bytes())
        target.chmod(0o600)
        return target

    monkeypatch.setattr(rootless_local_bootstrap_cli, "cache_tokenizer_source", cache)
    qa = tmp_path / "qa/pre-egress"
    qa.mkdir(mode=0o700, parents=True)
    (qa / "execution-anchor.json").write_bytes(canonical_json_file({
        "execution_commit": execution_commit,
    }))
    monkeypatch.setattr(rootless_local_bootstrap_cli, "qa_root", lambda _repository: qa.parent)
    transports: list[_Task7Transport] = []

    def fake_live_broker(binding: dict[str, JsonValue], root: Path, repository: Path) -> FakeBroker:
        seed = read_private_file(root / "keys/ed25519-private.key")
        compilation = load_live_stage_compilation(
            binding, root, repository, public_key_from_seed(seed)
        )
        transport = _Task7Transport(_answers(compilation))
        transports.append(transport)
        return FakeBroker(
            binding, transport, root, seed, acquire_runtime_lock(root / "runtime.lock"),
            "live", repository,
        )

    monkeypatch.setattr(rootless_local_bootstrap_cli, "build_live_broker", fake_live_broker)

    # When: every production administrative command and both stage runtimes execute in order.
    _run(
        state_home, "init-state", "--attempt-id", "task7-rehearsal",
        "--plan-source", os.fspath(plan), "--plan-descriptor", os.fspath(descriptor),
        "--review-metadata", os.fspath(metadata), "--tokenizer-source", os.fspath(tokenizer),
        "--execution-commit", execution_commit,
    )
    for index in (1, 2):
        _run(
            state_home, "acknowledge-plan", "--attempt-id", "task7-rehearsal",
            "--set-id", "plan", "--operator-index", str(index),
            "--operator-label", f"operator-{index}",
        )
    _run(state_home, "acknowledge-rate-capability", "--attempt-id", "task7-rehearsal",
         "--provider-account-label", "provider", "--rpm-limit", "6", "--tpm-limit", "30000")
    for stage in ("screening", "bct"):
        _run(state_home, f"bind-{stage}", "--attempt-id", "task7-rehearsal")
        for index in (1, 2):
            _run(
                state_home, "acknowledge-stage", "--attempt-id", "task7-rehearsal",
                "--stage", stage, "--set-id", stage, "--operator-index", str(index),
                "--operator-label", f"operator-{index}",
            )
        _run(
            state_home, "build-execution-authority", "--attempt-id", "task7-rehearsal",
            "--stage", stage, "--plan-set-id", "plan", "--stage-set-id", stage,
        )
        root = state_home / "memcontam/phase12-filter-v5-rootless-local"
        _run(
            state_home, "broker-runtime", "--attempt-id", "task7-rehearsal", "--stage", stage,
            "--authority", os.fspath(root / f"authorities/task7-rehearsal/{stage}.json"),
            "--worker-fd", "3",
        )
        if stage == "screening":
            _run(state_home, "derive-freeze-b", "--attempt-id", "task7-rehearsal")

    # Then: all four families and signed review-required closure exist with no HTTP provider.
    root = state_home / "memcontam/phase12-filter-v5-rootless-local"
    final = parse_canonical_object((root / "terminals/task7-rehearsal/final.json").read_bytes())
    families = tuple((root / "attempts/task7-rehearsal/bct/evidence/families").glob("BCT-FV5-*.json"))
    assert [transport.calls for transport in transports] == [90, 480]
    assert final["status"] == "review_required"
    assert final["reason_code"] == "BCT_COMPLETED_REVIEW_REQUIRED"
    assert len(families) == 4
