from __future__ import annotations

# allow: SIZE_OK — the exact seven-role command and evidence order is one closed authority.

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Final, Literal

import anyio

from memcontam.experiment.phase12.filter_challenge.registry_calibration import TASKS
from memcontam.experiment.phase12.filter_challenge.rootless_local_acknowledgement import (
    create_skip_receipt,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_contract import JsonValue
from memcontam.experiment.phase12.filter_challenge.rootless_local_execution import (
    CompileContext,
    FakeResponse,
    build_bct_compilation,
    build_screening_compilation,
    execute_fake_stage,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_operator import (
    PROFILE,
    record_t7,
    write_anchor,
    write_new_or_same,
)


QA_REL: Final = Path("runs/phase12-filter-v5-rootless-qa")
FINAL_REL: Final = QA_REL / "final"
PROCESS_RACES: Final = "tests/test_phase12_filter_v5_rootless_process_races.py"
PRE_EGRESS_PATHS: Final = (
    ("execution-anchor", f"{QA_REL.as_posix()}/pre-egress/execution-anchor.json"),
    ("pre-f1-plan-compliance", f"{QA_REL.as_posix()}/pre-egress/f1-plan-compliance.json"),
    ("pre-f2-broker-security", f"{QA_REL.as_posix()}/pre-egress/f2-broker-security.json"),
    ("pre-f3-cli-rehearsal", f"{QA_REL.as_posix()}/pre-egress/f3-cli-rehearsal.json"),
)


@dataclass(frozen=True, slots=True)
class SentinelSpec:
    role: str
    module: Literal["pytest", "ruff", "memcontam.cli"]
    target: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FixtureLineage:
    outcome: Literal["paid_attempt", "zero_call_skip"]
    source: Path
    locator: Path
    t7: Path


@dataclass(frozen=True, slots=True)
class FixtureSet:
    paid: FixtureLineage
    skipped: FixtureLineage


@dataclass(frozen=True, slots=True)
class LineageValidation:
    outcome: Literal["paid_attempt", "zero_call_skip"]
    provider_calls_issued: int


@dataclass(frozen=True, slots=True)
class SentinelResult:
    role: str
    exit_code: int
    stdout_sha256: str
    stderr_sha256: str


SENTINEL_SPECS: Final = (
    SentinelSpec("f1-pytest", "pytest", (
        "tests/test_phase12_filter_v5_rootless_legacy_fence.py", "tests/test_phase12_filter_v5_rootless_binding.py",
        "tests/test_phase12_filter_v5_rootless_external_authority.py", "tests/test_phase12_filter_v5_rootless_firewall.py",
        "tests/test_phase12_filter_v5_rootless_offline_qa.py", "-q")),
    SentinelSpec("f2-pytest", "pytest", (
        "tests/test_phase12_filter_v5_rootless_broker.py", "tests/test_phase12_filter_v5_rootless_ledger.py",
        "tests/test_phase12_filter_v5_rootless_offline_qa.py", "-q")),
    SentinelSpec("f3-pytest", "pytest", (
        "tests/test_phase12_filter_v5_rootless_cli.py", "tests/test_phase12_filter_v5_rootless_execution.py",
        "tests/test_phase12_filter_v5_rootless_post_bct.py", "tests/test_phase12_filter_v5_rootless_offline_qa.py", "-q")),
    SentinelSpec("f4-rootless-pytest", "pytest", ("<expanded-rootless-tests>",)),
    SentinelSpec("f4-ruff", "ruff", ("check", "--no-cache", "src", "tests", "scripts")),
    SentinelSpec("f4-validate-config", "memcontam.cli", ("validate-config", "configs/pilot_multitask_replay.yaml")),
    SentinelSpec("f4-replay-pytest", "pytest", (
        "tests/test_task_verifiers.py", "tests/test_cli_run.py", "tests/test_contamination_catalog.py",
        "tests/test_openai_compatible_client.py", "tests/test_aggregate.py", "-q")),
)


def f4_rootless_targets(repository: Path) -> tuple[str, ...]:
    rootless = sorted(
        (path.relative_to(repository).as_posix() for path in repository.glob("tests/test_phase12_filter_v5_*.py")
         if path.relative_to(repository).as_posix() != PROCESS_RACES),
        key=str.encode,
    )
    return (*rootless, "tests/test_phase12_canonical_configs.py", "tests/test_phase12_docs_scope.py",
            "tests/test_docs_scope.py", "tests/test_phase12_claim_scope.py",
            "tests/test_phase12_scientific_admission.py", "-q")


def write_json(path: Path, value: dict[str, JsonValue]) -> str:
    return write_new_or_same(path, value)


def evidence(repository: Path, role: str, path: Path) -> dict[str, JsonValue]:
    raw = path.read_bytes()
    return {"role": role, "repo_relative_path": path.relative_to(repository).as_posix(),
            "size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def evidence_envelope(wave_role: str, inputs: tuple[dict[str, JsonValue], ...],
                      results: tuple[SentinelResult, ...] = ()) -> dict[str, JsonValue]:
    return {
        "schema_version": "rootless_final_wave_evidence_v1", "profile": PROFILE,
        "wave_role": wave_role, "outcome": "passed", "provider_calls_before": 0,
        "provider_calls_after": 0, "ordered_input_evidence": list(inputs),
        "sentinel_results": [
            {"role": result.role, "exit_code": result.exit_code,
             "stdout_sha256": result.stdout_sha256, "stderr_sha256": result.stderr_sha256}
            for result in results
        ],
    }


def _fixture_repository(root: Path, source: dict[str, JsonValue], *, paid: bool,
                        state_home: Path | None = None) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    if paid:
        write_json(root / "docs/evidence/phase12-filter-v5-rootless-local/rehearsal-publication.json", source)
    else:
        write_json(root / QA_REL / "pre-egress/zero-call-skip.json", source)
    record_t7(root, state_home)
    return (
        json.loads((root / QA_REL / "f3-state-locator.json").read_bytes()),
        json.loads((root / QA_REL / "t7-real-attempt.json").read_bytes()),
    )


def build_synthetic_task7_fixtures(repository: Path, *, execution_commit: str,
                                   created_at: str) -> FixtureSet:
    qa = repository / QA_REL
    fixtures = qa / "final/fixtures"
    scratch = qa / "basetemp/final-fixtures"
    scratch.mkdir(mode=0o700, parents=True, exist_ok=True)
    probes = {task: tuple(f"{task}-probe-{index}" for index in range(6)) for task in TASKS}
    selected = {task: values[:2] for task, values in probes.items()}
    screening_context = CompileContext(
        "final-wave-paid-fixture", "screening", "1" * 64, "2" * 64, "3" * 64
    )
    bct_context = CompileContext(
        "final-wave-paid-fixture", "bct", "1" * 64, "2" * 64, "3" * 64
    )
    screening = build_screening_compilation(screening_context, probes).slots[:1]
    bct = build_bct_compilation(bct_context, selected).slots[:1]
    response = FakeResponse.completed(("fixture response",))
    screening_result = anyio.run(execute_fake_stage, screening, response, scratch / "execution-screening")
    bct_result = anyio.run(execute_fake_stage, bct, response, scratch / "execution-bct")
    paid_source: dict[str, JsonValue] = {
        "schema_version": "rootless_synthetic_publication_fixture_v1", "profile": PROFILE,
        "kind": "publication_receipt", "attempt_id": "final-wave-paid-fixture",
        "execution_commit": execution_commit, "plan_binding_sha256": "4" * 64,
        "final_terminal_sha256": "5" * 64, "screening_terminal_sha256": "6" * 64,
        "bct_terminal_sha256": "7" * 64, "bct_result_manifest_sha256": "8" * 64,
        "state_inventory_sha256": "9" * 64, "published_at": created_at,
        "transport_mode": "fake", "terminal": "LOCAL_ROOTLESS_BCT_REVIEW_REQUIRED",
        "provider_calls_issued": screening_result.provider_calls_issued + bct_result.provider_calls_issued,
    }
    skip_source = create_skip_receipt(
        reason="ROOTLESS_MISSING_SECRET", missing_input_role="OPENAI_API_KEY",
        attempt_id="final-wave-skip-fixture", reviewed_plan_sha256="4" * 64,
        created_at=created_at, seed=None,
    )
    paid_root, skip_root = scratch / "paid-repo", scratch / "skip-repo"
    paid_locator, paid_t7 = _fixture_repository(paid_root, paid_source, paid=True,
                                                  state_home=scratch / "paid-state-home")
    skip_locator, skip_t7 = _fixture_repository(skip_root, skip_source, paid=False)
    paid_source_path = repository / "docs/evidence/phase12-filter-v5-rootless-local/rehearsal-publication.json"
    skip_source_path = qa / "pre-egress/zero-call-skip.json"
    write_json(paid_source_path, paid_source)
    write_json(skip_source_path, skip_source)
    for name, value in (("paid/f3-state-locator.json", paid_locator), ("paid/t7-real-attempt.json", paid_t7),
                        ("skip/f3-state-locator.json", skip_locator), ("skip/t7-real-attempt.json", skip_t7)):
        write_json(fixtures / name, value)
    write_json(qa / "f3-state-locator.json", skip_locator)
    write_json(qa / "t7-real-attempt.json", skip_t7)
    shutil.rmtree(scratch)
    return FixtureSet(
        FixtureLineage("paid_attempt", paid_source_path, fixtures / "paid/f3-state-locator.json", fixtures / "paid/t7-real-attempt.json"),
        FixtureLineage("zero_call_skip", skip_source_path, fixtures / "skip/f3-state-locator.json", fixtures / "skip/t7-real-attempt.json"),
    )


def validate_fixture_lineage(lineage: FixtureLineage) -> LineageValidation:
    source, locator, t7 = (json.loads(path.read_bytes()) for path in (lineage.source, lineage.locator, lineage.t7))
    source_hash = hashlib.sha256(lineage.source.read_bytes()).hexdigest()
    locator_hash = hashlib.sha256(lineage.locator.read_bytes()).hexdigest()
    hash_field = "publication_receipt_sha256" if lineage.outcome == "paid_attempt" else "zero_call_skip_sha256"
    if locator["outcome"] != lineage.outcome or t7["outcome"] != lineage.outcome:
        raise ValueError("fixture outcome mismatch")
    if locator[hash_field] != source_hash or t7[hash_field] != source_hash or t7["f3_state_locator_sha256"] != locator_hash:
        raise ValueError("fixture hash mismatch")
    return LineageValidation(lineage.outcome, int(t7["provider_calls_issued"]))


def final_index_fixture(repository: Path, lineage: FixtureLineage, *,
                        pre_egress_paths: tuple[tuple[str, str], ...],
                        final_paths: tuple[tuple[str, str], ...], execution_commit: str,
                        legacy_input_manifest_sha256: str, created_at: str) -> dict[str, JsonValue]:
    source = json.loads(lineage.source.read_bytes())
    source_hash = hashlib.sha256(lineage.source.read_bytes()).hexdigest()
    paid = lineage.outcome == "paid_attempt"
    return {
        "schema_version": "rootless_final_verification_index_v1", "profile": PROFILE,
        "plan_binding_sha256": source.get("plan_binding_sha256") if paid else source.get("reviewed_plan_sha256"),
        "execution_commit": execution_commit, "outcome": lineage.outcome,
        "publication_receipt_sha256": source_hash if paid else None,
        "zero_call_skip_sha256": None if paid else source_hash,
        "state_inventory_sha256": source.get("state_inventory_sha256") if paid else None,
        "ordered_pre_egress_evidence": [evidence(repository, role, repository / path)
                                         for role, path in sorted(pre_egress_paths, key=lambda item: item[0].encode())],
        "ordered_final_evidence": [evidence(repository, role, repository / path)
                                    for role, path in sorted(final_paths, key=lambda item: item[0].encode())],
        "legacy_input_manifest_sha256": legacy_input_manifest_sha256,
        "provider_calls_before": 0, "provider_calls_after": 0, "created_at": created_at,
    }


def _prepare_pre_egress(repository: Path, execution_commit: str, created_at: str) -> None:
    write_anchor(repository, execution_commit)
    reports = (
        ("pre_f1", "f1-plan-compliance.json"),
        ("pre_f2", "f2-broker-security.json"),
        ("pre_f3", "f3-cli-rehearsal.json"),
    )
    for role, filename in reports:
        write_json(
            repository / QA_REL / "pre-egress" / filename,
            {"schema_version": "rootless_pre_egress_qa_v1", "profile": PROFILE,
             "role": role, "execution_commit": execution_commit, "exit_code": 0,
             "transport_mode": "fake", "provider_calls_before": 0,
             "provider_calls_after": 0, "created_at": created_at},
        )


def _run_sentinel(repository: Path, spec: SentinelSpec) -> SentinelResult:
    basetemp_names = {
        "f1-pytest": "f1", "f2-pytest": "f2", "f3-pytest": "f3",
        "f4-rootless-pytest": "f4-rootless", "f4-ruff": "f4-ruff",
        "f4-validate-config": "f4-validate-config", "f4-replay-pytest": "f4-replay",
    }
    role_root = repository / QA_REL / "basetemp" / basetemp_names[spec.role]
    shutil.rmtree(role_root, ignore_errors=True)
    sentinel = repository / FINAL_REL / "sentinels" / f"{spec.role}.json"
    sentinel.unlink(missing_ok=True)
    target = f4_rootless_targets(repository) if spec.role == "f4-rootless-pytest" else spec.target
    if spec.module == "pytest":
        target = ("-p", "no:cacheprovider", "--basetemp", os.fspath(role_root / "pytest"), *target)
    command = (
        os.path.realpath(sys.executable), "-B", os.fspath(repository / "scripts/run_phase12_rootless_offline_qa.py"),
        "--repo-root", os.fspath(repository), "--sentinel-role", spec.role,
        "--module", spec.module, "--", *target,
    )
    completed = subprocess.run(
        command, cwd=repository, env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"},
        capture_output=True, check=False, close_fds=True,
    )
    result = SentinelResult(
        spec.role, completed.returncode, hashlib.sha256(completed.stdout).hexdigest(),
        hashlib.sha256(completed.stderr).hexdigest(),
    )
    if result.exit_code != 0:
        sys.stderr.buffer.write(completed.stdout + completed.stderr)
        raise RuntimeError(f"offline sentinel failed: {spec.role}")
    return result


def _final_paths(repository: Path, *, paid: bool) -> tuple[tuple[str, str], ...]:
    paths = [
        ("f3-state-locator", f"{QA_REL.as_posix()}/f3-state-locator.json"),
        ("t7-real-attempt", f"{QA_REL.as_posix()}/t7-real-attempt.json"),
        ("final-f1-plan-compliance", f"{FINAL_REL.as_posix()}/f1-plan-compliance.json"),
        ("final-f2-broker-security", f"{FINAL_REL.as_posix()}/f2-broker-security.json"),
        ("final-f3-cli-rehearsal", f"{FINAL_REL.as_posix()}/f3-cli-rehearsal.json"),
        ("final-f4-claim-scope-regression", f"{FINAL_REL.as_posix()}/f4-claim-scope-regression.json"),
    ]
    paths.extend(
        (f"network-sentinel-{spec.role}", f"{FINAL_REL.as_posix()}/sentinels/{spec.role}.json")
        for spec in SENTINEL_SPECS
    )
    paths.append(
        ("rehearsal-publication", "docs/evidence/phase12-filter-v5-rootless-local/rehearsal-publication.json")
        if paid else ("zero-call-skip", f"{QA_REL.as_posix()}/pre-egress/zero-call-skip.json")
    )
    return tuple(paths)


def _clean_outputs(repository: Path) -> None:
    tracked_final = tuple(
        repository / FINAL_REL / name
        for name in (
            "f1-plan-compliance.json",
            "f2-broker-security.json",
            "f3-cli-rehearsal.json",
            "f4-claim-scope-regression.json",
            "final-verification-index.json",
        )
    )
    subprocess.run(
        ("/usr/bin/git", "checkout", "--", *(path.relative_to(repository).as_posix() for path in tracked_final)),
        cwd=repository,
        env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"},
        capture_output=True,
        check=False,
        close_fds=True,
    )
    for relative in (
        f"{QA_REL.as_posix()}/basetemp",
        f"{QA_REL.as_posix()}/final/fixtures",
        f"{QA_REL.as_posix()}/final/index-fixtures",
        f"{QA_REL.as_posix()}/final/sentinels",
    ):
        shutil.rmtree(repository / relative, ignore_errors=True)
    for path in (
        repository / QA_REL / "pre-egress/execution-anchor.json",
        repository / QA_REL / "pre-egress/f1-plan-compliance.json",
        repository / QA_REL / "pre-egress/f2-broker-security.json",
        repository / QA_REL / "pre-egress/f3-cli-rehearsal.json",
        repository / QA_REL / "pre-egress/zero-call-skip.json",
        repository / QA_REL / "f3-state-locator.json",
        repository / QA_REL / "t7-real-attempt.json",
        repository / "docs/evidence/phase12-filter-v5-rootless-local/rehearsal-publication.json",
        repository / "docs/evidence/phase12-filter-v5-rootless-local/final-verification-index.json",
    ):
        path.unlink(missing_ok=True)


def run(repository: Path) -> int:
    repository = repository.resolve(strict=True)
    os.umask(0o022)
    created_at = datetime.now(UTC).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")
    execution_commit = subprocess.run(
        ("/usr/bin/git", "rev-parse", "HEAD"), cwd=repository,
        env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"}, capture_output=True,
        text=True, check=True, close_fds=True,
    ).stdout.strip()
    _clean_outputs(repository)
    (repository / QA_REL / "basetemp").mkdir(mode=0o700, parents=True, exist_ok=True)
    final_dir = repository / FINAL_REL
    if final_dir.is_dir():
        final_dir.chmod(0o755)
    _prepare_pre_egress(repository, execution_commit, created_at)
    fixtures = build_synthetic_task7_fixtures(
        repository, execution_commit=execution_commit, created_at=created_at
    )
    validate_fixture_lineage(fixtures.paid)
    validate_fixture_lineage(fixtures.skipped)
    results = tuple(_run_sentinel(repository, spec) for spec in SENTINEL_SPECS)
    by_role = {result.role: result for result in results}
    def sentinel_path(role: str) -> Path:
        return repository / FINAL_REL / "sentinels" / f"{role}.json"
    f1_inputs = (evidence(repository, "network-sentinel-f1-pytest", sentinel_path("f1-pytest")),)
    f2_inputs = (evidence(repository, "network-sentinel-f2-pytest", sentinel_path("f2-pytest")),)
    f3_inputs = tuple(
        evidence(repository, role, path) for role, path in (
            ("network-sentinel-f3-pytest", sentinel_path("f3-pytest")),
            ("f3-state-locator", repository / QA_REL / "f3-state-locator.json"),
            ("t7-real-attempt", repository / QA_REL / "t7-real-attempt.json"),
            ("rehearsal-publication-fixture", fixtures.paid.source),
            ("zero-call-skip-fixture", fixtures.skipped.source),
        )
    )
    f4_inputs = tuple(
        evidence(repository, f"network-sentinel-{role}", sentinel_path(role))
        for role in ("f4-rootless-pytest", "f4-ruff", "f4-validate-config", "f4-replay-pytest")
    )
    envelopes = (
        ("f1-plan-compliance.json", evidence_envelope("F1", f1_inputs, (by_role["f1-pytest"],))),
        ("f2-broker-security.json", evidence_envelope("F2", f2_inputs, (by_role["f2-pytest"],))),
        ("f3-cli-rehearsal.json", evidence_envelope("F3", f3_inputs, (by_role["f3-pytest"],))),
        ("f4-claim-scope-regression.json", evidence_envelope("F4", f4_inputs, tuple(by_role[role] for role in
            ("f4-rootless-pytest", "f4-ruff", "f4-validate-config", "f4-replay-pytest")))),
    )
    for filename, envelope in envelopes:
        write_json(repository / FINAL_REL / filename, envelope)
    manifest = repository / "configs/phase12/filter_v5_rootless_local/external_inputs.json"
    legacy_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    final_paths = _final_paths(repository, paid=False)
    index = final_index_fixture(
        repository, fixtures.skipped, pre_egress_paths=PRE_EGRESS_PATHS,
        final_paths=final_paths, execution_commit=execution_commit,
        legacy_input_manifest_sha256=legacy_hash, created_at=created_at,
    )
    write_json(repository / FINAL_REL / "final-verification-index.json", index)
    write_json(repository / "docs/evidence/phase12-filter-v5-rootless-local/final-verification-index.json", index)
    scenario_root = repository / FINAL_REL / "index-fixtures"
    scenarios = (("paid", fixtures.paid, 4), ("p0", fixtures.skipped, 1),
                 ("p1", fixtures.skipped, 2), ("p2", fixtures.skipped, 3),
                 ("p3", fixtures.skipped, 4), ("orchestration-interrupted", fixtures.skipped, 3))
    for name, lineage, prefix in scenarios:
        write_json(
            scenario_root / f"{name}.json",
            final_index_fixture(
                repository, lineage, pre_egress_paths=PRE_EGRESS_PATHS[:prefix],
                final_paths=_final_paths(repository, paid=lineage.outcome == "paid_attempt"),
                execution_commit=execution_commit, legacy_input_manifest_sha256=legacy_hash,
                created_at=created_at,
            ),
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    return run(parser.parse_args().repo_root)


if __name__ == "__main__":
    raise SystemExit(main())
