from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from memcontam.clients.base import LLMResponse
from memcontam.readiness.phase13_main_runner import (
    MainRunRequest,
    open_main_run,
    prepare_main_run,
    resume_main,
    run_main,
)
from memcontam.readiness.phase13_v3_authority_models import V3Identity
from memcontam.readiness.phase13_v3_entrypoint import EntrypointError, SelectionRequest
from memcontam.readiness.phase13_v3_entrypoint_paths import EntrypointPathError

from .test_phase13_v3_entrypoint_fixture import (
    build_entrypoint_bytes,
)
from .test_phase13_v3_entrypoint_fixture import (
    entrypoint_bytes as entrypoint_bytes,
)
from .test_phase13_v3_entrypoint_fixture import (
    entrypoint_fixture as entrypoint_fixture,
)
from .test_phase13_v3_entrypoint_integration import deny_external as deny_external

ROOT = Path(__file__).resolve().parents[1]


def _request(selection: SelectionRequest) -> MainRunRequest:
    return MainRunRequest(
        repository_root=selection.repository_root,
        package_path=selection.package_path,
        authorization_path=selection.authorization_path,
        expected_authorization_sha256="",
        run_root=selection.repository_root,
        run_id=V3Identity().run_id,
        seed=0,
        authority_root=selection.authority_root,
        expected_authorization_sha256_file=selection.expected_authorization_sha256_file,
    )


def _read_only_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    assert arguments[0] in {"--help", "validate"}
    return subprocess.run(
        ("bash", "-c", "source .omo/evidence/phase13_shell_contract.sh; "
         'phase13_python -m memcontam.readiness.phase13_main_runner_cli "$@"',
         "phase13-runner-test", *arguments),
        cwd=ROOT, check=False, capture_output=True, text=True,
    )


def test_authorized_run_creation_binds_exact_frozen_inputs(entrypoint_fixture, deny_external):
    request = _request(entrypoint_fixture)
    run = prepare_main_run(request)
    try:
        assert run.status().pending_count == 1
        assert run.status().provider_calls_issued == 0
        assert run.ledger.binding.package_sha256 == hashlib.sha256(request.package_path.read_bytes()).hexdigest()
        assert run.ledger.binding.authorization_sha256 == hashlib.sha256(request.authorization_path.read_bytes()).hexdigest()
        assert run.ledger.path.name == "main_run_ledger_v3.sqlite3"
    finally:
        run.close()


def test_authorized_run_reopen_revalidates_package_and_authorization(entrypoint_fixture, deny_external):
    request = _request(entrypoint_fixture)
    prepare_main_run(request).close()
    reopened = open_main_run(request)
    try:
        assert reopened.status().pending_count == 1
    finally:
        reopened.close()
    request.authorization_path.write_bytes(b"{}\n")
    with pytest.raises(EntrypointError, match="MAIN_AUTHORIZATION_BINDING_MISMATCH"):
        open_main_run(request)


def test_authorized_run_and_resume_enforce_distinct_seed_boundaries(tmp_path, monkeypatch, deny_external):
    import memcontam.readiness.phase13_main_request_dispatch as dispatch

    for path, raw in build_entrypoint_bytes((0, 1)).items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    from .test_phase13_v3_entrypoint_fixture import AUTHORITY, seal_fixture_closure

    seal_fixture_closure(tmp_path)

    selection = SelectionRequest(tmp_path, tmp_path / "package.json", tmp_path / "authorization.json",
                                 AUTHORITY, tmp_path / "authorization.sha256", V3Identity().run_id)
    request = _request(selection)
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(dispatch, "count_prompt_tokens", lambda *_: 1)

    class FakeProvider:
        def send_compiled_v3(self, compiled, before_request):
            before_request()
            calls.append((compiled.key.parent_id, compiled.key.dispatch_id))
            return LLMResponse("final: 0", {"usage": {"input_tokens": 0, "output_tokens": 0}, "attempts": 1},
                               {"prompt_tokens": 0, "completion_tokens": 0}, 0)

    run = prepare_main_run(request)
    try:
        first = run.execute(tmp_path / "cache", max_units=1, tranche_ceiling_krw=500,
                            provider_factory=lambda _: FakeProvider())
        assert (first.completed_count, first.pending_count) == (1, 1)
    finally:
        run.close()
    resumed = open_main_run(replace(request, seed=1))
    try:
        final = resumed.execute(tmp_path / "cache", max_units=1, tranche_ceiling_krw=500,
                                provider_factory=lambda _: FakeProvider())
        assert (final.completed_count, final.pending_count) == (2, 0)
    finally:
        resumed.close()
    assert len({parent for parent, _request in calls}) == 2
    assert len(calls) == len({request_id for _parent, request_id in calls}) == 100


def test_run_and_resume_reject_seed_one_while_seed_zero_is_pending(
    tmp_path: Path, deny_external,
) -> None:
    for path, raw in build_entrypoint_bytes((0, 1)).items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    from .test_phase13_v3_entrypoint_fixture import AUTHORITY, seal_fixture_closure

    seal_fixture_closure(tmp_path)
    selection = SelectionRequest(tmp_path, tmp_path / "package.json", tmp_path / "authorization.json",
                                 AUTHORITY, tmp_path / "authorization.sha256", V3Identity().run_id)
    request = replace(_request(selection), seed=1)
    for invoke in (run_main, resume_main):
        with pytest.raises(ValueError, match="MAIN_TRANCHE_ORDER_MISMATCH"):
            invoke(request, cache_root=tmp_path / "cache", tranche_ceiling_krw=500, max_units=0)


def test_authorized_run_rejects_authorization_hash_tampering(entrypoint_fixture, deny_external):
    request = _request(entrypoint_fixture)
    assert request.expected_authorization_sha256_file is not None
    request.expected_authorization_sha256_file.write_bytes(b"0" * 64 + b"\n")
    with pytest.raises(EntrypointError, match="MAIN_AUTHORIZATION_BINDING_MISMATCH"):
        prepare_main_run(request)


def test_authorized_run_rejects_package_changed_after_validation(entrypoint_fixture, deny_external):
    request = _request(entrypoint_fixture)
    run = prepare_main_run(request)
    try:
        request.package_path.write_bytes(b"{}\n")
        with pytest.raises(EntrypointPathError, match="MAIN_PATH_UNSAFE"):
            run.selected.preflight(request.repository_root)
    finally:
        run.close()


@pytest.mark.parametrize("version", [1, 2])
def test_frozen_runner_binds_direct_authorization_trust_base(version):
    package = json.loads((ROOT / f"data/phase13/main/mr_p5/execution_package_v{version}.json").read_bytes())
    assert {"main_execution", "main_execution_models", "main_execution_bindings"} <= {
        binding["role"] for binding in package["artifacts"]
    }


def test_main_run_id_must_be_one_component(entrypoint_fixture, deny_external):
    request = replace(_request(entrypoint_fixture), run_id="../escape")
    with pytest.raises(EntrypointError, match="MAIN_CORRECTED_RUN_ID_MISMATCH"):
        prepare_main_run(request)


def test_phase13_help_exposes_main_execution_control_surface():
    result = _read_only_cli("--help")
    assert result.returncode == 0
    assert all(command in result.stdout for command in ("validate", "run", "status", "resume"))


def test_run_and_resume_require_explicit_authorized_seed() -> None:
    from memcontam.readiness.phase13_main_command import build_parser

    parser = build_parser("phase13-main-a", live=False)
    common = ["--repository-root", ".", "--package", "package.json", "--authorization", "authorization.json",
              "--expected-authorization-sha256", "a" * 64, "--run-root", ".", "--run-id", "run"]
    for command in ("run", "resume"):
        with pytest.raises(SystemExit):
            parser.parse_args([command, *common])
        assert parser.parse_args([command, *common, "--seed", "4"]).seed == 4


def test_main_runner_run_status_resume_are_offline_and_stable(entrypoint_fixture, deny_external):
    request = _request(entrypoint_fixture)
    started = run_main(request, cache_root=request.run_root / "cache", tranche_ceiling_krw=500, max_units=0)
    opened = open_main_run(request)
    try:
        status = opened.status()
    finally:
        opened.close()
    resumed = resume_main(request, cache_root=request.run_root / "cache", tranche_ceiling_krw=500, max_units=0)
    assert started == status == resumed
    assert status.pending_count == 1
    assert status.provider_calls_issued == 0


def test_main_cli_reports_bad_authorization_without_traceback(entrypoint_fixture):
    request = entrypoint_fixture
    assert request.expected_authorization_sha256_file is not None
    request.expected_authorization_sha256_file.write_bytes(b"0" * 64 + b"\n")
    result = _read_only_cli("validate", "--repository-root", str(request.repository_root),
        "--package", str(request.package_path), "--authorization", str(request.authorization_path),
        "--authority-root", str(request.authority_root), "--expected-authorization-sha256-file",
        str(request.expected_authorization_sha256_file))
    assert result.returncode != 0
    assert "MAIN_AUTHORIZATION_BINDING_MISMATCH" in result.stderr
    assert "Traceback" not in result.stderr
