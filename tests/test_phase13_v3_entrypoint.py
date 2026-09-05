from __future__ import annotations

import importlib
import importlib.util
import os
from contextlib import ExitStack

import pytest
from .test_phase13_v3_entrypoint_fixture import entrypoint_bytes as entrypoint_bytes, entrypoint_fixture as entrypoint_fixture
from .test_phase13_v3_entrypoint_integration import (
    deny_external as deny_external,
    test_exact_v3_cli_validate_is_provider_free as test_exact_v3_cli_validate_is_provider_free,
    test_validate_denies_all_outbound_sockets_and_never_loads_credentials as test_validate_denies_all_outbound_sockets_and_never_loads_credentials,
    test_both_active_runner_apis_reject_old_identity as test_both_active_runner_apis_reject_old_identity,
    test_valid_v3_guarded_run_and_resume_without_calls as test_valid_v3_guarded_run_and_resume_without_calls,
    test_v3_execution_uses_guarded_requests_and_real_ordinary_runtime as test_v3_execution_uses_guarded_requests_and_real_ordinary_runtime,
    test_proof_bytes_must_be_canonical_not_just_semantically_equal as test_proof_bytes_must_be_canonical_not_just_semantically_equal,
)


def test_v3_entrypoint_safety_contracts_exist():
    assert importlib.util.find_spec("memcontam.readiness.phase13_v3_runtime_identity")
    assert importlib.util.find_spec("memcontam.readiness.phase13_v3_entrypoint_paths")


@pytest.fixture
def lease():
    with ExitStack() as stack:
        yield stack


@pytest.fixture
def identity_api():
    name = "memcontam.readiness.phase13_v3_runtime_identity"
    assert importlib.util.find_spec(name), "V3 runtime identity gate is missing"
    return importlib.import_module(name)


@pytest.fixture
def path_api():
    name = "memcontam.readiness.phase13_v3_entrypoint_paths"
    assert importlib.util.find_spec(name), "V3 entrypoint descriptor gate is missing"
    return importlib.import_module(name)


@pytest.mark.parametrize("raw", [b"", b"a" * 64, b"A" * 64 + b"\n",
    b"a" * 63 + b"\n", b"a" * 65 + b"\n", b"g" * 64 + b"\n",
    b"a" * 64 + b"\r\n", b"a" * 64 + b"\n\n", b" " + b"a" * 64 + b"\n",
    b"a" * 64 + b"\n " , b"\x00" * 64 + b"\n"])
def test_sidecar_rejects_noncanonical_bytes(path_api, tmp_path, raw):
    sidecar = tmp_path / "authorization.sha256"
    sidecar.write_bytes(raw)
    with pytest.raises(ValueError, match="MAIN_AUTHORIZATION_BINDING_MISMATCH"):
        path_api.read_authorization_digest(tmp_path, sidecar)


def test_sidecar_accepts_exact_lowercase_hex_lf(path_api, tmp_path):
    sidecar = tmp_path / "authorization.sha256"
    sidecar.write_bytes(b"0123456789abcdef" * 4 + b"\n")
    assert path_api.read_authorization_digest(tmp_path, sidecar) == "0123456789abcdef" * 4


@pytest.mark.parametrize("kind", ["file", "parent", "escape", "fifo", "directory"])
def test_sidecar_rejects_unsafe_path(path_api, tmp_path, kind):
    root = tmp_path / "root"
    root.mkdir()
    target = tmp_path / "digest"
    target.write_bytes(b"a" * 64 + b"\n")
    sidecar = root / "digest"
    match kind:
        case "file":
            sidecar.symlink_to(target)
        case "parent":
            (root / "linked").symlink_to(tmp_path, target_is_directory=True)
            sidecar = root / "linked/digest"
        case "escape":
            sidecar = root / "../digest"
        case "fifo":
            os.mkfifo(sidecar)
        case "directory":
            sidecar.mkdir()
    with pytest.raises(ValueError, match="MAIN_PATH_UNSAFE"):
        path_api.read_authorization_digest(root, sidecar)


def test_runtime_freezes_actual_canonical_inventory(identity_api):
    identity = identity_api.freeze_runtime_identity()
    assert identity.executable == "/home/hyunwoo/miniconda3/envs/memcontam/bin/python"
    assert identity.version == "3.11.15"
    assert len(identity.versions) == 102
    identity_api.validate_runtime_identity(identity)


@pytest.mark.parametrize("field,value", [
    ("executable", "/usr/bin/python3"), ("prefix", "/tmp/other"),
    ("base_prefix", "/tmp/other"), ("implementation", "PyPy"),
    ("version", "3.11.14"), ("pythonpath", "/tmp/src"),
    ("module_origin", "/tmp/src/memcontam/__init__.py"),
    ("package_paths", ("/tmp/src/memcontam",)),
    ("requirements_sha256", "0" * 64), ("requirements_dev_sha256", "0" * 64),
])
def test_runtime_identity_rejects_frozen_drift(identity_api, field, value):
    current = identity_api.freeze_runtime_identity()
    with pytest.raises(ValueError, match="MAIN_RUNTIME_IDENTITY_DRIFT"):
        identity_api.validate_runtime_identity(current.model_copy(update={field: value}))


def test_runtime_distribution_drift_fails_before_provider(identity_api, monkeypatch, deny_external):
    current = identity_api.freeze_runtime_identity()
    version = identity_api.metadata.version
    for changed, _expected in current.versions:
        with monkeypatch.context() as scoped:
            scoped.setattr(identity_api.metadata, "version",
                          lambda name: "0.0.0" if name == changed else version(name))
            with pytest.raises(ValueError, match="MAIN_RUNTIME_IDENTITY_DRIFT"):
                identity_api.validate_runtime_identity(current)


def test_sibling_pythonpath_origin_fails_before_provider(identity_api, monkeypatch, deny_external):
    current = identity_api.freeze_runtime_identity()
    monkeypatch.setenv("PYTHONPATH", "/tmp/sibling/src:" + os.environ["PYTHONPATH"])
    with pytest.raises(ValueError, match="MAIN_RUNTIME_IDENTITY_DRIFT"):
        identity_api.validate_runtime_identity(current)


def test_entrypoint_rejects_symlinked_ledger_parent(path_api, tmp_path, deny_external):
    root = tmp_path / "root"
    root.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="MAIN_PATH_UNSAFE"):
        with path_api.private_ledger(linked / "run", create=True):
            pytest.fail("unsafe directory accepted")


def test_private_ledger_create_reopen(path_api, tmp_path):
    directory = tmp_path / "run"
    with path_api.private_ledger(directory, create=True) as ledger:
        with ledger.connect() as connection:
            connection.execute("CREATE TABLE fixture(value TEXT)")
            connection.execute("INSERT INTO fixture VALUES ('retained')")
        assert (directory.stat().st_mode & 0o777) == 0o700
        assert (directory / "main_run_ledger_v3.sqlite3").stat().st_mode & 0o777 == 0o600
    with path_api.private_ledger(directory, create=False) as ledger:
        with ledger.connect() as connection:
            assert connection.execute("SELECT value FROM fixture").fetchall() == [("retained",)]


@pytest.mark.parametrize("target", ["directory", "database", "wal", "shm"])
def test_private_ledger_rejects_mode_drift(path_api, tmp_path, target):
    directory = tmp_path / "run"
    with path_api.private_ledger(directory, create=True) as ledger:
        paths = {"directory": directory, "database": directory / "main_run_ledger_v3.sqlite3",
                 "wal": directory / "main_run_ledger_v3.sqlite3-wal",
                 "shm": directory / "main_run_ledger_v3.sqlite3-shm"}
        path = paths[target]
        if target in {"wal", "shm"}:
            path.touch(mode=0o600)
        path.chmod(0o755 if target == "directory" else 0o644)
        with pytest.raises(ValueError, match="MAIN_PATH_UNSAFE"):
            with ledger.connect():
                pytest.fail("mode drift accepted")


@pytest.mark.parametrize("suffix", ["-wal", "-shm"])
def test_private_ledger_rejects_same_mode_journal_replacement(path_api, tmp_path, suffix):
    directory = tmp_path / "run"
    with path_api.private_ledger(directory, create=True) as ledger:
        with ledger.connect() as connection:
            connection.execute("CREATE TABLE initial(value TEXT)")
        with pytest.raises(ValueError, match="MAIN_PATH_UNSAFE"):
            with ledger.connect() as connection:
                connection.execute("INSERT INTO initial VALUES ('fixture')")
                journal = directory / ("main_run_ledger_v3.sqlite3" + suffix)
                replacement = directory / "replacement"
                replacement.write_bytes(journal.read_bytes())
                replacement.chmod(0o600)
                replacement.replace(journal)


@pytest.mark.parametrize("mutation", ["rewrite", "symlink", "replace"])
def test_resource_swap_between_validation_and_preflight_fails(path_api, tmp_path, mutation, lease, deny_external):
    from memcontam.readiness.phase13_v3_resource_files import read_files

    path = tmp_path / "resource"
    path.write_bytes(b"original")
    resources = read_files(tmp_path, ("resource",), lease=lease)
    match mutation:
        case "rewrite":
            path.write_bytes(b"modified")
        case "symlink":
            path.unlink()
            path.symlink_to(tmp_path / "other")
        case "replace":
            replacement = tmp_path / "replacement"
            replacement.write_bytes(b"original")
            replacement.replace(path)
    with pytest.raises(ValueError, match="MAIN_PATH_UNSAFE"):
        path_api.verify_resource_namespace(tmp_path, resources)
    assert resources[0].raw == b"original"


def test_resource_handoff_does_not_reopen(path_api, tmp_path, monkeypatch, lease):
    from memcontam.readiness.phase13_v3_resource_files import read_files

    path = tmp_path / "resource"
    path.write_bytes(b"original")
    resources = read_files(tmp_path, ("resource",), lease=lease)
    original_open = os.open

    def directories_only(path, flags, *args, **kwargs):
        assert flags & os.O_DIRECTORY, "resource was reopened"
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", directories_only)
    path_api.verify_resource_namespace(tmp_path, resources)
    assert resources[0].raw == b"original"


def test_resource_swap_during_descriptor_recheck_fails(path_api, tmp_path, monkeypatch, lease):
    from memcontam.readiness.phase13_v3_resource_files import read_files

    path = tmp_path / "resource"
    path.write_bytes(b"original")
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"original")
    resources = read_files(tmp_path, ("resource",), lease=lease)
    pread = os.pread

    def swap(descriptor, count, offset):
        result = pread(descriptor, count, offset)
        replacement.replace(path)
        return result

    monkeypatch.setattr(os, "pread", swap)
    with pytest.raises(ValueError, match="MAIN_PATH_UNSAFE"):
        path_api.verify_resource_namespace(tmp_path, resources)


def test_ordinary_handoff_contract_exists():
    from memcontam.experiment import phase13_ordinary_runtime as runtime

    assert hasattr(runtime, "ValidatedOrdinaryResources")
    assert "validated_resources" in runtime.ProspectiveOrdinaryRun.__dataclass_fields__


@pytest.mark.parametrize("command", ["run", "resume"])
def test_all_active_commands_reject_old_run_id_before_provider_construction(tmp_path, command, deny_external):
    name = "memcontam.readiness.phase13_v3_entrypoint"
    assert importlib.util.find_spec(name), "shared V3 selector is missing"
    api = importlib.import_module(name)
    request = api.SelectionRequest(
        repository_root=tmp_path, package_path=tmp_path / "absent-package",
        authorization_path=tmp_path / "absent-auth", authority_root=None,
        expected_authorization_sha256_file=None, run_id="phase13-main-a-corrected-v2",
    )
    with pytest.raises(ValueError, match="MAIN_CORRECTED_RUN_ID_MISMATCH"):
        api.select_execution(request, command)
