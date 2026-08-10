from __future__ import annotations

# allow: SIZE_OK — the exact T3 QA argv requires all binding cases in this test module.

import argparse
import importlib.util
import json
import os
import stat
import subprocess
import sys
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import pytest

from memcontam.experiment.phase12 import cli as phase12_cli
from memcontam.experiment.phase12.filter_challenge.rootless_local_contract import JsonValue


def test_rootless_admin_parser_exposes_init_state_command() -> None:
    # Given: the Phase-12 command tree.
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    phase12_cli.add_parser(commands)

    # When: an operator requests the rootless state initializer.
    arguments = parser.parse_args(
        [
            "phase12",
            "filter-v5-rootless",
            "--repo-root",
            "/repository",
            "--state-home",
            "/state-home",
            "init-state",
            "--attempt-id",
            "attempt-001",
            "--plan-source",
            "/plans/reviewed.md",
            "--plan-descriptor",
            "/plans/reviewed.sha256",
            "--review-metadata",
            "/plans/review.json",
        ]
    )

    # Then: the dedicated rootless command family owns the operation.
    assert arguments.phase12_command == "filter-v5-rootless"
    assert arguments.rootless_command == "init-state"


def test_rootless_canonical_json_rejects_duplicate_float_and_noncanonical_bytes() -> None:
    from memcontam.experiment.phase12.filter_challenge.rootless_local_contract import (
        RootlessContractError,
        canonical_json_file,
        parse_canonical_object,
    )

    # Given: a closed JSON object and malformed representations of the same data.
    canonical = canonical_json_file({"a": 1, "b": "ok"})

    # When: the rootless canonical decoder receives each byte sequence.
    assert parse_canonical_object(canonical) == {"a": 1, "b": "ok"}
    with pytest.raises(RootlessContractError, match="ROOTLESS_JSON_INVALID"):
        parse_canonical_object(b'{"a":1,"a":2}\n')
    with pytest.raises(RootlessContractError, match="ROOTLESS_JSON_INVALID"):
        parse_canonical_object(b'{"a":1.0}\n')
    with pytest.raises(RootlessContractError, match="ROOTLESS_JSON_NONCANONICAL"):
        parse_canonical_object(b'{"b":"ok","a":1}\n')

    # Then: only compact sorted LF-terminated canonical bytes are accepted.


def test_rootless_canonical_json_uses_the_closed_escape_table() -> None:
    from memcontam.experiment.phase12.filter_challenge.rootless_local_contract import (
        RootlessContractError,
        canonical_json_file,
        parse_canonical_object,
    )

    # Given: every special escape class and invalid Unicode input.
    value: dict[str, JsonValue] = {"value": '"\\/\b\t\n\f\r\x00\x1f'}

    # When: the canonical encoder serializes it.
    raw = canonical_json_file(value)

    # Then: slash stays literal, short escapes are used, and other controls are lowercase.
    assert raw == b'{"value":"\\"\\\\/\\b\\t\\n\\f\\r\\u0000\\u001f"}\n'
    assert parse_canonical_object(raw) == value
    with pytest.raises(RootlessContractError, match="ROOTLESS_JSON_INVALID"):
        canonical_json_file({"value": "e\u0301"})
    with pytest.raises(RootlessContractError, match="ROOTLESS_JSON_INVALID"):
        canonical_json_file({"value": "\ud800"})


def test_rootless_signatures_bind_the_domain_and_payload() -> None:
    from memcontam.experiment.phase12.filter_challenge.rootless_local_contract import (
        RootlessContractError,
        public_key_from_seed,
        sign_object,
        verify_object_signature,
    )

    # Given: a deterministic local seed and an unsigned closed object.
    seed = bytes(range(32))
    payload = {"profile": "local_rootless_non_authoritative", "value": 1}
    public_key = public_key_from_seed(seed)

    # When: the object is signed in its declared authority domain.
    signature = sign_object(seed, "plan-acknowledgement-v1", payload)

    # Then: only the same domain and canonical payload verify.
    verify_object_signature(public_key, "plan-acknowledgement-v1", payload, signature)
    with pytest.raises(RootlessContractError, match="ROOTLESS_SIGNATURE_INVALID"):
        verify_object_signature(public_key, "rate-acknowledgement-v1", payload, signature)


def test_init_state_writes_private_key_and_immutable_plan_binding(tmp_path: Path) -> None:
    from memcontam.experiment.phase12.filter_challenge.rootless_local_state import (
        InitStateRequest,
        initialize_state,
    )

    # Given: a private state home and descriptor-bound reviewed plan bytes.
    state_home = tmp_path / "state"
    state_home.mkdir(mode=0o700)
    plan = tmp_path / "reviewed-plan.md"
    plan.write_bytes(b"reviewed rootless plan\n")
    descriptor = tmp_path / "reviewed-plan.sha256"
    descriptor.write_bytes(
        f"{sha256(plan.read_bytes()).hexdigest()}  phase12-filter-v5-rootless-local-execution.md\n".encode()
    )
    metadata = tmp_path / "review.json"
    metadata.write_text("{}\n", encoding="utf-8")
    for path in (plan, descriptor, metadata):
        os.chmod(path, 0o600)

    # When: rootless state is initialized without a provider input.
    result = initialize_state(
        InitStateRequest(state_home, plan, descriptor, metadata, "attempt-001")
    )

    # Then: the fixed plan and seed are current-UID private one-link regular files.
    assert result.plan_binding.read_bytes() == plan.read_bytes()
    assert len(result.private_seed.read_bytes()) == 32
    assert stat.S_IMODE(result.plan_binding.stat().st_mode) == 0o600
    assert stat.S_IMODE(result.private_seed.stat().st_mode) == 0o600
    assert stat.S_IMODE(result.state_root.stat().st_mode) == 0o700
    assert stat.S_IMODE(result.runtime_lock.stat().st_mode) == 0o600
    assert result.tokenizer_cache.is_dir()


def test_acknowledgements_and_execution_authority_use_closed_schemas(tmp_path: Path) -> None:
    from memcontam.experiment.phase12.filter_challenge.rootless_local_acknowledgement import (
        AcknowledgementClock,
        create_plan_acknowledgement,
        create_rate_acknowledgement,
        create_stage_acknowledgement,
        validate_acknowledgement_pair,
    )
    from memcontam.experiment.phase12.filter_challenge.rootless_local_binding import (
        build_execution_authority,
        build_stage_binding,
    )
    from memcontam.experiment.phase12.filter_challenge.rootless_local_contract import (
        public_key_from_seed,
    )

    # Given: one key and fresh immutable prerequisite hashes.
    seed = bytes(range(32))
    public_key = public_key_from_seed(seed)
    now = datetime(2026, 8, 9, 12, 0, tzinfo=UTC)
    clock = AcknowledgementClock(now)
    common = {
        "attempt_id": "attempt-001",
        "plan_binding_sha256": "1" * 64,
        "seed": seed,
        "clock": clock,
    }

    # When: two operators approve the plan/stage and one rate capability is recorded.
    plan = [
        create_plan_acknowledgement(
            **common,
            set_id="plan",
            operator_label=f"operator-{index}",
            operator_index=index,
            nonce=f"{index}" * 64,
        )
        for index in (1, 2)
    ]
    binding = build_stage_binding(
        attempt_id="attempt-001",
        stage="screening",
        plan_binding_sha256="1" * 64,
        trusted_base_commit="a" * 40,
        execution_commit="b" * 40,
        decoding_authority_sha256="2" * 64,
        rate_card_sha256="3" * 64,
        source_manifest_sha256="4" * 64,
        runtime_manifest_sha256="5" * 64,
        input_manifest_sha256="6" * 64,
        compiler_sha256="7" * 64,
        schedule_sha256="8" * 64,
        registered_slots=90,
        stage_cap_nanousd=2_000_000_000,
        created_at="2026-08-09T12:00:00Z",
    )
    binding_hash = sha256(json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    stage = [
        create_stage_acknowledgement(
            **common,
            stage="screening",
            set_id="screening",
            stage_binding_sha256=binding_hash,
            operator_label=f"operator-{index}",
            operator_index=index,
            nonce=f"{index + 2}" * 64,
        )
        for index in (1, 2)
    ]
    rate = create_rate_acknowledgement(
        attempt_id="attempt-001",
        provider_account_label="provider-account",
        rpm_limit=6,
        tpm_limit=30_000,
        observed_at="2026-08-09T11:59:30Z",
        nonce="5" * 64,
        seed=seed,
        clock=clock,
    )
    authority = build_execution_authority(
        binding,
        plan,
        stage,
        rate,
        seed=seed,
        issued_at="2026-08-09T12:00:00Z",
    )

    # Then: pairs are distinct and the authority resolves the acknowledgement cycle.
    validate_acknowledgement_pair(plan, public_key, now)
    validate_acknowledgement_pair(stage, public_key, now)
    assert binding["transport_mode"] == "live"
    assert binding["predecessor_terminal_sha256"] is None
    assert binding["freeze_b_sha256"] is None
    assert authority["stage_binding_sha256"] == binding_hash
    assert authority["expires_at"] == "2026-08-10T12:00:00Z"
    plan_hashes = authority["plan_acknowledgement_sha256s"]
    assert isinstance(plan_hashes, list) and len(plan_hashes) == 2


def test_live_stage_binding_rejects_zero_commit_placeholders() -> None:
    from memcontam.experiment.phase12.filter_challenge.rootless_local_binding import (
        build_stage_binding,
    )
    from memcontam.experiment.phase12.filter_challenge.rootless_local_contract import (
        RootlessContractError,
    )

    # Given: every required hash but placeholder Git identities.
    arguments = {
        "attempt_id": "attempt-001",
        "stage": "screening",
        "plan_binding_sha256": "1" * 64,
        "trusted_base_commit": "0" * 40,
        "execution_commit": "0" * 40,
        "decoding_authority_sha256": "2" * 64,
        "rate_card_sha256": "3" * 64,
        "source_manifest_sha256": "4" * 64,
        "runtime_manifest_sha256": "5" * 64,
        "input_manifest_sha256": "6" * 64,
        "compiler_sha256": "7" * 64,
        "schedule_sha256": "8" * 64,
        "registered_slots": 90,
        "stage_cap_nanousd": 2_000_000_000,
        "created_at": "2026-08-09T12:00:00Z",
    }

    # When/Then: no live binding can encode an unbound commit placeholder.
    with pytest.raises(RootlessContractError, match="ROOTLESS_BINDING_INVALID"):
        build_stage_binding(**arguments)


def test_rate_acknowledgement_rejects_stale_observation() -> None:
    from memcontam.experiment.phase12.filter_challenge.rootless_local_acknowledgement import (
        AcknowledgementClock,
        create_rate_acknowledgement,
    )
    from memcontam.experiment.phase12.filter_challenge.rootless_local_contract import (
        RootlessContractError,
    )

    # Given: an operator observation older than five minutes.
    clock = AcknowledgementClock(datetime(2026, 8, 9, 12, 0, tzinfo=UTC))

    # When/Then: rate authority creation fails before signing.
    with pytest.raises(RootlessContractError, match="ROOTLESS_RATE_ACKNOWLEDGEMENT_STALE"):
        create_rate_acknowledgement(
            attempt_id="attempt-001",
            provider_account_label="provider-account",
            rpm_limit=6,
            tpm_limit=30_000,
            observed_at="2026-08-09T11:54:59Z",
            nonce="5" * 64,
            seed=bytes(range(32)),
            clock=clock,
        )


def test_runtime_manifest_binds_installation_and_external_observations() -> None:
    from memcontam.experiment.phase12.filter_challenge.rootless_local_binding import (
        RuntimeInstallationEvidence,
        build_runtime_manifest,
    )
    from memcontam.experiment.phase12.filter_challenge.rootless_local_contract import (
        public_key_from_seed,
        verify_object_signature,
    )

    # Given: verified installation evidence and three ordered external observations.
    seed = bytes(range(32))
    evidence = RuntimeInstallationEvidence(
        python_path="/state/venvs/phase12-filter-v5-rootless-local/bin/python",
        python_version="3.11.13",
        pip_version="25.2",
        memcontam_import_path="/repo/src/memcontam/__init__.py",
        repo_root_mode_bits=493,
        editable_direct_url_sha256="1" * 64,
        distribution_record_sha256="2" * 64,
        native_extension_hashes=("3" * 64,),
        requirements_lock_sha256="4" * 64,
        requirements_dev_lock_sha256="5" * 64,
        tiktoken_version="0.11.0",
        tokenizer_source_sha256="446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d",
    )
    observations: list[JsonValue] = [
        {"role": role, "full_sha256": str(index) * 64}
        for index, role in enumerate(
            ("experiment-design", "filter-v5-amendment", "authority-agents"), start=6
        )
    ]

    # When: the signed runtime manifest is constructed.
    manifest = build_runtime_manifest(
        evidence,
        observations,
        seed=seed,
        created_at="2026-08-09T12:00:00Z",
    )

    # Then: bootstrap literals, installation hashes, and external order are immutable inputs.
    assert manifest["bootstrap_index_url"] == "https://pypi.org/simple/"
    assert manifest["bootstrap_egress_policy"] == "public_pypi_hash_pinned"
    assert manifest["ordered_external_authorities"] == observations
    signature = manifest["signature"]
    assert isinstance(signature, str)
    unsigned = dict(manifest)
    del unsigned["signature"]
    verify_object_signature(public_key_from_seed(seed), "runtime-manifest-v1", unsigned, signature)


def test_rootless_parser_accepts_closed_administrative_command_argv() -> None:
    # Given: every Task-3 and future-task administrative command name.
    command_arguments = {
        "verify-bootstrap-runtime": ["--python", "/state/venvs/rootless/bin/python"],
        "acknowledge-plan": [
            "--attempt-id", "attempt-001", "--set-id", "plan", "--operator-index", "1",
            "--operator-label", "operator-1",
        ],
        "acknowledge-stage": [
            "--attempt-id", "attempt-001", "--stage", "screening", "--set-id", "screening",
            "--operator-index", "1", "--operator-label", "operator-1",
        ],
        "acknowledge-rate-capability": [
            "--attempt-id", "attempt-001", "--provider-account-label", "provider",
            "--rpm-limit", "6", "--tpm-limit", "30000",
        ],
        "bind-screening": ["--attempt-id", "attempt-001"],
        "bind-bct": ["--attempt-id", "attempt-001"],
        "build-execution-authority": [
            "--attempt-id", "attempt-001", "--stage", "screening", "--plan-set-id", "plan",
            "--stage-set-id", "screening",
        ],
        "preflight": ["--attempt-id", "attempt-001"],
        "finalize-zero-call-preclaim": [
            "--attempt-id", "attempt-001", "--execution-commit", "a" * 40,
            "--failed-command", "preflight", "--observed-exit", "1",
        ],
        "record-t7-qa": [],
        "publish-receipt": ["--attempt-id", "attempt-001"],
        "reconcile-attempt": [
            "--attempt-id", "attempt-001", "--execution-commit", "a" * 40,
        ],
        "write-execution-anchor": ["--execution-commit", "a" * 40],
        "continue-after-screening": ["--attempt-id", "attempt-001"],
        "seal-post-screening-block": [
            "--attempt-id", "attempt-001", "--reason", "ROOTLESS_BCT_SETUP_FAILED",
        ],
        "finalize-after-stage-exit": [
            "--attempt-id", "attempt-001", "--stage", "screening", "--observed-exit", "69",
        ],
        "run-pre-egress-qa": [
            "--role", "pre_f1", "--execution-commit", "a" * 40,
        ],
        "verify-pre-egress-qa": ["--execution-commit", "a" * 40],
    }
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    phase12_cli.add_parser(commands)

    # When/Then: each exact command shape parses under the isolated rootless family.
    for command, options in command_arguments.items():
        state = [] if command == "verify-bootstrap-runtime" else ["--state-home", "/state"]
        arguments = parser.parse_args(
            ["phase12", "filter-v5-rootless", "--repo-root", "/repo", *state, command, *options]
        )
        assert arguments.rootless_command == command


def test_cleanup_helper_removes_only_fixed_bootstrap_children(tmp_path: Path) -> None:
    # Given: the stable venv parent, both invocation-owned children, and an unrelated child.
    venvs = tmp_path / "venvs"
    venvs.mkdir(mode=0o700)
    transient = venvs / ".phase12-filter-v5-rootless-local-bootstrap-tmp"
    final = venvs / "phase12-filter-v5-rootless-local"
    unrelated = venvs / "keep"
    for path in (transient, final, unrelated):
        path.mkdir(mode=0o700)
        (path / "value").write_text(path.name, encoding="utf-8")

    # When: dependency-free cleanup receives both closed removal flags.
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-I",
            "scripts/cleanup_phase12_filter_v5_rootless_bootstrap.py",
            "--venvs-root",
            os.fspath(venvs),
            "--remove-transient",
            "--remove-incomplete-venv",
        ],
        cwd=Path.cwd(),
        capture_output=True,
        check=False,
    )

    # Then: only the two fixed children disappear and the helper emits no bytes.
    assert result.returncode == 0
    assert result.stdout == b""
    assert result.stderr == b""
    assert not transient.exists()
    assert not final.exists()
    assert unrelated.is_dir()


def test_skip_receipt_nullability_distinguishes_external_and_nonexternal_failures() -> None:
    from memcontam.experiment.phase12.filter_challenge.rootless_local_acknowledgement import (
        create_skip_receipt,
    )

    # Given: one external-authority failure before a recoverable key exists.
    external = create_skip_receipt(
        reason="ROOTLESS_MISSING_EXTERNAL_INPUT",
        missing_input_role="ROOTLESS_THEORETICAL_EXPERIMENT_DESIGN",
        external_authority_diagnostic="ROOTLESS_EXTERNAL_AUTHORITY_HASH_MISMATCH",
        attempt_id="attempt-001",
        reviewed_plan_sha256="1" * 64,
        created_at="2026-08-09T12:00:00Z",
    )

    # When: a nonexternal missing input is represented by the same closed schema.
    ordinary = create_skip_receipt(
        reason="ROOTLESS_MISSING_EXTERNAL_INPUT",
        missing_input_role="ROOTLESS_PLAN_SOURCE",
        attempt_id=None,
        reviewed_plan_sha256=None,
        created_at="2026-08-09T12:00:00Z",
    )

    # Then: diagnostic and signature nullability cannot leak across branches.
    assert external["external_authority_diagnostic"] == "ROOTLESS_EXTERNAL_AUTHORITY_HASH_MISMATCH"
    assert external["key_fingerprint"] is None and external["signature"] is None
    assert ordinary["external_authority_diagnostic"] is None
    assert ordinary["failed_command"] is None and ordinary["observed_exit"] is None


def test_t3_qa_payload_repeats_runtime_manifest_evidence() -> None:
    path = Path("scripts/write_phase12_filter_v5_rootless_task_qa.py")
    spec = importlib.util.spec_from_file_location("rootless_t3_writer", path)
    assert spec is not None and spec.loader is not None
    writer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = writer
    try:
        spec.loader.exec_module(writer)
    finally:
        del sys.modules[spec.name]

    # Given: a successful exact command result and its verified runtime manifest.
    command = writer.CommandResult(("/python", "-B", "-m", "pytest"), 0, b"pass", b"", 0, 0)
    runtime: dict[str, JsonValue] = {
        "bootstrap_index_url": "https://pypi.org/simple/",
        "bootstrap_egress_policy": "public_pypi_hash_pinned",
        "distribution_record_sha256": "1" * 64,
        "native_extension_hashes": ["2" * 64],
        "editable_direct_url_sha256": "3" * 64,
        "top_p_parameter_sent": False,
        "tokenizer_name": "o200k_base",
        "tokenizer_source_sha256": "4" * 64,
        "normalized_repo_mode_bits": 493,
    }

    # When: the dedicated Task-3 envelope is assembled.
    payload = writer.build_rootless_t3_binding_payload(
        command,
        runtime,
        runtime_manifest_sha256="5" * 64,
        created_at="2026-08-09T12:00:00Z",
    )

    # Then: the final refined schema has no generic assertion array or nullable fields.
    assert payload["schema_version"] == "rootless_t3_binding_qa_v1"
    assert payload["distribution_record_sha256"] == runtime["distribution_record_sha256"]
    assert payload["runtime_manifest_sha256"] == "5" * 64
    assert payload["mountinfo_longest_prefix_passed"] is True
    assert payload["provider_calls_before"] == payload["provider_calls_after"] == 0
    assert "ordered_assertions" not in payload


def test_tokenizer_source_is_validated_copied_and_rechecked(tmp_path: Path) -> None:
    from memcontam.experiment.phase12.filter_challenge.rootless_local_state import (
        cache_tokenizer_source,
    )

    # Given: a private canonical BPE source and an empty private cache directory.
    source = tmp_path / "o200k.tiktoken"
    source.write_bytes(b"YQ== 0\nYg== 1\n")
    source.chmod(0o600)
    cache = tmp_path / "cache"
    cache.mkdir(mode=0o700)
    expected = sha256(source.read_bytes()).hexdigest()

    # When: the tokenizer authority is installed into the fixed cache entry.
    destination = cache_tokenizer_source(source, cache, expected_sha256=expected)

    # Then: exact bytes and private metadata survive the copy/recheck boundary.
    assert destination.name == "fb374d419588a4632f3f557e76b4b70aebbca790"
    assert destination.read_bytes() == source.read_bytes()
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600


def test_fake_and_live_stage_bindings_are_schema_disjoint() -> None:
    from memcontam.experiment.phase12.filter_challenge.rootless_local_binding import (
        build_fake_stage_binding,
        validate_live_stage_binding,
    )
    from memcontam.experiment.phase12.filter_challenge.rootless_local_contract import (
        RootlessContractError,
    )

    # Given: a closed unsigned fake-only binding.
    fake = build_fake_stage_binding(
        fixture_id="fixture-1",
        stage="screening",
        source_manifest_sha256="1" * 64,
        input_manifest_sha256="2" * 64,
        compiler_sha256="3" * 64,
        schedule_sha256="4" * 64,
        fake_scenario_sha256="5" * 64,
    )

    # When/Then: production validation rejects fake schema and transport before state access.
    assert fake["schema_version"] == "rootless_fake_stage_binding_v1"
    assert fake["transport_mode"] == "fake"
    with pytest.raises(RootlessContractError, match="ROOTLESS_BINDING_INVALID"):
        validate_live_stage_binding(fake)


def test_rootless_config_validator_rejects_any_canonical_config_drift(tmp_path: Path) -> None:
    from memcontam.experiment.phase12.filter_challenge.rootless_local_binding import (
        validate_rootless_configs,
    )

    # Given: the repository's four canonical rootless config artifacts.
    repository = Path.cwd()

    # When: validation runs over the checked-in config root.
    bindings = validate_rootless_configs(repository)

    # Then: each named authority is represented by an immutable SHA-256 digest.
    assert set(bindings) == {"bct", "decoding_authority", "rate_card", "screening"}
    assert all(len(digest) == 64 for digest in bindings.values())
