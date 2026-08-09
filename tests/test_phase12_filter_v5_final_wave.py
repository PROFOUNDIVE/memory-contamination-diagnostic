from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PROCESS_RACES = "tests/test_phase12_filter_v5_rootless_process_races.py"


def _module():
    path = ROOT / "scripts/run_phase12_filter_v5_final_wave.py"
    spec = importlib.util.spec_from_file_location("phase12_final_wave", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_sentinel_roles_and_targets_are_the_exact_final_wave_contract() -> None:
    module = _module()

    assert [spec.role for spec in module.SENTINEL_SPECS] == [
        "f1-pytest",
        "f2-pytest",
        "f3-pytest",
        "f4-rootless-pytest",
        "f4-ruff",
        "f4-validate-config",
        "f4-replay-pytest",
    ]
    assert module.SENTINEL_SPECS[0].target[-6:] == (
        "tests/test_phase12_filter_v5_rootless_legacy_fence.py",
        "tests/test_phase12_filter_v5_rootless_binding.py",
        "tests/test_phase12_filter_v5_rootless_external_authority.py",
        "tests/test_phase12_filter_v5_rootless_firewall.py",
        "tests/test_phase12_filter_v5_rootless_offline_qa.py",
        "-q",
    )
    assert module.SENTINEL_SPECS[-1].target[-6:] == (
        "tests/test_task_verifiers.py",
        "tests/test_cli_run.py",
        "tests/test_contamination_catalog.py",
        "tests/test_openai_compatible_client.py",
        "tests/test_aggregate.py",
        "-q",
    )


def test_f4_rootless_glob_is_sorted_and_excludes_only_process_races() -> None:
    module = _module()

    targets = module.f4_rootless_targets(ROOT)
    rootless = [target for target in targets if target.startswith("tests/test_phase12_filter_v5_")]

    assert rootless == sorted(rootless, key=str.encode)
    assert PROCESS_RACES not in rootless
    assert "tests/test_phase12_filter_v5_rootless_external_authority.py" in rootless
    assert "tests/test_phase12_filter_v5_final_wave.py" in rootless


def test_synthetic_paid_and_skip_lineages_are_hash_bound_and_zero_egress(
    tmp_path: Path,
) -> None:
    module = _module()
    repository = tmp_path / "repo"
    repository.mkdir(mode=0o700)

    fixtures = module.build_synthetic_task7_fixtures(
        repository,
        execution_commit="a" * 40,
        created_at="2026-08-09T00:00:00Z",
    )

    paid = module.validate_fixture_lineage(fixtures.paid)
    skipped = module.validate_fixture_lineage(fixtures.skipped)
    publication = json.loads(fixtures.paid.source.read_bytes())
    skip = json.loads(fixtures.skipped.source.read_bytes())

    assert paid.outcome == "paid_attempt"
    assert paid.provider_calls_issued == 2
    assert publication["transport_mode"] == "fake"
    assert publication["terminal"] == "LOCAL_ROOTLESS_BCT_REVIEW_REQUIRED"
    assert skipped.outcome == "zero_call_skip"
    assert skipped.provider_calls_issued == 0
    assert skip["reason"] == "ROOTLESS_MISSING_SECRET"
    assert skip["missing_input_role"] == "OPENAI_API_KEY"


def test_evidence_envelopes_and_index_preserve_closed_array_order(tmp_path: Path) -> None:
    module = _module()
    repository = tmp_path / "repo"
    repository.mkdir(mode=0o700)
    module._prepare_pre_egress(repository, "b" * 40, "2026-08-09T00:00:00Z")
    fixtures = module.build_synthetic_task7_fixtures(
        repository,
        execution_commit="b" * 40,
        created_at="2026-08-09T00:00:00Z",
    )
    sentinel = repository / "runs/phase12-filter-v5-rootless-qa/final/sentinels/f1-pytest.json"
    module.write_json(sentinel, {"role": "f1-pytest"})

    envelope = module.evidence_envelope("F1", (module.evidence(repository, "network-sentinel-f1-pytest", sentinel),))
    index = module.final_index_fixture(
        repository,
        fixtures.skipped,
        pre_egress_paths=module.PRE_EGRESS_PATHS[:3],
        final_paths=(("network-sentinel-f1-pytest", sentinel.relative_to(repository).as_posix()),),
        execution_commit="b" * 40,
        legacy_input_manifest_sha256="c" * 64,
        created_at="2026-08-09T00:00:00Z",
    )

    assert envelope["schema_version"] == "rootless_final_wave_evidence_v1"
    assert envelope["profile"] == "local_rootless_non_authoritative"
    assert envelope["ordered_input_evidence"][0]["role"] == "network-sentinel-f1-pytest"
    assert [item["role"] for item in index["ordered_pre_egress_evidence"]] == sorted(
        (role for role, _ in module.PRE_EGRESS_PATHS[:3]), key=str.encode
    )
    assert [item["role"] for item in index["ordered_final_evidence"]] == [
        "network-sentinel-f1-pytest"
    ]
    assert index["publication_receipt_sha256"] is None
    assert index["state_inventory_sha256"] is None
