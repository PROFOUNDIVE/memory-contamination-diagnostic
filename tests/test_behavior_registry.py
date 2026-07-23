from __future__ import annotations

import json
import importlib
import shutil
from pathlib import Path

import pytest

from memcontam.experiment.phase12.contracts import canonical_json_hash


REGISTRIES = Path(__file__).parents[1] / "data" / "phase12" / "registries"
REQUIRED_IDS = (
    "MFT-01",
    "MFT-02",
    "MFT-03",
    "MFT-04",
    "INV-01",
    "INV-02",
    "INV-03",
    "DIR-01",
    "DIR-02",
    "DIR-03",
    "DIR-04",
)


def _copy_registries(tmp_path: Path) -> Path:
    root = tmp_path / f"registries-{len(list(tmp_path.iterdir()))}"
    shutil.copytree(REGISTRIES, root)
    return root


def _read_rows(root: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (root / "behavior_tests.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]


def _write_rows(root: Path, rows: list[dict[str, object]]) -> None:
    (root / "behavior_tests.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _refresh_row_hash(row: dict[str, object]) -> None:
    row["row_hash"] = canonical_json_hash({key: value for key, value in row.items() if key != "row_hash"})


def test_loads_complete_frozen_registry_bundle() -> None:
    registry = importlib.import_module("memcontam.behavior.registry")

    bundle = registry.load_behavior_registry_bundle(REGISTRIES)
    report = registry.validate_registry_bundle(bundle)

    assert tuple(row.test_id for row in bundle.behavior_tests.rows) == REQUIRED_IDS
    assert all(row.row_hash != "PENDING" for row in bundle.behavior_tests.rows)
    assert report.valid is True
    assert report.bundle_hash == bundle.artifact_hash
    inv03 = next(row for row in bundle.behavior_tests.rows if row.test_id == "INV-03")
    assert inv03.auxiliary_contract_refs["inv03_equivalence_registry_id"] == (
        "inv03-equivalence-registry-v1"
    )
    assert bundle.fidelity_certificate.overall_status == "blocked"


def test_rejects_missing_duplicate_stale_or_outcome_tuned_registry_rows(tmp_path: Path) -> None:
    registry = importlib.import_module("memcontam.behavior.registry")

    root = _copy_registries(tmp_path)
    rows = _read_rows(root)
    _write_rows(root, rows[:-1])
    with pytest.raises(registry.BehaviorRegistryError, match="BEHAVIOR_REGISTRY_INCOMPLETE"):
        registry.load_behavior_registry_bundle(root)

    root = _copy_registries(tmp_path)
    rows = _read_rows(root)
    rows.append(rows[0])
    _write_rows(root, rows)
    with pytest.raises(registry.BehaviorRegistryError, match="REGISTRY_DUPLICATE_ID"):
        registry.load_behavior_registry_bundle(root)

    root = _copy_registries(tmp_path)
    rows = _read_rows(root)
    rows[0]["transformation_generator_version"] = "mutated-generator-v1"
    rows[0]["row_hash"] = "stale"
    _write_rows(root, rows)
    with pytest.raises(registry.BehaviorRegistryError, match="REGISTRY_HASH_MISMATCH"):
        registry.load_behavior_registry_bundle(root)

    root = _copy_registries(tmp_path)
    rows = _read_rows(root)
    rows[0]["row_hash"] = "PENDING"
    _write_rows(root, rows)
    with pytest.raises(registry.BehaviorRegistryError, match="REGISTRY_HASH_MISMATCH"):
        registry.load_behavior_registry_bundle(root)

    root = _copy_registries(tmp_path)
    rows = _read_rows(root)
    rows[0]["auxiliary_contract_refs"] = {"selection_basis": "observed_outcomes"}
    _refresh_row_hash(rows[0])
    _write_rows(root, rows)
    with pytest.raises(registry.BehaviorRegistryError, match="OUTCOME_TUNED_REGISTRY"):
        registry.load_behavior_registry_bundle(root)

    root = _copy_registries(tmp_path)
    rows = _read_rows(root)
    inv03 = next(row for row in rows if row["test_id"] == "INV-03")
    inv03["auxiliary_contract_refs"] = {
        "inv03_equivalence_registry_id": "inv03-equivalence-registry-v1",
        "inv03_equivalence_registry_hash": "stale",
    }
    _refresh_row_hash(inv03)
    _write_rows(root, rows)
    with pytest.raises(registry.BehaviorRegistryError, match="INV03_EQUIVALENCE_CONTRACT_MISMATCH"):
        registry.load_behavior_registry_bundle(root)
