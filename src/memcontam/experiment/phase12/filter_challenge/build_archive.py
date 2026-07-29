from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Final

from pydantic import BaseModel

from memcontam.experiment.phase12.filter_challenge.build_archive_models import (
    ArchiveValidationRequest,
    ArtifactBinding,
    BuildArchiveReport,
    BuildArchiveRequest,
    BuildArchiveRun,
    BuildArchiveSeal,
    PublicArtifactManifest,
)
from memcontam.experiment.phase12.filter_challenge.mft import (
    MergedMftReport,
    build_mft_report,
)
from memcontam.experiment.phase12.filter_challenge.registry import (
    validate_registry_closure,
)
from memcontam.experiment.phase12.filter_challenge.registry_manifests import (
    OperationalSuiteRegistry,
    ProbeInventoryRegistry,
)
from memcontam.experiment.phase12.filter_challenge.registry_search import SearchConfig


PUBLIC_FILES: Final = (
    "run.json", "inputs/search_config.json", "inputs/probe_inventory_manifest.json",
    "inputs/operational_suite_manifest.json", "mft.json",
)
ALL_FILES: Final = (*PUBLIC_FILES, "public_artifact_manifest.json", "archive_seal.json")
_HEX_40: Final = re.compile(r"[0-9a-f]{40}")
_HEX_64: Final = re.compile(r"[0-9a-f]{64}")
_SAFE_COMPONENT: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class BuildArchiveError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def build_archive(request: BuildArchiveRequest) -> BuildArchiveReport:
    _validate_binding_inputs(
        request.implementation_commit, request.freeze_id, request.run_id
    )
    validate_registry_closure(request.search_config, request.inventory, request.suite)
    target = request.output_root / request.run_id
    if target.exists():
        raise BuildArchiveError("ARCHIVE_ROOT_EXISTS")
    request.output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{request.run_id}.tmp-", dir=request.output_root))
    try:
        _write_archive(staging, request)
        report = validate_archive(
            ArchiveValidationRequest(
                archive=staging,
                expected_implementation_commit=request.implementation_commit,
                expected_search_config_hash=request.search_config.search_config_hash,
            )
        )
        staging.rename(target)
        return report
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def validate_archive(request: ArchiveValidationRequest) -> BuildArchiveReport:
    _require_hex(request.expected_implementation_commit, 40, "IMPLEMENTATION_COMMIT_INVALID")
    _require_hex(request.expected_search_config_hash, 64, "SEARCH_CONFIG_HASH_INVALID")
    files = {
        path.relative_to(request.archive).as_posix()
        for path in request.archive.rglob("*")
        if path.is_file()
    }
    if files != set(ALL_FILES):
        raise BuildArchiveError("ARCHIVE_STREAM_SET_INVALID")
    run = BuildArchiveRun.model_validate_json(
        (request.archive / "run.json").read_text(encoding="utf-8")
    )
    _require_component(run.freeze_id, "FREEZE_ID_INVALID")
    _require_component(run.run_id, "RUN_ID_INVALID")
    search = SearchConfig.model_validate_json(
        (request.archive / "inputs" / "search_config.json").read_text(encoding="utf-8")
    )
    inventory = ProbeInventoryRegistry.model_validate_json(
        (request.archive / "inputs" / "probe_inventory_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    suite = OperationalSuiteRegistry.model_validate_json(
        (request.archive / "inputs" / "operational_suite_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    mft = MergedMftReport.model_validate_json(
        (request.archive / "mft.json").read_text(encoding="utf-8")
    )
    for path, model in (
        (request.archive / "run.json", run),
        (request.archive / "inputs" / "search_config.json", search),
        (request.archive / "inputs" / "probe_inventory_manifest.json", inventory),
        (request.archive / "inputs" / "operational_suite_manifest.json", suite),
        (request.archive / "mft.json", mft),
    ):
        _require_canonical(path, model)
    validate_registry_closure(search, inventory, suite)
    if run.implementation_commit != request.expected_implementation_commit:
        raise BuildArchiveError("IMPLEMENTATION_COMMIT_MISMATCH")
    if run.search_config_hash != request.expected_search_config_hash:
        raise BuildArchiveError("SEARCH_CONFIG_HASH_MISMATCH")
    _validate_run_bindings(run, search, inventory, suite, mft)
    manifest = PublicArtifactManifest.model_validate_json(
        (request.archive / "public_artifact_manifest.json").read_text(encoding="utf-8")
    )
    _require_canonical(request.archive / "public_artifact_manifest.json", manifest)
    if set(manifest.artifacts) != set(PUBLIC_FILES) or any(
        manifest.artifacts[name].sha256 != _sha256(request.archive / name)
        for name in PUBLIC_FILES
    ):
        raise BuildArchiveError("ARCHIVE_HASH_MISMATCH")
    seal_path = request.archive / "archive_seal.json"
    seal = BuildArchiveSeal.model_validate_json(seal_path.read_text(encoding="utf-8"))
    _require_canonical(seal_path, seal)
    manifest_hash = _sha256(request.archive / "public_artifact_manifest.json")
    if (
        seal.public_artifact_manifest_sha256 != manifest_hash
        or (seal.implementation_commit, seal.freeze_id, seal.run_id, seal.search_config_hash)
        != (run.implementation_commit, run.freeze_id, run.run_id, run.search_config_hash)
    ):
        raise BuildArchiveError("ARCHIVE_SEAL_MISMATCH")
    return _report(run, manifest_hash, _sha256(seal_path))


def _write_archive(root: Path, request: BuildArchiveRequest) -> None:
    closure = validate_registry_closure(request.search_config, request.inventory, request.suite)
    run = BuildArchiveRun(
        implementation_commit=request.implementation_commit,
        freeze_id=request.freeze_id,
        run_id=request.run_id,
        search_config_id=closure.search_config_id,
        search_config_hash=request.search_config.search_config_hash,
        calibration_probe_inventory_id=closure.calibration_probe_inventory_id,
        calibration_probe_inventory_manifest_hash=closure.calibration_probe_inventory_manifest_hash,
        operational_probe_suite_manifest_id=closure.operational_probe_suite_manifest_id,
        operational_probe_suite_manifest_hash=closure.operational_probe_suite_manifest_hash,
    )
    (root / "inputs").mkdir()
    _write_json(root / "run.json", run)
    _write_json(root / "inputs" / "search_config.json", request.search_config)
    _write_json(root / "inputs" / "probe_inventory_manifest.json", request.inventory)
    _write_json(root / "inputs" / "operational_suite_manifest.json", request.suite)
    _write_json(
        root / "mft.json",
        build_mft_report(request.search_config, request.inventory, request.suite),
    )
    manifest = PublicArtifactManifest(
        artifacts={name: ArtifactBinding(sha256=_sha256(root / name)) for name in PUBLIC_FILES}
    )
    _write_json(root / "public_artifact_manifest.json", manifest)
    _write_json(
        root / "archive_seal.json",
        BuildArchiveSeal(
            public_artifact_manifest_sha256=_sha256(root / "public_artifact_manifest.json"),
            implementation_commit=run.implementation_commit,
            freeze_id=run.freeze_id,
            run_id=run.run_id,
            search_config_hash=run.search_config_hash,
        ),
    )


def _validate_run_bindings(
    run: BuildArchiveRun,
    search: SearchConfig,
    inventory: ProbeInventoryRegistry,
    suite: OperationalSuiteRegistry,
    mft: MergedMftReport,
) -> None:
    if (
        (run.search_config_id, run.search_config_hash)
        != (search.registry_id, search.search_config_hash)
        or (
            run.calibration_probe_inventory_id,
            run.calibration_probe_inventory_manifest_hash,
        )
        != (inventory.registry_id, inventory.calibration_probe_inventory_manifest_hash)
        or (
            run.operational_probe_suite_manifest_id,
            run.operational_probe_suite_manifest_hash,
        )
        != (suite.registry_id, suite.operational_probe_suite_manifest_hash)
        or (
            mft.search_config_id,
            mft.search_config_hash,
            mft.calibration_probe_inventory_id,
            mft.calibration_probe_inventory_manifest_hash,
            mft.operational_probe_suite_manifest_id,
            mft.operational_probe_suite_manifest_hash,
        )
        != (
            run.search_config_id,
            run.search_config_hash,
            run.calibration_probe_inventory_id,
            run.calibration_probe_inventory_manifest_hash,
            run.operational_probe_suite_manifest_id,
            run.operational_probe_suite_manifest_hash,
        )
        or not mft.all_passed
    ):
        raise BuildArchiveError("ARCHIVE_BINDING_MISMATCH")


def _validate_binding_inputs(implementation_commit: str, freeze_id: str, run_id: str) -> None:
    _require_hex(implementation_commit, 40, "IMPLEMENTATION_COMMIT_INVALID")
    _require_component(freeze_id, "FREEZE_ID_INVALID")
    _require_component(run_id, "RUN_ID_INVALID")


def _require_hex(value: str, length: int, code: str) -> None:
    pattern = _HEX_40 if length == 40 else _HEX_64
    if pattern.fullmatch(value) is None:
        raise BuildArchiveError(code)


def _require_component(value: str, code: str) -> None:
    if _SAFE_COMPONENT.fullmatch(value) is None or value in {".", ".."}:
        raise BuildArchiveError(code)


def _write_json(path: Path, model: BaseModel) -> None:
    path.write_text(_canonical_text(model), encoding="utf-8")


def _require_canonical(path: Path, model: BaseModel) -> None:
    if path.read_text(encoding="utf-8") != _canonical_text(model):
        raise BuildArchiveError("ARCHIVE_CANONICAL_JSON_REQUIRED")


def _canonical_text(model: BaseModel) -> str:
    return json.dumps(
        model.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _report(run: BuildArchiveRun, manifest_hash: str, seal_hash: str) -> BuildArchiveReport:
    return BuildArchiveReport(
        implementation_commit=run.implementation_commit,
        freeze_id=run.freeze_id,
        run_id=run.run_id,
        search_config_id=run.search_config_id,
        search_config_hash=run.search_config_hash,
        calibration_probe_inventory_id=run.calibration_probe_inventory_id,
        calibration_probe_inventory_manifest_hash=run.calibration_probe_inventory_manifest_hash,
        operational_probe_suite_manifest_id=run.operational_probe_suite_manifest_id,
        operational_probe_suite_manifest_hash=run.operational_probe_suite_manifest_hash,
        public_artifact_manifest_hash=manifest_hash,
        archive_seal_hash=seal_hash,
    )


__all__ = (
    "ALL_FILES", "ArchiveValidationRequest", "BuildArchiveError", "BuildArchiveReport",
    "BuildArchiveRequest", "build_archive", "validate_archive",
)
