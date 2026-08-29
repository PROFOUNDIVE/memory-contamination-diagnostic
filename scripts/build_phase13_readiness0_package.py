from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from memcontam.readiness.phase13_readiness0_f1c_report import (
    build_f1c_registry,
    build_f1c_report,
)
from memcontam.readiness.phase13_readiness0_live import READINESS0_CASES
from memcontam.readiness.phase13_readiness0_live_models import ArtifactBinding, LiveRequest
from memcontam.readiness.phase13_readiness0_package import (
    build_implementation_manifest,
    build_window_proof,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--cache-root", type=Path)
    args = parser.parse_args()
    repository_root = args.repository_root.resolve()
    root = repository_root / "data/phase13/main/mr_p4"
    cache_root = args.cache_root or Path(os.environ["MEMCONTAM_BGE_CACHE_DIR"])
    implementation = build_implementation_manifest(repository_root)
    window = build_window_proof(root, repository_root)
    report = build_f1c_report(repository_root, cache_root)
    report_raw = (report.model_dump_json(indent=2) + "\n").encode()
    legacy_raw = (repository_root / "data/phase13/rag/legacy/manifest.json").read_bytes()
    registry = build_f1c_registry(report_raw, legacy_raw)
    outputs = {
        "readiness0_live_implementation_manifest_v1.json": implementation,
        "readiness0_window_proof_v1.json": window,
        "readiness0_f1c_report_v1.json": report,
        "readiness0_f1c_registry_v1.json": registry,
    }
    for name, model in outputs.items():
        (root / name).write_text(model.model_dump_json(indent=2) + "\n", encoding="utf-8")
    request = LiveRequest(
        schema_version="phase13_readiness0_live_request_v1",
        status="PRE_LIVE_AUTHORIZED",
        scientific_result=False,
        main_result=False,
        measured_main_a_trajectory_count=0,
        case_ids=tuple(case.case_id for case in READINESS0_CASES),
        maximum_provider_calls=12,
        implementation_manifest=_binding(
            "data/phase13/main/mr_p4/readiness0_live_implementation_manifest_v1.json",
            (root / "readiness0_live_implementation_manifest_v1.json").read_bytes(),
        ),
        window_proof=_binding(
            "data/phase13/main/mr_p4/readiness0_window_proof_v1.json",
            (root / "readiness0_window_proof_v1.json").read_bytes(),
        ),
        f1c_registry=_binding(
            "data/phase13/main/mr_p4/readiness0_f1c_registry_v1.json",
            (root / "readiness0_f1c_registry_v1.json").read_bytes(),
        ),
        core_manifest=_path_binding(repository_root, "data/phase13/core/materialized/manifest.json"),
        legacy_rag_manifest=_path_binding(repository_root, "data/phase13/rag/legacy/manifest.json"),
        checkpoint_registry=_path_binding(
            repository_root,
            "data/phase13/main/mr_p4/main_a_common_checkpoint_registry_v1.json",
        ),
        observability_packet=_path_binding(
            repository_root,
            "data/phase13/observability/registration_packet_v1.json",
        ),
        credentials_source="CURRENT_PROCESS_ENVIRONMENT_ONLY",
        request_hash="0" * 64,
    )
    payload = request.model_dump(mode="json", exclude={"request_hash"})
    request = request.model_copy(update={"request_hash": _canonical_hash(payload)})
    (root / "readiness0_live_request_v1.json").write_text(
        request.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )


def _path_binding(root: Path, path: str) -> ArtifactBinding:
    return _binding(path, (root / path).read_bytes())


def _binding(path: str, raw: bytes) -> ArtifactBinding:
    return ArtifactBinding(path=path, sha256=hashlib.sha256(raw).hexdigest())


def _canonical_hash(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


if __name__ == "__main__":
    main()
