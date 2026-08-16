from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from memcontam.readiness import phase13_core_datasets as core_datasets
from memcontam.readiness.phase13_core_datasets import (
    GPQA_SOURCE_SHA256,
    GPQA_REVISION,
    MMLU_PRO_REVISION,
    MMLU_PRO_TEST_SHA256,
    SELECTION_PATH,
    CoreDatasetError,
    load_core_task,
    materialize_core_datasets,
    paired_trajectory_order,
    validate_core_datasets,
)
from memcontam.readiness.phase13_core_bundle import (
    CoreSources,
    CoreTask,
    SelectionProvenance,
    SourceArtifact,
    write_bundle,
)
from memcontam.tasks.base import TaskInstance


MMLU_SELECTION = SELECTION_PATH


def _selection_digest(question_ids: list[int]) -> str:
    payload = json.dumps(sorted(question_ids), separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _trust_tiny_bundle(bundle: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = json.loads((bundle / "manifest.json").read_text())
    monkeypatch.setattr(
        core_datasets,
        "CANONICAL_CORE_ARTIFACT_SHA256",
        {task: artifact["sha256"] for task, artifact in manifest["artifacts"].items()},
    )


def test_mmlu_selection_manifest_fixes_the_two_released_250_identity_sets() -> None:
    payload = json.loads(MMLU_SELECTION.read_text(encoding="utf-8"))

    engineering = payload["tasks"]["mmlu_pro_engineering"]["question_ids"]
    physics = payload["tasks"]["mmlu_pro_physics"]["question_ids"]

    assert payload["upstream_revision"] == MMLU_PRO_REVISION
    assert payload["dynamic_cheatsheet_revision"] == (
        "5cfe3c37e8e52b1d858d0f3df46e7f17c50991b9"
    )
    assert len(engineering) == len(set(engineering)) == 250
    assert len(physics) == len(set(physics)) == 250
    assert set(engineering).isdisjoint(physics)
    assert _selection_digest(engineering) == (
        "fd41e99416a523444b06e188a46d811f3fef0eb7549fa6a2c65d3815805bb539"
    )
    assert _selection_digest(physics) == (
        "3e81880fd1e04196b61b8aa51376031146eba94cd448b791c6c5bce86bf14ff4"
    )


def test_paired_trajectory_order_is_seeded_paired_and_task_isolated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _write_tiny_bundle(tmp_path)
    _trust_tiny_bundle(bundle, monkeypatch)
    engineering = load_core_task(bundle, "mmlu_pro_engineering")
    physics = load_core_task(bundle, "mmlu_pro_physics")

    first = paired_trajectory_order(engineering, trajectory_seed=17)
    paired = paired_trajectory_order(engineering, trajectory_seed=17)
    different_seed = paired_trajectory_order(engineering, trajectory_seed=18)
    other_task = paired_trajectory_order(physics, trajectory_seed=17)

    assert [row.sample_id for row in first] == [row.sample_id for row in paired]
    assert [row.sample_id for row in first] != [row.sample_id for row in different_seed]
    assert {row.sample_id for row in first} == {row.sample_id for row in engineering}
    assert {row.task_name for row in first} == {"mmlu_pro_engineering"}
    assert {row.task_name for row in other_task} == {"mmlu_pro_physics"}
    assert set(row.sample_id for row in first).isdisjoint(row.sample_id for row in other_task)


def test_core_loader_preserves_upstream_identity_without_exposing_gold_in_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _write_tiny_bundle(tmp_path)
    _trust_tiny_bundle(bundle, monkeypatch)

    row = load_core_task(bundle, "gpqa_diamond")[0]

    assert row.sample_id == "gpqa_diamond:record-1"
    assert row.metadata == {
        "dataset_repo": "Idavidrein/gpqa",
        "dataset_revision": GPQA_REVISION,
        "dataset_config": "gpqa_diamond",
        "dataset_split": "train",
        "upstream_record_id": "record-1",
        "upstream_row_index": 0,
        "category": "Physics",
        "source": "gpqa_diamond.csv",
        "choice_order_version": "sha256_record_v1",
    }
    assert row.input == {"question": "local gated question", "options": ["A", "B", "C", "D"]}
    assert row.verifier_spec == {"answer_index": 2, "answer_label": "C"}
    assert "answer" not in row.input


def test_materialization_fails_closed_when_gpqa_access_is_unavailable(tmp_path: Path) -> None:
    def denied_download(
        *,
        repo_id: str,
        repo_type: str,
        filename: str,
        revision: str,
        token: bool | None = None,
        cache_dir: str | None = None,
    ) -> str:
        del repo_id, repo_type, filename, revision, token, cache_dir
        raise PermissionError("gated")

    with pytest.raises(CoreDatasetError, match="GPQA_ACCESS_REQUIRED"):
        materialize_core_datasets(tmp_path / "bundle", download=denied_download)


def test_materialization_rejects_unignored_repository_output_before_download() -> None:
    called = False

    def unexpected_download(
        *,
        repo_id: str,
        repo_type: str,
        filename: str,
        revision: str,
        token: bool | None = None,
        cache_dir: str | None = None,
    ) -> str:
        del repo_id, repo_type, filename, revision, token, cache_dir
        nonlocal called
        called = True
        raise AssertionError("download must not run")

    root = Path(__file__).resolve().parents[1]
    with pytest.raises(CoreDatasetError, match="CORE_DATASET_OUTPUT_NOT_PROTECTED"):
        materialize_core_datasets(
            root / "data/phase13/core/custom-output",
            download=unexpected_download,
        )
    assert called is False


def test_materialization_parses_private_snapshots_bound_to_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mmlu = tmp_path / "source.parquet"
    gpqa = tmp_path / "source.csv"
    mmlu.write_bytes(b"fixed mmlu bytes")
    gpqa.write_bytes(b"fixed gpqa bytes")
    monkeypatch.setattr(core_datasets, "MMLU_PRO_TEST_SHA256", hashlib.sha256(mmlu.read_bytes()).hexdigest())
    monkeypatch.setattr(core_datasets, "GPQA_SOURCE_SHA256", hashlib.sha256(gpqa.read_bytes()).hexdigest())

    private_paths: list[Path] = []

    def download(**kwargs: object) -> str:
        private_paths.append(Path(str(kwargs["cache_dir"])))
        return str(gpqa if kwargs["filename"] == "gpqa_diamond.csv" else mmlu)

    def build_rows(
        mmlu_path: Path,
        gpqa_path: Path,
        selection_path: Path,
        *,
        cache_dir: Path,
    ) -> dict:
        private_paths.append(cache_dir)
        assert mmlu_path != mmlu
        assert gpqa_path != gpqa
        assert selection_path != SELECTION_PATH
        mmlu.write_bytes(b"mutated after snapshot")
        gpqa.write_bytes(b"mutated after snapshot")
        return {
            "mmlu_pro_engineering": (
                TaskInstance.model_validate(_row("mmlu_pro_engineering", "e", 0)),
            ),
            "mmlu_pro_physics": (
                TaskInstance.model_validate(_row("mmlu_pro_physics", "p", 0)),
            ),
            "gpqa_diamond": (TaskInstance.model_validate(_row("gpqa_diamond", "g", 0)),),
        }

    from memcontam.readiness import phase13_core_dataset_materialization as materialization

    monkeypatch.setattr(materialization, "build_rows", build_rows)
    manifest = materialize_core_datasets(tmp_path / "bundle", download=download)

    assert manifest.sources.mmlu_pro.sha256 == hashlib.sha256(b"fixed mmlu bytes").hexdigest()
    assert manifest.sources.gpqa.sha256 == hashlib.sha256(b"fixed gpqa bytes").hexdigest()
    assert private_paths
    assert all(not path.exists() for path in private_paths)


def test_hugging_face_download_forces_xet_off(monkeypatch: pytest.MonkeyPatch) -> None:
    import huggingface_hub
    from huggingface_hub import constants
    from huggingface_hub.utils._runtime import is_xet_available

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **_kwargs: "/tmp/source")
    monkeypatch.setattr(constants, "HF_HUB_DISABLE_XET", False)
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)

    core_datasets._hf_download(
        repo_id="example/repo",
        repo_type="dataset",
        filename="source",
        revision="revision",
        cache_dir="/tmp/private-cache",
    )

    assert constants.HF_HUB_DISABLE_XET is True
    assert os.environ["HF_HUB_DISABLE_XET"] == "1"
    assert is_xet_available() is False


def test_bundle_publication_rejects_existing_root_and_symlinked_ancestor(tmp_path: Path) -> None:
    rows: dict[CoreTask, Sequence[TaskInstance]] = {
        "mmlu_pro_engineering": (TaskInstance.model_validate(_row("mmlu_pro_engineering", "e", 0)),),
        "mmlu_pro_physics": (TaskInstance.model_validate(_row("mmlu_pro_physics", "p", 0)),),
        "gpqa_diamond": (TaskInstance.model_validate(_row("gpqa_diamond", "g", 0)),),
    }
    sources = CoreSources(
        mmlu_pro=SourceArtifact(
            repo="TIGER-Lab/MMLU-Pro",
            revision=MMLU_PRO_REVISION,
            path="data/test-00000-of-00001.parquet",
            sha256=MMLU_PRO_TEST_SHA256,
        ),
        gpqa=SourceArtifact(
            repo="Idavidrein/gpqa",
            revision=GPQA_REVISION,
            path="gpqa_diamond.csv",
            sha256=GPQA_SOURCE_SHA256,
        ),
    )
    selection = SelectionProvenance(
        resource="memcontam.readiness/data/mmlu_pro_dc_selection_v1.json",
        sha256=hashlib.sha256(SELECTION_PATH.read_bytes()).hexdigest(),
        upstream_revision=MMLU_PRO_REVISION,
        dynamic_cheatsheet_revision="5cfe3c37e8e52b1d858d0f3df46e7f17c50991b9",
    )
    destination = tmp_path / "bundle"
    write_bundle(destination, rows, sources=sources, selection=selection)

    with pytest.raises(CoreDatasetError, match="CORE_DATASET_OUTPUT_EXISTS"):
        write_bundle(destination, rows, sources=sources, selection=selection)

    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "link"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(CoreDatasetError, match="CORE_DATASET_OUTPUT_SYMLINK"):
        write_bundle(link / "bundle", rows, sources=sources, selection=selection)


def test_repository_ignores_interrupted_core_bundle_staging() -> None:
    result = subprocess.run(
        ["git", "check-ignore", "-q", "data/phase13/core/.materialized.interrupted"],
        check=False,
    )

    assert result.returncode == 0


def test_bundle_validation_reports_task_counts_and_seeded_order_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _write_tiny_bundle(tmp_path)
    _trust_tiny_bundle(bundle, monkeypatch)

    report = validate_core_datasets(bundle, trajectory_seed=23, expected_counts={})

    assert report.trajectory_seed == 23
    assert len(report.manifest_sha256) == 64
    assert report.selection_sha256 == hashlib.sha256(SELECTION_PATH.read_bytes()).hexdigest()
    assert report.tasks["mmlu_pro_engineering"].rows == 5
    assert report.tasks["mmlu_pro_engineering"].artifact_sha256 == (
        json.loads((bundle / "manifest.json").read_text())["artifacts"]
        ["mmlu_pro_engineering"]["sha256"]
    )
    assert report.tasks["mmlu_pro_physics"].rows == 5
    assert report.tasks["gpqa_diamond"].rows == 1
    assert report.tasks["mmlu_pro_engineering"].order_sha256 != (
        report.tasks["mmlu_pro_physics"].order_sha256
    )


def test_bundle_manifest_and_seal_are_cryptographically_closed(tmp_path: Path) -> None:
    bundle = _write_tiny_bundle(tmp_path)
    manifest = bundle / "manifest.json"
    manifest.write_text(manifest.read_text() + " ", encoding="utf-8")

    with pytest.raises(CoreDatasetError, match="CORE_DATASET_SEAL_MISMATCH"):
        load_core_task(bundle, "mmlu_pro_engineering")


def test_bundle_validation_rejects_fabricated_rows_after_self_reseal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _write_tiny_bundle(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    monkeypatch.setattr(
        core_datasets,
        "CANONICAL_CORE_ARTIFACT_SHA256",
        {task: artifact["sha256"] for task, artifact in manifest["artifacts"].items()},
    )
    artifact_path = bundle / "mmlu_pro_engineering.jsonl"
    rows = artifact_path.read_text().splitlines()
    fabricated = json.loads(rows[0])
    fabricated["input"]["question"] = "fabricated replacement"
    rows[0] = json.dumps(fabricated, sort_keys=True)
    artifact_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    manifest["artifacts"]["mmlu_pro_engineering"]["sha256"] = hashlib.sha256(
        artifact_path.read_bytes()
    ).hexdigest()
    manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest_path.write_bytes(manifest_bytes)
    seal = {
        "schema_version": "phase13_core_dataset_seal_v1",
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }
    (bundle / "seal.json").write_text(
        json.dumps(seal, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )

    with pytest.raises(CoreDatasetError, match="CORE_DATASET_CANONICAL_MISMATCH"):
        validate_core_datasets(bundle, trajectory_seed=23, expected_counts={})
    with pytest.raises(CoreDatasetError, match="CORE_DATASET_CANONICAL_MISMATCH"):
        load_core_task(bundle, "mmlu_pro_engineering")


def _write_tiny_bundle(root: Path) -> Path:
    root.mkdir(exist_ok=True)
    rows = {
        "mmlu_pro_engineering": [
            _row("mmlu_pro_engineering", f"engineering:{index}", index)
            for index in range(5)
        ],
        "mmlu_pro_physics": [
            _row("mmlu_pro_physics", f"physics:{index}", index) for index in range(5)
        ],
        "gpqa_diamond": [
            {
                "sample_id": "gpqa_diamond:record-1",
                "task_name": "gpqa_diamond",
                "input": {"question": "local gated question", "options": ["A", "B", "C", "D"]},
                "verifier_spec": {"answer_index": 2, "answer_label": "C"},
                "metadata": {
                    "dataset_repo": "Idavidrein/gpqa",
                    "dataset_revision": GPQA_REVISION,
                    "dataset_config": "gpqa_diamond",
                    "dataset_split": "train",
                    "upstream_record_id": "record-1",
                    "upstream_row_index": 0,
                    "category": "Physics",
                    "source": "gpqa_diamond.csv",
                    "choice_order_version": "sha256_record_v1",
                },
            }
        ],
    }
    artifacts = {}
    for task, task_rows in rows.items():
        path = root / f"{task}.jsonl"
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in task_rows),
            encoding="utf-8",
        )
        artifacts[task] = {
            "path": path.name,
            "rows": len(task_rows),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    manifest = {
        "schema_version": "phase13_core_dataset_bundle_v1",
        "sources": {
            "mmlu_pro": {
                "repo": "TIGER-Lab/MMLU-Pro",
                "revision": MMLU_PRO_REVISION,
                "path": "data/test-00000-of-00001.parquet",
                "sha256": MMLU_PRO_TEST_SHA256,
            },
            "gpqa": {
                "repo": "Idavidrein/gpqa",
                "revision": GPQA_REVISION,
                "path": "gpqa_diamond.csv",
                "sha256": GPQA_SOURCE_SHA256,
            },
        },
        "selection": {
            "resource": "memcontam.readiness/data/mmlu_pro_dc_selection_v1.json",
            "sha256": hashlib.sha256(SELECTION_PATH.read_bytes()).hexdigest(),
            "upstream_revision": MMLU_PRO_REVISION,
            "dynamic_cheatsheet_revision": "5cfe3c37e8e52b1d858d0f3df46e7f17c50991b9",
        },
        "artifacts": artifacts,
    }
    manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    (root / "manifest.json").write_bytes(manifest_bytes)
    (root / "seal.json").write_text(
        json.dumps(
            {
                "schema_version": "phase13_core_dataset_seal_v1",
                "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    return root


def _row(task: str, sample_id: str, index: int) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "task_name": task,
        "input": {"question": f"question {index}", "options": ["A", "B", "C", "D"]},
        "verifier_spec": {"answer_index": 0, "answer_label": "A"},
        "metadata": {"upstream_row_index": index},
    }
