from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Final

from datasets import Dataset, load_from_disk

from memcontam.main_registry import FrozenTaskPool, SourceValue, Task, freeze_task_pool

ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_BOT_ROOT: Final = Path("/home/hyunwoo/git/buffer-of-thought-llm")
DEFAULT_DC_ROOT: Final = Path("/home/hyunwoo/git/dynamic-cheatsheet")


class MainRegistryBuildError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("git", *arguments),
        cwd=root,
        env={**os.environ, "GIT_MASTER": "1"},
        check=False,
        capture_output=True,
        text=True,
    )


def _source_identity(repository: Path, relative_path: Path, content_hash: str) -> dict[str, str | bool]:
    commit = _git(repository, "rev-parse", "HEAD")
    tracked = _git(repository, "ls-files", "--error-unmatch", str(relative_path))
    clean = _git(repository, "diff", "--quiet", "--", str(relative_path))
    status = _git(repository, "status", "--porcelain")
    if commit.returncode or tracked.returncode or clean.returncode:
        raise MainRegistryBuildError("APPROVED_SOURCE_IDENTITY_INVALID")
    return {
        "repository": str(repository),
        "commit": commit.stdout.strip(),
        "repository_worktree_clean": not status.stdout,
        "approved_path": str(repository / relative_path),
        "approved_path_clean": True,
        "content_sha256": content_hash,
    }


def _jsonl_rows(path: Path) -> tuple[dict[str, SourceValue], ...]:
    rows: list[dict[str, SourceValue]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw:
            continue
        value = json.loads(raw)
        if not isinstance(value, dict) or any(not isinstance(item, (str, int)) for item in value.values()):
            raise MainRegistryBuildError("APPROVED_JSONL_SOURCE_INVALID")
        rows.append(value)
    return tuple(rows)


def _meb_rows(path: Path) -> tuple[dict[str, SourceValue], ...]:
    loaded = load_from_disk(str(path))
    if not isinstance(loaded, Dataset) or loaded.column_names != ["input", "target", "target_value"]:
        raise MainRegistryBuildError("APPROVED_MEB_SOURCE_INVALID")
    return tuple(
        {
            "input": str(loaded[index]["input"]),
            "target": str(loaded[index]["target"]),
            "target_value": int(loaded[index]["target_value"]),
        }
        for index in range(loaded.num_rows)
    )


def _write_registry(path: Path, registry: FrozenTaskPool) -> str:
    raw = b"".join(
        (
            json.dumps(row.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        for row in registry.rows
    )
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def build(bot_root: Path, dc_root: Path, output_root: Path) -> dict[str, object]:
    exclusions_path = ROOT / "data/phase13/main/exclusions_v1.json"
    exclusions_payload = json.loads(exclusions_path.read_text(encoding="utf-8"))
    excluded = exclusions_payload["excluded_signatures"]
    game_path = bot_root / "benchmarks/gameof24.jsonl"
    words_path = bot_root / "benchmarks/word_sorting.jsonl"
    meb_path = dc_root / "data/MathEquationBalancer"
    state = json.loads((meb_path / "state.json").read_text(encoding="utf-8"))
    data_files = tuple(item["filename"] for item in state["_data_files"])
    if data_files != ("data-00000-of-00001.arrow",):
        raise MainRegistryBuildError("APPROVED_MEB_SOURCE_INVALID")

    pools: dict[Task, FrozenTaskPool] = {
        "game24": freeze_task_pool(
            task="game24",
            rows=_jsonl_rows(game_path),
            excluded_signatures=frozenset(excluded["game24"]),
        ),
        "math_equation_balancer": freeze_task_pool(
            task="math_equation_balancer",
            rows=_meb_rows(meb_path),
            excluded_signatures=frozenset(excluded["math_equation_balancer"]),
        ),
        "word_sorting": freeze_task_pool(
            task="word_sorting",
            rows=_jsonl_rows(words_path),
            excluded_signatures=frozenset(excluded["word_sorting"]),
        ),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    registry_paths: dict[Task, Path] = {
        task: output_root / f"{task}_main_v1.jsonl"
        for task in pools
    }
    registry_hashes = {
        task: _write_registry(registry_paths[task], pool)
        for task, pool in pools.items()
    }
    meb_file_hashes = {
        name: _sha256(meb_path / name)
        for name in ("dataset_info.json", "state.json", *data_files)
    }
    manifest: dict[str, object] = {
        "schema_version": "phase13_reduced_main_registry_manifest_v1",
        "selection_law": "source_order_after_prospective_canonical_exclusion_v1",
        "exclusion_registry": {
            "path": str(exclusions_path.relative_to(ROOT)),
            "sha256": _sha256(exclusions_path),
        },
        "sources": {
            "game24": _source_identity(
                bot_root, Path("benchmarks/gameof24.jsonl"), _sha256(game_path)
            ),
            "word_sorting": _source_identity(
                bot_root, Path("benchmarks/word_sorting.jsonl"), _sha256(words_path)
            ),
            "math_equation_balancer": {
                **_source_identity(
                    dc_root,
                    Path("data/MathEquationBalancer/data-00000-of-00001.arrow"),
                    meb_file_hashes[data_files[0]],
                ),
                "dataset_fingerprint": state["_fingerprint"],
                "load_contract": "datasets.load_from_disk",
                "dataset_files": meb_file_hashes,
            },
        },
        "registries": {
            task: {
                "path": path.name,
                "sha256": registry_hashes[task],
                "source_count": pools[task].source_count,
                "main_count": len(pools[task].rows),
                "exclusions": [item.model_dump(mode="json") for item in pools[task].exclusions],
            }
            for task, path in registry_paths.items()
        },
    }
    manifest_path = output_root / "main_registry_manifest_v1.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bot-root", type=Path, default=DEFAULT_BOT_ROOT)
    parser.add_argument("--dc-root", type=Path, default=DEFAULT_DC_ROOT)
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/phase13/main")
    arguments = parser.parse_args()
    manifest = build(arguments.bot_root, arguments.dc_root, arguments.output_root)
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
