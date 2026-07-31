from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from memcontam.experiment.phase12.filter_challenge.freeze_a import FreezeAError, validate_freeze_a


def _files(root: Path) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(root.rglob("*")) if path.is_file()}


def _canonical(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def _refresh_freeze(root: Path) -> None:
    freeze_path = root / "freeze_a.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze["manifest_sha256"]["probe_construction_manifest_v1.json"] = hashlib.sha256((root / "probe_construction_manifest_v1.json").read_bytes()).hexdigest()
    _canonical(freeze_path, freeze)


def _mutation_code(config: Path, source: Path, expected: Path, task: str, certificate: dict[str, object]) -> str:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "freeze"
        shutil.copytree(expected, root)
        path = root / "probe_construction_manifest_v1.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["probes"][task][0]["certificate"] = certificate
        _canonical(path, payload)
        _refresh_freeze(root)
        try:
            validate_freeze_a(config, source, root)
        except FreezeAError as error:
            return str(error)
    return "MUTATION_ACCEPTED"


def _source_code(config: Path, source: Path, expected: Path) -> str:
    with tempfile.TemporaryDirectory() as temporary:
        copied = Path(temporary) / "source.json"
        shutil.copyfile(source, copied)
        payload = json.loads(copied.read_text(encoding="utf-8"))
        payload["source_files"]["data/tasks/game24_pilot.jsonl"] = "0" * 64
        _canonical(copied, payload)
        try:
            validate_freeze_a(config, copied, expected)
        except FreezeAError as error:
            return str(error)
    return "MUTATION_ACCEPTED"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-universe", type=Path, required=True)
    parser.add_argument("--expected-root", type=Path, required=True)
    parser.add_argument("--repeat-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    payload = {
        "byte_equal": _files(arguments.expected_root) == _files(arguments.repeat_root),
        "mutations": {
            "game24": _mutation_code(arguments.config, arguments.source_universe, arguments.expected_root, "game24", {"input_canonical": "3,3,8,8"}),
            "meb": _mutation_code(arguments.config, arguments.source_universe, arguments.expected_root, "math_equation_balancer", {"input_canonical": "1,2,3,7"}),
            "words": _mutation_code(arguments.config, arguments.source_universe, arguments.expected_root, "word_sorting", {"input_canonical": "ayz|aza"}),
            "source": _source_code(arguments.config, arguments.source_universe, arguments.expected_root),
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    _canonical(arguments.output, payload)
    expected = {"game24": "LEAKAGE_PILOT_INSTANCE", "meb": "LEAKAGE_CANDIDATE_EXAMPLE", "words": "LEAKAGE_CANDIDATE_EXAMPLE", "source": "SOURCE_UNIVERSE_DIGEST_MISMATCH"}
    return 0 if payload["byte_equal"] and payload["mutations"] == expected else 2


if __name__ == "__main__":
    raise SystemExit(main())
