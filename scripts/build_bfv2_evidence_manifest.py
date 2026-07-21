from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest(config_path: Path, run_dir: Path, inspector_output: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    fixture_path = config.get("replay", {}).get("fixture_path")
    artifacts = [config_path, run_dir / "run.json", run_dir / "resolved_config.json", run_dir / "provider_profile.json", run_dir / "trials.jsonl", run_dir / "calls.jsonl", run_dir / "failures.jsonl", run_dir / "memory_events.jsonl", inspector_output]
    if isinstance(fixture_path, str):
        artifacts.append((config_path.parent / fixture_path).resolve())
    prompt_dir = config_path.parents[1] / "tests" / "fixtures" / "prompts" / "baseline_fidelity_v2"
    artifacts.extend(sorted(prompt_dir.glob("*.json")))
    missing = [str(path) for path in artifacts if not path.is_file()]
    if missing:
        raise ValueError(f"missing evidence artifact: {', '.join(missing)}")
    return {
        "fidelity_gate_layer": config["run"]["fidelity_gate_layer"],
        "artifacts": {str(path): _sha256(path) for path in artifacts},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Hash F1A or F1B replay evidence artifacts.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--inspector-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest(args.config, args.run_dir, args.inspector_output)
    args.output.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
