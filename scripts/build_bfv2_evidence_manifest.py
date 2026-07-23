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
    root = config_path.parents[1]
    fixture_path = config.get("replay", {}).get("fixture_path")
    artifacts = [
        config_path,
        run_dir / "run.json",
        run_dir / "resolved_config.json",
        run_dir / "provider_profile.json",
        run_dir / "trials.jsonl",
        run_dir / "calls.jsonl",
        run_dir / "failures.jsonl",
        run_dir / "filter_events.jsonl",
        run_dir / "memory_events.jsonl",
        inspector_output,
    ]
    if isinstance(fixture_path, str):
        artifacts.append((config_path.parent / fixture_path).resolve())
    prompt_dir = root / "tests" / "fixtures" / "prompts" / "baseline_fidelity_v2"
    artifacts.extend(sorted(prompt_dir.glob("*.json")))
    artifacts.append(root / "tests" / "fixtures" / "baseline_fidelity_v2_semantic_call_hashes.json")
    plan_path = root / ".sisyphus" / "plans" / "BASELINE-FIDELITY-V2_source-contract_remediation.md"
    artifacts.append(plan_path)
    missing = [str(path) for path in artifacts if not path.is_file()]
    if missing:
        raise ValueError(f"missing evidence artifact: {', '.join(missing)}")
    run_metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["run_metadata"]
    resolved = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    provider_profile = json.loads((run_dir / "provider_profile.json").read_text(encoding="utf-8"))
    corpus_path = config.get("memory", {}).get("corpus_manifest_path")
    corpus_manifest = (
        json.loads((root / corpus_path).read_text(encoding="utf-8"))
        if isinstance(corpus_path, str)
        else None
    )
    inspector = json.loads(inspector_output.read_text(encoding="utf-8"))
    return {
        "fidelity_gate_layer": config["run"]["fidelity_gate_layer"],
        "commit": run_metadata["git_commit"],
        "plan": {"path": str(plan_path), "sha256": _sha256(plan_path)},
        "versions": {
            "prompt_version": resolved["logging"]["prompt_version"],
            "memory_policy_version": resolved["logging"]["memory_policy_version"],
            "retry_policy_version": resolved["run"]["retry_policy_version"],
            "baseline_execution_contract_version": resolved["run"][
                "baseline_execution_contract_version"
            ],
            "failure_taxonomy_version": resolved["run"]["failure_taxonomy_version"],
        },
        "embedding_identity": provider_profile,
        "corpus_identity": corpus_manifest,
        "commands": [
            {
                "command": "inspect_baseline_fidelity_v2",
                "exit_code": 0 if inspector.get("overall") == "pass" else 1,
            }
        ],
        "strict_streams": [
            "trials.jsonl",
            "calls.jsonl",
            "failures.jsonl",
            "filter_events.jsonl",
            "memory_events.jsonl",
        ],
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
