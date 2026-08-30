from __future__ import annotations

import argparse
from pathlib import Path

from memcontam.readiness.phase13_readiness0_current_status import (
    PreliveArtifactBytes,
)
from memcontam.readiness.phase13_readiness0_current_status_v2 import (
    ClosedReadiness0ArtifactBytes,
    build_current_readiness0_status_v2,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.repository_root / "data/phase13/main/mr_p4"
    evidence_root = root / "readiness0_live_evidence_v1"
    status = build_current_readiness0_status_v2(
        ClosedReadiness0ArtifactBytes(
            prelive=PreliveArtifactBytes(
                live_request=(root / "readiness0_live_request_v1.json").read_bytes(),
                live_authorization=(root / "readiness0_live_authorization_v1.json").read_bytes(),
                f1c_registry=(root / "readiness0_f1c_registry_v1.json").read_bytes(),
                f1c_report=(root / "readiness0_f1c_report_v1.json").read_bytes(),
                implementation_manifest=(
                    root / "readiness0_live_implementation_manifest_v1.json"
                ).read_bytes(),
                window_proof=(root / "readiness0_window_proof_v1.json").read_bytes(),
                repository_root=args.repository_root.resolve(),
                credential_present=False,
            ),
            evidence_manifest=(evidence_root / "evidence_manifest.json").read_bytes(),
            evidence_cases=(evidence_root / "cases.jsonl").read_bytes(),
        )
    )
    (root / "readiness0_current_status_v2.json").write_text(
        status.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
