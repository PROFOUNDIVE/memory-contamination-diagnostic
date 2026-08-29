from __future__ import annotations

import argparse
import os
from pathlib import Path

from memcontam.readiness.phase13_readiness0_current_status import (
    PreliveArtifactBytes,
    build_current_readiness0_status,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.repository_root / "data/phase13/main/mr_p4"
    status = build_current_readiness0_status(
        PreliveArtifactBytes(
            live_request=(root / "readiness0_live_request_v1.json").read_bytes(),
            live_authorization=(root / "readiness0_live_authorization_v1.json").read_bytes(),
            f1c_registry=(root / "readiness0_f1c_registry_v1.json").read_bytes(),
            f1c_report=(root / "readiness0_f1c_report_v1.json").read_bytes(),
            implementation_manifest=(
                root / "readiness0_live_implementation_manifest_v1.json"
            ).read_bytes(),
            window_proof=(root / "readiness0_window_proof_v1.json").read_bytes(),
            repository_root=args.repository_root.resolve(),
            credential_present=bool(os.environ.get("OPENAI_API_KEY")),
        )
    )
    (root / "readiness0_current_status_v1.json").write_text(
        status.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
