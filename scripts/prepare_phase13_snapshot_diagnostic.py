from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import ValidationError

from memcontam.readiness.phase13_authority_files import AuthorityFileError
from memcontam.readiness.phase13_snapshot_packet import PacketError, prepare_packet, verify_packet


def main() -> int:
    parser = argparse.ArgumentParser(description="Provider-free snapshot preparation and read-only verification only.",
                                     allow_abbrev=False)
    parser.add_argument("--proposal", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--provider-free", action="store_true")
    parser.add_argument("--verify-packet", type=Path)
    args = parser.parse_args()
    if args.verify_packet is not None:
        if args.proposal is not None or args.output_dir is not None or args.provider_free:
            parser.error("--verify-packet cannot be combined with preparation options")
    elif args.proposal is None or args.output_dir is None or not args.provider_free:
        parser.error("preparation requires --proposal PATH --output-dir PATH --provider-free")
    try:
        result = (verify_packet(args.verify_packet) if args.verify_packet is not None
                  else prepare_packet(args.proposal, args.output_dir))
    except (PacketError, AuthorityFileError, ValidationError, OSError, ValueError) as error:
        parser.exit(1, f"PACKET_REFUSED: {error}\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
