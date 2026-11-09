"""Ephemeral QA owners only; none of these signatures is operational evidence."""
from __future__ import annotations

import json
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from .phase13_budget_fixtures import save


def signed_evidence(module, root: Path, subject="a" * 64, keys=None):
    witness = save(root / "synthetic-owner.json", {"purpose": "SYNTHETIC_QA_NOT_SCIENCE"})
    keys = keys or {role: Ed25519PrivateKey.generate() for role in ("science", "launch", "credential")}
    policy = module.TrustPolicy.model_validate_json(json.dumps({
        "purpose": "SYNTHETIC_QA", **{role: key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
                                      for role, key in keys.items()}}))
    statements = {}
    for role, key in keys.items():
        payload = {"role": role, "purpose": "SYNTHETIC_QA", "subject_sha256": subject,
                   "account_id": "qa-account", "not_before": "2026-01-01", "expires": "2099-12-31",
                   "evidence": [witness], "credential_fingerprint": module.credential_fingerprint("qa-not-an-operational-secret")
                   if role == "credential" else None}
        statement = module.SignedStatement.model_validate_json(json.dumps({**payload, "signature": "00" * 64}))
        statements[role] = {**payload, "signature": key.sign(module.signing_bytes(statement)).hex()}
    return module.AdmissionEvidence.model_validate_json(json.dumps(statements)), policy
