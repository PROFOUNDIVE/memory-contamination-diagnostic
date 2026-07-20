from __future__ import annotations

import hashlib
import json

from memcontam.clients.config import ProviderConfig
from memcontam.clients.provider_profile import normalize_provider_profile, provider_profile_id


def test_provider_profile_is_hashable_and_excludes_credentials() -> None:
    profile = normalize_provider_profile(
        ProviderConfig(
            provider="openai_compatible",
            base_url="HTTPS://user:secret@example.invalid:8443/v1?credential=secret#fragment",
            api_key_env="OPENAI_COMPATIBLE_API_KEY",
            timeout_seconds=30,
            max_retries=2,
        ),
        served_models=["model-b", "model-a"],
        model_snapshots={"model-b": "snapshot-b", "model-a": "snapshot-a"},
    )

    assert profile.normalized_base_url == "https://example.invalid:8443/v1"
    assert profile.served_models == ("model-a", "model-b")
    payload = json.dumps(profile.to_dict(), sort_keys=True, separators=(",", ":"))
    assert provider_profile_id(profile) == hashlib.sha256(payload.encode("utf-8")).hexdigest()
    assert "secret" not in payload
    assert "credential" not in payload
