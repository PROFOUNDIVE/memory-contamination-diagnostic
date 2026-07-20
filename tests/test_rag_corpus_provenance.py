from __future__ import annotations

import pytest

from memcontam.memory.corpus import (
    CorpusManifest,
    CorpusRecord,
    corpus_content_hash,
    trusted_corpus_card,
)


def _record() -> CorpusRecord:
    return CorpusRecord(
        entry_id="seed-1",
        task="game24",
        target_baselines=["retrieval_rag"],
        memory_type="strategy",
        content="Look for complementary subexpressions before combining results.",
        source="trusted-fixture",
        clean_or_contaminated="clean",
    )


def test_trusted_corpus_card_requires_manifest_and_matching_content_hash() -> None:
    records = [_record()]
    manifest = CorpusManifest(
        manifest_id="memory-catalog-v1",
        corpus_version="v1",
        content_hash=corpus_content_hash(records),
    )

    identity = trusted_corpus_card(
        records,
        manifest=manifest,
        task_family="game24",
        embedding_provider_identity="BAAI/bge-m3@5617a9f61b028005a4858fdac845db406aefb181",
    )

    assert identity.manifest_id == "memory-catalog-v1"
    with pytest.raises(ValueError, match="manifest content_hash"):
        trusted_corpus_card(
            records,
            manifest=CorpusManifest(
                manifest_id="memory-catalog-v1", corpus_version="v1", content_hash="sha256:wrong"
            ),
            task_family="game24",
            embedding_provider_identity="provider",
        )
    with pytest.raises(ValueError, match="manifest is required"):
        trusted_corpus_card(
            records,
            manifest=None,
            task_family="game24",
            embedding_provider_identity="provider",
        )
