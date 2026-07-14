from __future__ import annotations

import pytest

from memcontam.baselines import dynamic_cheatsheet_optional
from memcontam.clients.base import LLMResponse
from memcontam.logging.schema import VerifierResult
from memcontam.memory.stores import MemoryEntry, MemoryState
from memcontam.tasks.base import TaskInstance


class _QueuedClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[tuple[list[dict[str, str]], str, dict]] = []

    def chat(self, messages: list[dict[str, str]], model: str, config: dict) -> LLMResponse:
        self.calls.append((messages, model, config))
        return LLMResponse(
            content=self.responses.pop(0),
            raw={"replay": True},
            token_usage={},
            latency_ms=0,
        )


class _TieEmbeddingProvider:
    def __init__(self) -> None:
        self.documents: list[str] = []
        self.queries: list[str] = []

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "model_id": "test-tie-provider",
            "revision": "test",
            "embedding_library_version": "test",
            "vector_dimension": 2,
        }

    def encode_document(self, text: str) -> list[float]:
        self.documents.append(text)
        return [1.0, 0.0]

    def encode_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [1.0, 0.0]


def _task() -> TaskInstance:
    return TaskInstance(
        sample_id="sample-1",
        task_name="game24",
        input={"numbers": [1, 2, 3, 4], "target": 24},
        verifier_spec={"gold": "GOLD_SECRET", "contamination": "CONTAMINATION_SECRET"},
    )


def _config() -> dict[str, str]:
    return {
        "run_id": "run-1",
        "baseline": "dynamic_cheatsheet_rs_optional",
        "arm": "contaminated",
        "model": "replay",
    }


def _pair(entry_id: str, input_text: str, output_text: str) -> MemoryEntry:
    return MemoryEntry(
        entry_id=entry_id,
        content=input_text,
        memory_type="dc_rs_io_pair",
        clean_or_contaminated="clean",
        source_trial_id=f"prior:{entry_id}",
        metadata={"output_text": output_text, "provenance": "PROVENANCE_SECRET"},
    )


def _cheatsheet(content: str = "old cheatsheet") -> MemoryEntry:
    return MemoryEntry(
        entry_id="dc_cheatsheet:seed",
        content=content,
        memory_type="dynamic_cheatsheet",
        clean_or_contaminated="clean",
    )


def _verifier(answer: str, task: TaskInstance) -> VerifierResult:
    assert task == _task()
    return VerifierResult(
        is_correct=answer == "current output",
        parsed_answer=answer,
        reason="VERIFIER_SECRET",
        metadata={"expected_answer": "EXPECTED_SECRET"},
    )


def _policy(provider: _TieEmbeddingProvider, tmp_path):  # noqa: ANN001
    policy_class = getattr(
        dynamic_cheatsheet_optional, "DynamicCheatsheetRetrievalSynthesisPolicy", None
    )
    assert policy_class is not None
    return policy_class(embedding_provider=provider, cache_dir=tmp_path)


def _prompt_text(client: _QueuedClient) -> str:
    return "\n".join(message["content"] for messages, _, _ in client.calls for message in messages)


def test_dc_rs_zero_candidates_synthesizes_before_generation_and_appends_afterward(tmp_path) -> None:
    memory = MemoryState(entries=[_cheatsheet()])
    client = _QueuedClient(["<cheatsheet>new cheatsheet</cheatsheet>", "final: current output"])

    result = _policy(_TieEmbeddingProvider(), tmp_path).run(
        _task(), memory, client=client, model="replay", config=_config(), verifier=_verifier
    )

    assert [call.stage for call in result["method_calls"]] == [
        "dc_rs_synthesize",
        "dc_rs_generate",
    ]
    assert [record.document_id for record in result["retrieved_records"]] == []
    assert "new cheatsheet" in client.calls[1][0][0]["content"]
    assert result["memory_before"] == [entry.model_dump() for entry in memory.entries]
    assert memory.entries == [_cheatsheet()]

    appended = next(entry for entry in result["memory_after"] if entry["memory_type"] == "dc_rs_io_pair")
    assert appended["content"] == str(_task().input)
    assert appended["metadata"]["output_text"] == "current output"
    assert appended["entry_id"].startswith("dc_rs_pair:run-1:game24:sample-1:")
    assert appended["entry_id"] not in [record.document_id for record in result["retrieved_records"]]
    assert "current output" not in _prompt_text(client)
    assert result["memory_write_event"]["synthesis_update"]["status"] == "replaced"
    assert result["memory_write_event"]["pair_appended"]["entry_id"] == appended["entry_id"]


@pytest.mark.parametrize(
    ("pairs", "expected_ids"),
    [
        ([_pair("one", "INPUT_ONE", "OUTPUT_ONE")], ["one"]),
        (
            [
                _pair("charlie", "INPUT_CHARLIE", "OUTPUT_CHARLIE"),
                _pair("alpha", "INPUT_ALPHA", "OUTPUT_ALPHA"),
                _pair("bravo", "INPUT_BRAVO", "OUTPUT_BRAVO"),
            ],
            ["alpha", "bravo", "charlie"],
        ),
    ],
)
def test_dc_rs_retrieves_all_available_pairs_up_to_three(
    pairs: list[MemoryEntry], expected_ids: list[str], tmp_path
) -> None:
    provider = _TieEmbeddingProvider()
    client = _QueuedClient(["<cheatsheet>synthesized</cheatsheet>", "final: current output"])

    result = _policy(provider, tmp_path).run(
        _task(),
        MemoryState(entries=[_cheatsheet(), *pairs]),
        client=client,
        model="replay",
        config=_config(),
        verifier=_verifier,
    )

    assert [record.document_id for record in result["retrieved_records"]] == expected_ids
    assert provider.documents == [pair.content for pair in pairs]
    assert provider.queries == [str(_task().input)]
    synthesis_prompt = client.calls[0][0][0]["content"]
    for pair in pairs:
        assert pair.content in synthesis_prompt
        assert pair.metadata["output_text"] in synthesis_prompt
        assert pair.metadata["output_text"] not in provider.documents


def test_dc_rs_top_three_breaks_ties_by_entry_id_and_recovers_outputs(tmp_path) -> None:
    provider = _TieEmbeddingProvider()
    pairs = [
        _pair("delta", "INPUT_DELTA", "OUTPUT_DELTA"),
        _pair("bravo", "INPUT_BRAVO", "OUTPUT_BRAVO"),
        _pair("alpha", "INPUT_ALPHA", "OUTPUT_ALPHA"),
        _pair("charlie", "INPUT_CHARLIE", "OUTPUT_CHARLIE"),
    ]
    client = _QueuedClient(["<cheatsheet>synthesized</cheatsheet>", "final: current output"])

    result = _policy(provider, tmp_path).run(
        _task(),
        MemoryState(entries=[_cheatsheet(), *pairs]),
        client=client,
        model="replay",
        config=_config(),
        verifier=_verifier,
    )

    assert [record.document_id for record in result["retrieved_records"]] == ["alpha", "bravo", "charlie"]
    synthesis_prompt = client.calls[0][0][0]["content"]
    assert "OUTPUT_ALPHA" in synthesis_prompt
    assert "OUTPUT_BRAVO" in synthesis_prompt
    assert "OUTPUT_CHARLIE" in synthesis_prompt
    assert "OUTPUT_DELTA" not in synthesis_prompt
    assert "alpha" not in synthesis_prompt
    assert "rank=" not in synthesis_prompt


@pytest.mark.parametrize(
    ("synthesis", "parser_status"),
    [
        ("no cheatsheet tag", "preserved_missing_tag"),
        ("<cheatsheet> </cheatsheet>", "preserved_empty"),
    ],
)
def test_dc_rs_malformed_synthesis_preserves_cheatsheet_and_records_status(
    synthesis: str, parser_status: str, tmp_path
) -> None:
    memory = MemoryState(entries=[_cheatsheet("keep this exact cheatsheet")])
    client = _QueuedClient([synthesis, "final: current output"])

    result = _policy(_TieEmbeddingProvider(), tmp_path).run(
        _task(), memory, client=client, model="replay", config=_config(), verifier=_verifier
    )

    assert [call.stage for call in result["method_calls"]] == [
        "dc_rs_synthesize",
        "dc_rs_generate",
    ]
    current_cheatsheet = next(
        entry for entry in result["memory_after"] if entry["memory_type"] == "dynamic_cheatsheet"
    )
    assert current_cheatsheet == _cheatsheet("keep this exact cheatsheet").model_dump()
    assert result["memory_write_event"]["synthesis_update"] == {
        "status": "preserved",
        "parser_status": parser_status,
    }
    assert any(entry["memory_type"] == "dc_rs_io_pair" for entry in result["memory_after"])


def test_dc_rs_prompts_exclude_labels_provenance_and_other_identity_pairs(tmp_path) -> None:
    own_pair = _pair("own", "OWN_INPUT", "OWN_OUTPUT")
    other_identity_pair = _pair("other", "OTHER_IDENTITY_INPUT", "OTHER_IDENTITY_OUTPUT")
    client = _QueuedClient(["<cheatsheet>synthesized</cheatsheet>", "final: current output"])

    result = _policy(_TieEmbeddingProvider(), tmp_path).run(
        _task(),
        MemoryState(entries=[_cheatsheet(), own_pair]),
        client=client,
        model="replay",
        config={**_config(), "arm": "contaminated", "provenance": "PROVENANCE_SECRET"},
        verifier=_verifier,
    )

    assert [record.document_id for record in result["retrieved_records"]] == ["own"]
    prompts = _prompt_text(client)
    assert "OWN_INPUT" in prompts
    assert "OWN_OUTPUT" in prompts
    assert other_identity_pair.content not in prompts
    assert other_identity_pair.metadata["output_text"] not in prompts
    for forbidden in [
        "GOLD_SECRET",
        "CONTAMINATION_SECRET",
        "VERIFIER_SECRET",
        "EXPECTED_SECRET",
        "PROVENANCE_SECRET",
        "OTHER_IDENTITY_INPUT",
        "OTHER_IDENTITY_OUTPUT",
        "contaminated",
    ]:
        assert forbidden not in prompts
