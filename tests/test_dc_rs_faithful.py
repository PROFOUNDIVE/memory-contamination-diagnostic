from __future__ import annotations

import pytest

from memcontam.baselines import dynamic_cheatsheet_optional
from memcontam.clients.base import LLMResponse
from memcontam.logging.provenance import compute_exposure_from_spans, normalize_memory_event
from memcontam.logging.schema import VerifierResult
from memcontam.memory.stores import MemoryEntry, MemoryState
from memcontam.tasks.base import TaskInstance
from memcontam.tasks.dispatch import canonical_task_json


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


def _pair(
    entry_id: str,
    input_text: str,
    generated_output: str,
    parsed_answer: str = "prior parsed answer",
) -> MemoryEntry:
    return MemoryEntry(
        entry_id=entry_id,
        content=input_text,
        memory_type="dc_rs_io_pair",
        clean_or_contaminated="clean",
        source_trial_id=f"prior:{entry_id}",
        metadata={
            "generated_output": generated_output,
            "parsed_answer": parsed_answer,
            "provenance": "PROVENANCE_SECRET",
        },
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
    assert appended["content"] == canonical_task_json(_task())
    assert appended["metadata"]["generated_output"] == "final: current output"
    assert appended["metadata"]["parsed_answer"] == "current output"
    assert appended["entry_id"].startswith("dc_rs_pair:run-1:game24:sample-1:")
    assert appended["entry_id"] not in [record.document_id for record in result["retrieved_records"]]
    assert "current output" not in _prompt_text(client)
    assert result["memory_write_event"]["synthesis_update"]["status"] == "replaced"
    assert result["memory_write_event"]["pair_appended"]["entry_id"] == appended["entry_id"]


def test_dc_rs_derived_cheatsheet_keeps_synthesis_lineage_in_answer_prompt(tmp_path) -> None:
    memory = MemoryState(
        entries=[
            MemoryEntry(
                entry_id="dc_cheatsheet:seed",
                content="old cheatsheet",
                memory_type="dynamic_cheatsheet",
                clean_or_contaminated="clean",
                metadata={"source_entry_ids": ["clean-origin"]},
            ),
            MemoryEntry(
                entry_id="contaminated-pair",
                content="prior contaminated input",
                memory_type="dc_rs_io_pair",
                clean_or_contaminated="contaminated",
                source_trial_id="prior:contaminated-pair",
                metadata={
                    "generated_output": "prior contaminated output",
                    "parsed_answer": "prior contaminated output",
                    "parent_entry_ids": ["pair-parent"],
                    "source_entry_ids": ["contaminated-origin"],
                },
            ),
        ]
    )
    client = _QueuedClient(["<cheatsheet>new cheatsheet</cheatsheet>", "final: current output"])

    result = _policy(_TieEmbeddingProvider(), tmp_path).run(
        _task(), memory, client=client, model="replay", config=_config(), verifier=_verifier
    )

    synthesis_call, answer_call = result["method_calls"]
    assert result["answer_call_id"] == answer_call.call_id
    assert result["answer_call_id"] != synthesis_call.call_id
    assert len(answer_call.source_spans) == 1
    span = answer_call.source_spans[0]
    assert answer_call.messages[0]["content"][span.start : span.end] == "new cheatsheet"
    assert span.parent_call_id == synthesis_call.call_id
    assert "contaminated-origin" in span.source_ids
    assert "pair-parent" in span.parent_ids
    assert span.clean_or_contaminated == "contaminated"
    exposure = compute_exposure_from_spans(
        result["answer_call_id"], answer_call.source_spans, "contaminated"
    )
    assert exposure.is_exposed is True
    assert "contaminated-origin" in exposure.exposed_source_ids


def test_dc_rs_synthesized_answer_span_records_exact_source_parents(tmp_path) -> None:
    injected = MemoryEntry(
        entry_id="injected-pair",
        content="prior injected input",
        memory_type="dc_rs_io_pair",
        clean_or_contaminated="contaminated",
        metadata={
            "generated_output": "prior injected output",
            "parsed_answer": "prior injected output",
            "contamination_class": "injected",
            "injected_root_ids": ["injected-pair"],
            "lineage_status": "exact",
            "lineage_basis": "seed",
            "direct_parent_ids": [],
            "target_set_id": "controlled_injected_derived_v1",
            "is_target_contamination": True,
        },
    )
    client = _QueuedClient(["<cheatsheet>new cheatsheet</cheatsheet>", "final: current output"])

    clean_cheatsheet = _cheatsheet()
    clean_cheatsheet.metadata = {
        "contamination_class": "clean",
        "injected_root_ids": [],
        "lineage_status": "exact",
        "lineage_basis": "seed",
        "direct_parent_ids": [],
        "target_set_id": "controlled_injected_derived_v1",
        "is_target_contamination": False,
    }
    result = _policy(_TieEmbeddingProvider(), tmp_path).run(
        _task(),
        MemoryState(entries=[clean_cheatsheet, injected]),
        client=client,
        model="replay",
        config={
            **_config(),
            "_logging_target_set_id": "controlled_injected_derived_v1",
        },
        verifier=_verifier,
    )

    synthesis_call, answer_call = result["method_calls"]
    span = answer_call.source_spans[0]
    assert span.parent_call_id == synthesis_call.call_id
    assert span.direct_parent_ids == ["dc_cheatsheet:seed", "injected-pair"]
    assert span.contamination_class == "derived"
    assert span.injected_root_ids == ["injected-pair"]
    assert span.lineage_status == "exact"
    assert span.lineage_basis == "recorded_parent"
    assert span.target_set_id == "controlled_injected_derived_v1"
    assert span.is_target_contamination is True


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
    assert provider.queries == [canonical_task_json(_task())]
    synthesis_prompt = client.calls[0][0][0]["content"]
    for pair in pairs:
        assert pair.content in synthesis_prompt
        assert pair.metadata["generated_output"] in synthesis_prompt
        assert pair.metadata["generated_output"] not in provider.documents


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
    assert other_identity_pair.metadata["generated_output"] not in prompts
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


def test_dc_rs_replaced_cheatsheet_normalizes_replace_and_append(tmp_path) -> None:
    memory = MemoryState(entries=[_cheatsheet()])
    client = _QueuedClient(["<cheatsheet>new cheatsheet</cheatsheet>", "final: current output"])

    result = _policy(_TieEmbeddingProvider(), tmp_path).run(
        _task(), memory, client=client, model="replay", config=_config(), verifier=_verifier
    )

    source_trial_id = "run-1:game24:sample-1:dynamic_cheatsheet_rs_optional:contaminated:replay"
    before = [MemoryEntry.model_validate(entry) for entry in result["memory_before"]]
    after = [MemoryEntry.model_validate(entry) for entry in result["memory_after"]]
    event = normalize_memory_event(
        "dynamic_cheatsheet_rs_optional",
        source_trial_id,
        before,
        after,
        result["memory_write_event"],
    )

    assert event is not None
    assert event.status == "accepted"
    assert event.operation == "replace_and_append"
    assert event.baseline == "dynamic_cheatsheet_rs_optional"
    assert event.source_trial_id == source_trial_id
    assert event.before_entry_ids == ["dc_cheatsheet:seed"]
    assert len(event.after_entry_ids) == 2
    assert "dc_cheatsheet:seed" not in event.after_entry_ids
    assert event.new_entry_ids == [e.entry_id for e in after if e.entry_id != "dc_cheatsheet:seed"]
    assert event.removed_entry_ids == ["dc_cheatsheet:seed"]
    assert event.before_snapshot_hash != event.after_snapshot_hash
    assert event.creation_origin == "dynamic_cheatsheet"


@pytest.mark.parametrize(
    ("synthesis", "parser_status"),
    [
        ("no cheatsheet tag", "preserved_missing_tag"),
        ("<cheatsheet> </cheatsheet>", "preserved_empty"),
    ],
)
def test_dc_rs_malformed_synthesis_still_appends_pair(
    synthesis: str, parser_status: str, tmp_path
) -> None:
    memory = MemoryState(entries=[_cheatsheet("keep this exact cheatsheet")])
    client = _QueuedClient([synthesis, "final: current output"])

    result = _policy(_TieEmbeddingProvider(), tmp_path).run(
        _task(), memory, client=client, model="replay", config=_config(), verifier=_verifier
    )

    source_trial_id = "run-1:game24:sample-1:dynamic_cheatsheet_rs_optional:contaminated:replay"
    before = [MemoryEntry.model_validate(entry) for entry in result["memory_before"]]
    after = [MemoryEntry.model_validate(entry) for entry in result["memory_after"]]
    event = normalize_memory_event(
        "dynamic_cheatsheet_rs_optional",
        source_trial_id,
        before,
        after,
        result["memory_write_event"],
    )

    assert event is not None
    # The synthesis sub-operation is preserved, but an I/O pair is appended, so
    # the overall decision is an accepted mutation.
    assert event.status == "accepted"
    assert event.operation == "replace_and_append"
    assert event.before_entry_ids == ["dc_cheatsheet:seed"]
    assert event.after_entry_ids != ["dc_cheatsheet:seed"]
    assert event.new_entry_ids == [e.entry_id for e in after if e.memory_type == "dc_rs_io_pair"]
    assert event.removed_entry_ids == []
    assert event.before_snapshot_hash != event.after_snapshot_hash


def test_dc_rs_synthesizes_raw_solution_history_and_generates_from_cheatsheet_only(tmp_path) -> None:
    raw_prior_output = "STRATEGY_MARKER: make a factor table\nCODE_MARKER: for candidate in options"
    memory = MemoryState(entries=[_cheatsheet(), _pair("prior", "PRIOR_INPUT", raw_prior_output)])
    client = _QueuedClient(
        [
            "<cheatsheet>synthesized transferable strategy</cheatsheet>",
            "Reasoning for the current task\nfinal: current final answer",
        ]
    )

    result = _policy(_TieEmbeddingProvider(), tmp_path).run(
        _task(), memory, client=client, model="replay", config=_config(), verifier=_verifier
    )

    synthesis_prompt = client.calls[0][0][0]["content"]
    generation_prompt = client.calls[1][0][0]["content"]
    appended = next(
        entry
        for entry in result["memory_after"]
        if entry["entry_id"].startswith("dc_rs_pair:")
    )
    assert raw_prior_output in synthesis_prompt
    assert "prior parsed answer" not in synthesis_prompt
    assert canonical_task_json(_task()) in synthesis_prompt
    assert "current final answer" not in synthesis_prompt
    assert "synthesized transferable strategy" in generation_prompt
    assert raw_prior_output not in generation_prompt
    assert appended["content"] == canonical_task_json(_task())
    assert appended["metadata"]["generated_output"] == "Reasoning for the current task\nfinal: current final answer"
    assert appended["metadata"]["parsed_answer"] == "current final answer"


def test_dc_rs_synthesis_lineage_excludes_nonretrieved_contaminated_pairs(tmp_path) -> None:
    outside_pair = _pair("z-outside", "OUTSIDE_INPUT", "OUTSIDE_OUTPUT")
    outside_pair.clean_or_contaminated = "contaminated"
    outside_pair.metadata.update(
        {"parent_entry_ids": ["outside-parent"], "source_entry_ids": ["outside-origin"]}
    )
    memory = MemoryState(
        entries=[
            _cheatsheet(),
            _pair("alpha", "ALPHA_INPUT", "ALPHA_OUTPUT"),
            _pair("bravo", "BRAVO_INPUT", "BRAVO_OUTPUT"),
            _pair("charlie", "CHARLIE_INPUT", "CHARLIE_OUTPUT"),
            outside_pair,
        ]
    )
    client = _QueuedClient(["<cheatsheet>new cheatsheet</cheatsheet>", "final: current output"])

    result = _policy(_TieEmbeddingProvider(), tmp_path).run(
        _task(), memory, client=client, model="replay", config=_config(), verifier=_verifier
    )

    synthesized = next(
        entry for entry in result["memory_after"] if entry["memory_type"] == "dynamic_cheatsheet"
    )
    appended = next(
        entry
        for entry in result["memory_after"]
        if entry["entry_id"].startswith("dc_rs_pair:")
    )
    assert [record.document_id for record in result["retrieved_records"]] == [
        "alpha",
        "bravo",
        "charlie",
    ]
    assert "OUTSIDE_OUTPUT" not in client.calls[0][0][0]["content"]
    assert "outside-origin" not in synthesized["metadata"]["source_entry_ids"]
    assert "outside-parent" not in synthesized["metadata"]["parent_entry_ids"]
    assert "outside-origin" not in appended["metadata"]["source_entry_ids"]
    assert "outside-parent" not in appended["metadata"]["parent_entry_ids"]


def test_dc_rs_invalid_final_answer_retains_raw_pair_before_parser_failure(tmp_path) -> None:
    def verifier_must_not_run(_answer: str, _task: TaskInstance) -> VerifierResult:
        pytest.fail("verifier must not run after invalid final-answer parsing")

    raw_response = "Reasoning without a final marker"
    result = _policy(_TieEmbeddingProvider(), tmp_path).run(
        _task(),
        MemoryState(entries=[_cheatsheet()]),
        client=_QueuedClient(["<cheatsheet>new cheatsheet</cheatsheet>", raw_response]),
        model="replay",
        config=_config(),
        verifier=verifier_must_not_run,
    )

    appended = next(entry for entry in result["memory_after"] if entry["memory_type"] == "dc_rs_io_pair")
    assert result["status"] == "failed"
    assert result["failure_disposition"] == "dc_rs_invalid_final_answer"
    assert result["scientific_ineligibility_reason"] == "invalid_final_answer"
    assert result["parsed_answer"] is None
    assert appended["metadata"]["generated_output"] == raw_response
    assert appended["metadata"]["parsed_answer"] is None
    assert result["memory_write_event"]["synthesis_update"]["status"] == "replaced"


def test_dc_rs_verifier_failure_retains_synthesized_state_and_appended_pair(tmp_path) -> None:
    def raising_verifier(_answer: str, _task: TaskInstance) -> VerifierResult:
        raise RuntimeError("verifier unavailable")

    result = _policy(_TieEmbeddingProvider(), tmp_path).run(
        _task(),
        MemoryState(entries=[_cheatsheet()]),
        client=_QueuedClient(["<cheatsheet>new cheatsheet</cheatsheet>", "final: current output"]),
        model="replay",
        config=_config(),
        verifier=raising_verifier,
    )

    appended = next(entry for entry in result["memory_after"] if entry["memory_type"] == "dc_rs_io_pair")
    assert result["status"] == "failed"
    assert result["failure_disposition"] == "verifier_contract_failed"
    assert result["memory_write_event"]["synthesis_update"]["status"] == "replaced"
    assert appended["metadata"]["generated_output"] == "final: current output"
    assert appended["metadata"]["parsed_answer"] == "current output"


def test_dc_rs_requires_an_explicit_shared_embedding_provider(tmp_path) -> None:
    policy_class = dynamic_cheatsheet_optional.DynamicCheatsheetRetrievalSynthesisPolicy

    with pytest.raises(ValueError, match="explicit shared embedding provider"):
        policy_class(cache_dir=tmp_path).run(
            _task(),
            MemoryState(entries=[_cheatsheet()]),
            client=_QueuedClient([]),
            model="replay",
            config=_config(),
        )
