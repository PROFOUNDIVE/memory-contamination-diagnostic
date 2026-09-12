from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Final

from memcontam.baselines.bot_read import BoTRetrievalDecision, distill_problem
from memcontam.baselines.bot_solve import render_bot_solve_prompt
from memcontam.baselines.bot_write import distill_thought_template
from memcontam.baselines.dynamic_cheatsheet_phase12 import core_synthesis_message
from memcontam.baselines.dynamic_cheatsheet_optional import _dc_rs_generation_message
from memcontam.baselines.full_history_adapter import _messages as history_messages
from memcontam.baselines.full_history import FullHistoryState
from memcontam.baselines.no_memory import NoMemoryAdapter, NoMemoryPolicy
from memcontam.baselines.reflexion_adapter import _generation_messages, _reflection_messages
from memcontam.baselines.retrieval_rag_adapter import _messages as rag_messages
from memcontam.clients.base import LLMResponse
from memcontam.experiment.phase12.filter_challenge.mft_state_models import JsonValue
from memcontam.memory.stores import MemoryState
from memcontam.tasks.base import TaskInstance
from memcontam.tasks.dispatch import canonical_task_json, render_common_task_spec

from .phase13_authority_files import read_regular_nofollow
from .phase13_cost_policy_models import Sha256
from .phase13_main_live_runtime_support import task_name, verifier
from .phase13_v3_authority_models import FrozenModel, ROUTED_DOCUMENTS, V3Identity
from .phase13_v3_cost_models import digest
from .phase13_v3_publication import ArtifactError
from .phase13_v3_resource_files import FileBinding, read_files

TASKS: Final = ("game24", "math_equation_balancer", "word_sorting", "mmlu_pro_engineering", "mmlu_pro_physics")
TEST_PATHS: Final = tuple(f"tests/test_phase13_v3_{name}.py" for name in ("artifact_builder", "mr_p4", "mr_p5", "mr_p6"))


class Predicate(FrozenModel):
    name: str
    passed: bool
    observation_sha256: Sha256


class ConformanceV3(FrozenModel):
    schema_version: str = "phase13_main_provider_free_conformance_v3"
    identity: V3Identity
    predicates: tuple[Predicate, ...]
    test_bindings: tuple[FileBinding, ...]
    scientific_result: bool = False
    measured_trajectories: int = 0
    real_provider_calls: int = 0
    conformance_hash: Sha256 = "0" * 64


class CaptureClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.messages: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]], model: str, config: dict[str, JsonValue]) -> LLMResponse:
        self.messages.append(messages)
        return LLMResponse(content=self.response, raw={"replay": True, "attempts": 1},
                           token_usage={"prompt_tokens": 1, "completion_tokens": 1}, latency_ms=0)


def synthetic_tasks() -> tuple[TaskInstance, ...]:
    return (
        TaskInstance(sample_id="conformance-game", task_name=TASKS[0], input={"numbers": [3, 3, 8, 8]}, verifier_spec={"target": 24}),
        TaskInstance(sample_id="conformance-meb", task_name=TASKS[1], input={"input": "2 ? 3 ? 4 = 14"}, verifier_spec={"target": "2 + 3 * 4 = 14", "target_value": 14}),
        TaskInstance(sample_id="conformance-words", task_name=TASKS[2], input={"words": ["alpha!", "Beta", "alpha"]}, verifier_spec={"sorted_words": ["Beta", "alpha", "alpha!"]}),
        *(TaskInstance(sample_id=f"conformance-{task}", task_name=task, input={"question": "Select the first displayed option.", "options": ["first", "second", "third"]}, verifier_spec={"answer_index": 0, "answer_label": "A"}) for task in TASKS[3:]),
    )


def task_receiving_messages(task: TaskInstance) -> tuple[tuple[str, str], ...]:
    block = render_common_task_spec(task)
    reader = CaptureClient('{"key_information":"public","restrictions":"registered","distilled_task":"synthetic"}')
    problem = distill_problem(task, reader, "fixture", {})
    decision = BoTRetrievalDecision("empty_buffer", None, None, 0.7)
    writer = CaptureClient('{"description":"public","template":"registered","category":"procedure-based","explicitly_used_memory_ids":[]}')
    distill_thought_template(canonical_task=block, distilled_problem=problem, retrieval_decision=decision,
                            selected_structure="procedure-based", solution_trace="synthetic", final_answer="synthetic",
                            visible_memory=(), client=writer, model="fixture", config={})
    messages: tuple[tuple[str, list[dict[str, str]]], ...] = (
        ("NoMem_generation", NoMemoryPolicy().build_prompt(task, MemoryState())),
        ("FH_generation", history_messages(task, FullHistoryState(records=[]))[0]),
        ("BoT_problem_distillation", reader.messages[0]),
        ("BoT_instantiation", [{"role": "user", "content": render_bot_solve_prompt(task, problem, decision)[0]}]),
        ("BoT_thought_distillation", writer.messages[0]),
        ("Reflexion_generation", _generation_messages(task, [])[0]),
        ("Reflexion_reflection", _reflection_messages(task, [], "final: synthetic", "synthetic", failed_actor_call_id="fixture", failed_actor_spans=[], target_set=None)[0]),
        ("DC_RS_synthesis", [core_synthesis_message(block, None, [])[0]]),
        ("DC_RS_generation", [_dc_rs_generation_message(block, "", None, [])[0]]),
    )
    if task.task_name in TASKS[:3]:
        messages = (*messages, ("RAG_generation", rag_messages(task, [], [])[0]))
    return tuple((stage, "\n".join(message["content"] for message in rows)) for stage, rows in messages)


def evaluate_conformance(repository: Path, authority_root: Path, identity: V3Identity) -> ConformanceV3:
    authority = read_regular_nofollow(authority_root / ROUTED_DOCUMENTS[5][1]).decode("utf-8")
    templates = tuple(block for block in re.findall(r"```text\n(.*?)\n```", authority, re.DOTALL) if block.startswith("Task family:"))
    if len(templates) != 4:
        raise ArtifactError("MAIN_PROVIDER_FREE_CONFORMANCE_FAILED")
    observations: list[Predicate] = []
    answers = ("8/(3-8/3)", "2+3*4=14", "Beta alpha alpha!", "A", "A")
    for index, (task, answer) in enumerate(zip(synthetic_tasks(), answers, strict=True)):
        template = templates[min(index, 3)]
        replacements = {"{numbers}": "[3,3,8,8]", "{operator_slot_equation}": "2 ? 3 ? 4 = 14",
                        "{words}": '["alpha!","Beta","alpha"]', "{question}": "Select the first displayed option.",
                        "{task_family}": "MMLU-Pro Engineering" if index == 3 else "MMLU-Pro Physics",
                        "{displayed_option_0}": "first", "{displayed_option_1}": "second", "{displayed_option_2}": "third"}
        expected = template.replace("\n...", "")
        for placeholder, value in replacements.items():
            expected = expected.replace(placeholder, value)
        actual = render_common_task_spec(task)
        hidden = task.model_copy(update={"verifier_spec": {"target": "HIDDEN", "answer_index": 2, "audit": "HIDDEN"}})
        checks = [("template_dynamic_binding", actual == expected, actual),
                  ("hidden_information", render_common_task_spec(hidden) == actual, actual),
                  ("retrieval_noninterference", canonical_task_json(task) == canonical_task_json(hidden) and canonical_task_json(task) != actual, canonical_task_json(task))]
        for stage, content in task_receiving_messages(task):
            checks.append((f"task_block/{stage}", content.count(actual) == 1, content))
        checked = verifier(task_name(task.task_name))(answer, task)
        accepted = checked if isinstance(checked, bool) else checked.is_correct
        checks.append(("semantic_payload", accepted, answer))
        fake = CaptureClient("final: " + answer)
        memory = MemoryState()
        outcome = NoMemoryAdapter().execute(task, memory, client=fake, model="fixture", config={}, verifier=verifier(task_name(task.task_name)))
        checks.append(("fake_backend", outcome.status == "succeeded" and len(fake.messages) == 1 and not memory.entries, outcome.status))
        for name, passed, observed in checks:
            observations.append(Predicate(name=f"{task.task_name}/{name}", passed=passed,
                                          observation_sha256=hashlib.sha256(observed.encode()).hexdigest()))
    if not all(row.passed for row in observations):
        raise ArtifactError("MAIN_PROVIDER_FREE_CONFORMANCE_FAILED")
    result = ConformanceV3(identity=identity, predicates=tuple(observations), test_bindings=tuple(row.binding for row in read_files(repository, TEST_PATHS)))
    return result.model_copy(update={"conformance_hash": digest(result, "conformance_hash")})
