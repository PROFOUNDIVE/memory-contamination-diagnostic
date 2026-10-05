from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
from typing import Literal, assert_never

import pytest

from memcontam.contamination.phase12.models import (
    CandidateCertificationError,
    CandidateTriplet,
    canonical_content_hash,
)
from memcontam.contamination.phase12.registry import (
    load_current_candidate_registry,
    validate_current_candidate_registry,
)
from memcontam.contamination.phase12.renderers import RendererError, RendererRegistry
from memcontam.evaluation.phase13_observability_registration import registered_failure_class
from memcontam.memory.checkpoint_v3 import NativeState, serialize_checkpoint
from memcontam.readiness.phase13_main_live_runtime import ProductionMainRuntime
from memcontam.tasks.base import TaskInstance

from .test_phase13_native_consumption_matrix import NativeTransport
from .test_phase13_v3_entrypoint_integration import deny_external as deny_external


ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.usefixtures("deny_external")
ROLES = ("false_candidate", "correct_twin", "irrelevant_control")
BASELINES = ("fh_bounded", "rag_frozen", "bot_style", "reflexion_style", "dc_rs")
Mutation = Literal["content_hash", "render_id", "rule_id", "rehashed_content"]


def _tamper(triplet: CandidateTriplet, role: str, mutation: Mutation) -> CandidateTriplet:
    candidate = getattr(triplet, role)
    match mutation:
        case "content_hash":
            candidate = replace(candidate, content_hash="0" * 64)
        case "render_id":
            candidate = replace(candidate, render_id=candidate.render_id + "-stale")
        case "rule_id":
            candidate = replace(candidate, rule_id=candidate.rule_id + "-stale")
        case "rehashed_content":
            content = candidate.content + " forged"
            candidate = replace(candidate, content=content, content_hash=canonical_content_hash(content))
        case unreachable:
            assert_never(unreachable)
    return replace(triplet, **{role: candidate})


@pytest.mark.parametrize("task_index", range(3))
@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("mutation", ("content_hash", "render_id", "rule_id", "rehashed_content"))
def test_current_registry_rejects_mutated_matched_candidate(
    task_index: int, role: str, mutation: Mutation,
) -> None:
    registry = load_current_candidate_registry(ROOT / "data/phase12/registries/candidate_registry_v2.json")
    triplets = list(registry.triplets)
    triplets[task_index] = _tamper(triplets[task_index], role, mutation)

    with pytest.raises(CandidateCertificationError):
        validate_current_candidate_registry(replace(registry, triplets=tuple(triplets)))


@pytest.mark.parametrize("task_index", range(3))
@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("mutation", ("content_hash", "render_id", "rule_id", "rehashed_content"))
def test_governed_renderer_construction_rejects_mutated_candidate_registry(
    task_index: int, role: str, mutation: Mutation,
) -> None:
    path = ROOT / "data/phase12/registries/candidate_registry_v2.json"
    registry = load_current_candidate_registry(path)
    triplets = list(registry.triplets)
    triplets[task_index] = _tamper(triplets[task_index], role, mutation)

    with pytest.raises(ValueError):
        RendererRegistry.governed(
            (ROOT / "data/phase13/main/legacy_dc_rs_intervention_registry_v2.json").read_bytes(),
            replace(registry, triplets=tuple(triplets)),
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )


@pytest.mark.parametrize("baseline", BASELINES)
@pytest.mark.parametrize("task_index", range(3))
@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("mutation", ("content_hash", "render_id", "rule_id", "rehashed_content"))
def test_production_governed_renderer_rejects_mutated_matched_carrier(
    baseline: str, task_index: int, role: str, mutation: Mutation, tmp_path: Path,
) -> None:
    runtime = ProductionMainRuntime(ROOT, tmp_path / "cache", client=NativeTransport())
    triplet = _tamper(runtime._candidate_registry.triplets[task_index], role, mutation)
    prefix = serialize_checkpoint(NativeState(baseline, (), {}))
    render = {
        "false_candidate": runtime._renderers.render_false,
        "correct_twin": runtime._renderers.render_correct,
        "irrelevant_control": runtime._renderers.render_irrelevant,
    }[role]

    with pytest.raises(RendererError, match="GOVERNED_CANDIDATE_BINDING_MISMATCH"):
        render(baseline, triplet, prefix)


@pytest.mark.parametrize("field", ("registry_id", "schema_version"))
def test_current_registry_rejects_stale_version(field: str) -> None:
    registry = load_current_candidate_registry(ROOT / "data/phase12/registries/candidate_registry_v2.json")

    with pytest.raises(CandidateCertificationError):
        validate_current_candidate_registry(replace(registry, **{field: "historical-v1"}))


@pytest.mark.parametrize("task_index", (0, 2))
@pytest.mark.parametrize("field", ("applicability_id", "applicability_sha256"))
def test_production_renderer_rejects_stale_applicability(
    task_index: int, field: str, tmp_path: Path,
) -> None:
    runtime = ProductionMainRuntime(ROOT, tmp_path / "cache", client=NativeTransport())
    triplet = replace(runtime._candidate_registry.triplets[task_index], **{field: "stale-v1"})
    prefix = serialize_checkpoint(NativeState("fh_bounded", (), {}))

    with pytest.raises(RendererError, match="GOVERNED_CANDIDATE_BINDING_MISMATCH"):
        runtime._renderers.render_false("fh_bounded", triplet, prefix)


@pytest.mark.parametrize(("numbers", "answer", "applicable"), (
    ([1, 1, 1, 8], "8+1+1+1", True),
    ([1, 2, 3, 4], "1+2+3+4", False),
    ([1, 1, 1, 1], "1+1+1+1", False),
))
def test_production_game24_failure_class_uses_full_minus_flat_law(
    numbers: list[int], answer: str, applicable: bool,
) -> None:
    task = TaskInstance(sample_id="construction-g24", task_name="game24",
                        input={"numbers": numbers}, verifier_spec={"target": 24})

    failure = registered_failure_class(task, answer, 0)

    assert (failure is not None) == applicable


@pytest.mark.parametrize(("words", "applicable"), (
    (("ayz", "aza"), True),
    (("az", "ba"), False),
    (("ab", "ac"), False),
    (("aya", "aza"), False),
    (("ab", "abc"), False),
))
def test_production_word_failure_class_requires_nonempty_prefix_reversal(
    words: tuple[str, str], applicable: bool,
) -> None:
    task = TaskInstance(sample_id="construction-ws", task_name="word_sorting",
                        input={"words": list(words)}, verifier_spec={"sorted_words": sorted(words)})

    failure = registered_failure_class(task, " ".join(reversed(sorted(words))), 0)

    assert (failure is not None) == applicable
