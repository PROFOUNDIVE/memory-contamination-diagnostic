from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Literal

import pytest

from memcontam.baselines.dynamic_cheatsheet_phase12 import DcRsStateV3
from memcontam.contamination.phase12.models import (
    CandidateCertificationError,
    canonical_content_hash,
)
from memcontam.contamination.phase12.registry import load_current_candidate_registry
from memcontam.contamination.phase12.renderers import RendererError, RendererRegistry
from memcontam.experiment.phase12.runtime_registry import PHASE13_CORE_BASELINE_REGISTRY
from memcontam.memory.checkpoint_v3 import NativeState, serialize_checkpoint
from memcontam.readiness.phase13_main_live_runtime import ProductionMainRuntime

from .test_phase13_native_rendering import _Client, _Embedder, _clean_state
from .test_phase13_v3_entrypoint_integration import deny_external as deny_external


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = Path("data/phase12/registries/candidate_registry_v2.json")
DC_REGISTRY = Path("data/phase13/main/legacy_dc_rs_intervention_registry_v2.json")
pytestmark = pytest.mark.usefixtures("deny_external")


@pytest.mark.parametrize("task_index", (0, 2))
@pytest.mark.parametrize(
    ("field", "replacement", "code"),
    (
        ("applicability_id", "obsolete-v1", "CURRENT_APPLICABILITY_BINDING_MISMATCH"),
        ("applicability_sha256", "0" * 64, "CURRENT_APPLICABILITY_BINDING_MISMATCH"),
        ("triplet_id", "foreign-triplet", "CURRENT_CANDIDATE_BINDING_MISMATCH"),
    ),
)
def test_active_runtime_rejects_v2_binding_tamper(
    task_index: int, field: str, replacement: str, code: str, tmp_path: Path,
) -> None:
    for relative in (
        Path("data/phase13/main/mr_p4/main_a_common_checkpoint_registry_v1.json"),
        Path("data/phase13/observability/registration_packet_v2.json"),
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
    payload = json.loads((ROOT / CANDIDATES).read_bytes())
    payload["triplets"][task_index][field] = replacement
    destination = tmp_path / CANDIDATES
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CandidateCertificationError, match=code):
        ProductionMainRuntime(tmp_path, tmp_path / "cache", client=_Client())


@pytest.mark.parametrize("task_index", (0, 2))
@pytest.mark.parametrize("role", ("false_candidate", "correct_twin", "irrelevant_control"))
def test_current_loader_rejects_rehashed_candidate_content(
    task_index: int, role: str, tmp_path: Path,
) -> None:
    payload = json.loads((ROOT / CANDIDATES).read_bytes())
    variant = payload["triplets"][task_index][role]
    variant["content"] = "foreign construction procedure"
    variant["content_hash"] = canonical_content_hash(variant["content"])
    destination = tmp_path / "rehashed.json"
    destination.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CandidateCertificationError, match="CURRENT_CANDIDATE_BINDING_MISMATCH"):
        load_current_candidate_registry(destination)


@pytest.mark.parametrize("task_index", (0, 2))
@pytest.mark.parametrize("baseline", ("fh_bounded", "rag_frozen", "bot_style", "reflexion_style", "dc_rs"))
@pytest.mark.parametrize("role", ("false_candidate", "correct_twin", "irrelevant_control"))
def test_governed_native_renderer_rejects_self_consistent_foreign_triplet(
    task_index: int, baseline: str,
    role: Literal["false_candidate", "correct_twin", "irrelevant_control"],
) -> None:
    registry = load_current_candidate_registry(ROOT / CANDIDATES)
    renderers = RendererRegistry.governed(
        (ROOT / DC_REGISTRY).read_bytes(), registry,
        hashlib.sha256((ROOT / CANDIDATES).read_bytes()).hexdigest(),
    )
    triplet = registry.triplets[task_index]
    variant = getattr(triplet, role)
    forged = replace(variant, content="foreign procedure", content_hash=canonical_content_hash("foreign procedure"))
    foreign = replace(triplet, **{role: forged})
    state = (
        DcRsStateV3(archive=[], allow_unparented_strategies=True)
        if baseline == "dc_rs" else _clean_state(baseline, _Embedder())
    )
    snapshot = PHASE13_CORE_BASELINE_REGISTRY[baseline].serialize_state(state)
    assert isinstance(snapshot, NativeState)
    checkpoint = serialize_checkpoint(snapshot)
    renderer = {
        "false_candidate": renderers.render_false,
        "correct_twin": renderers.render_correct,
        "irrelevant_control": renderers.render_irrelevant,
    }[role]

    with pytest.raises(RendererError, match="GOVERNED_CANDIDATE_BINDING_MISMATCH"):
        renderer(baseline, foreign, checkpoint)


@pytest.mark.parametrize("task_index", (0, 2))
def test_current_native_rendering_requires_governed_registry(task_index: int) -> None:
    registry = load_current_candidate_registry(ROOT / CANDIDATES)
    snapshot = PHASE13_CORE_BASELINE_REGISTRY["fh_bounded"].serialize_state(
        _clean_state("fh_bounded", _Embedder())
    )
    assert isinstance(snapshot, NativeState)
    checkpoint = serialize_checkpoint(snapshot)

    with pytest.raises(RendererError, match="CURRENT_RENDERER_REGISTRY_REQUIRED"):
        RendererRegistry.native().render_false("fh_bounded", registry.triplets[task_index], checkpoint)


def test_governed_renderer_rejects_candidate_file_hash_swap() -> None:
    registry = load_current_candidate_registry(ROOT / CANDIDATES)

    with pytest.raises(ValueError, match="LEGACY_DC_RS_REGISTRY_INVALID"):
        RendererRegistry.governed((ROOT / DC_REGISTRY).read_bytes(), registry, "0" * 64)
