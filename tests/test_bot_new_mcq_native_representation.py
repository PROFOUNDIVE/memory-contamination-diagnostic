from __future__ import annotations

from pathlib import Path

from memcontam.readiness.phase13_main_new_mcq_runtime import (
    load_new_mcq_runtime_registry,
    new_mcq_native_entries,
)
from memcontam.readiness.phase13_new_mcq_candidate_evidence_v2_rendering import (
    RenderInput,
    native_payload,
)

ROOT = Path(__file__).resolve().parents[1]


def test_new_mcq_bot_fields_survive_native_entry_materialization() -> None:
    registry = load_new_mcq_runtime_registry(ROOT)
    task = "mmlu_pro_engineering"

    entries = new_mcq_native_entries(task, "bot_style", registry)

    for arm, native in entries.items():
        document = registry.tasks[task].documents[arm]
        payload = native_payload(
            RenderInput(
                registry.tasks[task].selected_candidate_id,
                task,
                "BoT-style",
                "thought_template",
                "false" if arm == "contam" else arm,
                document.semantic_id,
                document.text,
                f"main-a::{task}",
            )
        )
        assert isinstance(payload, dict)
        assert native.retrieval_description == payload["retrieval_description"]
        assert native.template_body == payload["procedural_body"]
        assert native.content == native.template_body
        assert native.category == "procedure-based"
