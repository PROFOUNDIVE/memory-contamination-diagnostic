from __future__ import annotations

from .phase13_new_mcq_rag_models import (
    TASKS,
    AuthoritySelection,
)


def authority_selection() -> AuthoritySelection:
    return AuthoritySelection(
        schema_version="phase13_new_mcq_rag_authority_selection_v1",
        protocol_authority_sha256=(
            "022879f559b145e30e645b6ccbd139e9927899d370f1956d27a0562580acf85f"
        ),
        selection_law="accepted_continuation_state_2026_08_21",
        task_selections={task: "MCQ-H2-DETAIL-LENGTH-v1" for task in TASKS},
        gpqa_extension_status="H2_BLINDED_PLAUSIBILITY_GATE_FAILED_EXTENSION_ONLY",
    )


__all__ = ["authority_selection"]
