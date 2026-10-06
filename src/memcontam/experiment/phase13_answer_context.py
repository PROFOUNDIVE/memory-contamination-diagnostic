from typing import Protocol

from memcontam.baselines.contracts import BaselineExecutionOutcome
from memcontam.logging.schema import MethodCall
from memcontam.logging.schema_v3 import ContextEvent


class AnswerContextIdentity(Protocol):
    @property
    def run_id(self) -> str: ...

    @property
    def trial_id(self) -> str: ...


def answer_context_event(
    outcome: BaselineExecutionOutcome,
    identity: AnswerContextIdentity,
    event_seq: int = 0,
) -> ContextEvent | None:
    answer_calls = tuple(
        call for call in outcome.method_calls
        if isinstance(call, MethodCall) and call.call_id == outcome.answer_call_id
    )
    if not answer_calls:
        return None
    return ContextEvent(
        record_type="context_event",
        event_id=f"{identity.trial_id}:context",
        context_id=f"{identity.trial_id}:context",
        run_id=identity.run_id,
        trial_id=identity.trial_id,
        event_seq=event_seq,
        final_entry_ids=list(dict.fromkeys(
            span.entry_id for span in answer_calls[-1].source_spans
        )),
    )
