from __future__ import annotations

import pytest

from memcontam.clients.base import LLMResponse
from memcontam.readiness.phase13_main_production import ProductionObject
from memcontam.readiness.phase13_main_run_journal import ReconstructionFailureV3, RunJournalV3
from memcontam.readiness.phase13_production_observability import ProductionObservabilityError
from memcontam.readiness.phase13_v3_terminal_ledger import TerminalLedgerV3
from memcontam.readiness.phase13_v3_terminal_models import TerminalEvidenceError

from .phase13_count_fake import CountedProvider
from .test_phase13_v3_envelope_gate import api as api
from .test_phase13_v3_envelope_gate import rig as rig


@pytest.mark.parametrize("cause,expected", [
    (None, "UNREGISTERED_RECONSTRUCTION_CAUSE"),
    (ProductionObservabilityError("PRODUCTION_REGISTRATION_PACKET_MISMATCH"),
     "PRODUCTION_REGISTRATION_PACKET_MISMATCH"),
])
def test_reconstruction_reopens_completed_reference_without_duplicate_cost_reconciliation(
    rig, cause: ProductionObservabilityError | None, expected: str,
) -> None:
    class Provider(CountedProvider):
        def send_compiled_v3(self, compiled, before_request):
            before_request()
            return LLMResponse("final: 24", {
                "status": "completed", "model": "gpt-5.6-luna", "service_tier": "default",
                "usage": {"input_tokens": 1, "output_tokens": 1, "cached_input_tokens": 0},
                "authoritative_provider_cost_usd": "0.0000014", "currency": "USD",
                "response_id": "fake-journal-response",
            }, {"prompt_tokens": 1, "completion_tokens": 1}, 0)

    rig.dispatcher._factory = lambda _binding: Provider()
    rig.dispatcher.dispatch(rig.keys[0], rig.material, rig.semantic)
    key = rig.keys[0]
    with pytest.raises(TerminalEvidenceError):
        rig.ledger.reconcile_cost(key.dispatch_id, {
            "usage": {"input_tokens": 1, "output_tokens": 1, "cached_input_tokens": 0},
        }, "f" * 64)
    state = rig.ledger.state(key.dispatch_id)
    assert state.completion_hash is not None
    with rig.ledger.connection() as connection:
        connection.execute("CREATE TABLE run_journal (sequence INTEGER PRIMARY KEY, raw BLOB NOT NULL, sha256 TEXT NOT NULL)")
    unit = ProductionObject(sequence=0, unit_id=key.parent_id, kind="MEMORY_BEARING", seed=0,
                            task="game24", memory_baseline="rag_frozen", arm="contam",
                            prefix_unit_id=None, projected_cost_krw=0)
    journal = RunJournalV3(rig.ledger, (unit,))

    journal.reconstruction(unit, cause)
    reopened = TerminalLedgerV3.open(rig.ledger.path, rig.ledger.binding)
    try:
        rows = RunJournalV3(reopened, (unit,)).rows()
        assert len(rows) == 1
        assert isinstance(rows[0], ReconstructionFailureV3)
        assert rows[0].completions[0].event_hash == state.completion_hash
        assert rows[0].inner_code == expected
    finally:
        reopened.close()
