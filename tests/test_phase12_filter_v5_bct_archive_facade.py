from __future__ import annotations

from pathlib import Path

from memcontam.experiment.phase12.filter_challenge.bct_archive import (
    ArchiveValidation as facade_archive_validation,
)
from memcontam.experiment.phase12.filter_challenge.bct_archive import (
    BudgetLedger as facade_budget_ledger,
)
from memcontam.experiment.phase12.filter_challenge.bct_archive import (
    LedgerError as facade_ledger_error,
)
from memcontam.experiment.phase12.filter_challenge.bct_archive import (
    ProcessDeadline as facade_process_deadline,
)
from memcontam.experiment.phase12.filter_challenge.bct_archive import (
    ProcessReservation as facade_process_reservation,
)
from memcontam.experiment.phase12.filter_challenge.bct_archive import (
    ResourceBudget as facade_resource_budget,
)
from memcontam.experiment.phase12.filter_challenge.bct_archive import (
    SHARED_WALL_SECONDS as facade_shared_wall_seconds,
)
from memcontam.experiment.phase12.filter_challenge.bct_archive import (
    append_archive_record as facade_append_archive_record,
)
from memcontam.experiment.phase12.filter_challenge.bct_archive import (
    build_evidence_report as facade_build_evidence_report,
)
from memcontam.experiment.phase12.filter_challenge.bct_archive import (
    validate_evidence_bundle as facade_validate_evidence_bundle,
)
from memcontam.experiment.phase12.filter_challenge.bct_archive import (
    validate_live_archive as facade_validate_live_archive,
)
from memcontam.experiment.phase12.filter_challenge.bct_archive_evidence import (
    build_evidence_report,
    validate_evidence_bundle,
)
from memcontam.experiment.phase12.filter_challenge.bct_archive_ledger import (
    BudgetLedger,
    SHARED_WALL_SECONDS,
)
from memcontam.experiment.phase12.filter_challenge.bct_archive_models import (
    ArchiveValidation,
    LedgerError,
    ProcessDeadline,
    ProcessReservation,
    ResourceBudget,
)
from memcontam.experiment.phase12.filter_challenge.bct_archive_records import (
    append_archive_record,
    validate_live_archive,
)


def test_archive_facade_reexports_each_moved_contract_symbol() -> None:
    assert facade_archive_validation is ArchiveValidation
    assert facade_budget_ledger is BudgetLedger
    assert facade_ledger_error is LedgerError
    assert facade_process_deadline is ProcessDeadline
    assert facade_process_reservation is ProcessReservation
    assert facade_resource_budget is ResourceBudget
    assert facade_shared_wall_seconds == SHARED_WALL_SECONDS
    assert facade_append_archive_record is append_archive_record
    assert facade_build_evidence_report is build_evidence_report
    assert facade_validate_evidence_bundle is validate_evidence_bundle
    assert facade_validate_live_archive is validate_live_archive


def test_archive_and_live_split_modules_stay_within_the_pure_loc_ceiling() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src/memcontam/experiment/phase12/filter_challenge"
    modules = (*source_root.glob("bct_archive*.py"), *source_root.glob("bct_live*.py"))

    assert all(
        sum(bool(line.strip()) and not line.lstrip().startswith("#") for line in path.read_text().splitlines())
        <= 250
        for path in modules
    )
