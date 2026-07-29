from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from pydantic import BaseModel
from memcontam.experiment.phase12.filter_challenge.bct import BCT_TEST_IDS
from memcontam.experiment.phase12.filter_challenge.contracts import FilterPolicyIdentity
from memcontam.experiment.phase12.filter_challenge.domain_schema import (
    policy_visible_schema_boundary_valid,
    public_domain_schema_hashes,
)
from memcontam.experiment.phase12.filter_challenge.evidence_contract import (
    EVIDENCE_FILENAMES,
    EvidenceBuildError,
    canonical_json_bytes,
    json_value_from_bytes,
    sha256_bytes,
)
from memcontam.experiment.phase12.filter_challenge.mft import MFT_IDS, MergedMftReport
from memcontam.experiment.phase12.filter_challenge.mft_state_models import JsonValue
from memcontam.experiment.phase12.filter_challenge.final_verifier_plan_terminal_checks import (
    clause_10,
    clause_11,
    clause_12,
)
from memcontam.experiment.phase12.filter_challenge.registry_search import SearchConfig


LedgerCheck = Callable[[Path, JsonValue], bool]
_COMMON_HEADER_FIELDS: Final = {
    "amendment", "authority_hashes", "config_schema_hashes", "implementation_commit",
    "plan_sha256", "policy", "validation_summary_sha256",
}


@dataclass(frozen=True, slots=True)
class LedgerChecks:
    checks: tuple[LedgerCheck, ...]
    descriptions: tuple[tuple[str, str], ...]

    def __iter__(self) -> Iterator[LedgerCheck]:
        return iter(self.checks)


def _reports(root: Path) -> dict[str, dict[str, JsonValue]]:
    try:
        values = {
            name: json_value_from_bytes((root / name).read_bytes(), "EVIDENCE_REPORT_INVALID")
            for name in EVIDENCE_FILENAMES
        }
    except (EvidenceBuildError, OSError):
        return {}
    return {name: value for name, value in values.items() if isinstance(value, dict)}


def _report(root: Path, name: str) -> dict[str, JsonValue] | None:
    return _reports(root).get(name)


def _value(value: JsonValue, key: str) -> JsonValue | None:
    return value.get(key) if isinstance(value, dict) else None


def _mft(root: Path) -> dict[str, JsonValue] | None:
    report = _report(root, "mft_fv5_report.json")
    value = _value(report, "report") if report is not None else None
    return value if isinstance(value, dict) else None


def _state_result(mft: dict[str, JsonValue], test_id: str) -> dict[str, JsonValue] | None:
    state = _value(mft, "state_report")
    results = _value(state, "results")
    if not isinstance(results, list):
        return None
    return next((item for item in results if isinstance(item, dict) and item.get("test_id") == test_id), None)


def _safety_case(mft: dict[str, JsonValue], test_id: str) -> dict[str, JsonValue] | None:
    safety = _value(mft, "safety_report")
    cases = _value(safety, "cases")
    if not isinstance(cases, list):
        return None
    return next((item for item in cases if isinstance(item, dict) and item.get("test_id") == test_id), None)


def _machine_pass(result: dict[str, JsonValue] | None) -> bool:
    return result is not None and result.get("status") == "pass" and result.get("expected") == result.get("actual")


def _case_pass(case: dict[str, JsonValue] | None) -> bool:
    assertions = _value(case, "assertions")
    return (
        case is not None
        and case.get("status") == "pass"
        and isinstance(assertions, list)
        and all(
            isinstance(item, dict)
            and item.get("matched") is True
            and item.get("expected") == item.get("actual")
            for item in assertions
        )
    )


def _clause_1(root: Path, summary: JsonValue) -> bool:
    del summary
    policy = _report(root, "policy_schema_hashes.json")
    header = _value(policy, "header")
    return (
        policy is not None
        and _value(_value(header, "policy"), "identity") == "Filter-Challenge-v1"
        and _value(policy, "domain_model_schema_hashes") == public_domain_schema_hashes()
        and _value(policy, "policy_visible_schema_boundary") == "pass"
        and policy_visible_schema_boundary_valid()
    )


def _clause_2(root: Path, summary: JsonValue) -> bool:
    del summary
    mft = _mft(root)
    return mft is not None and _machine_pass(_state_result(mft, "MFT-FV5-01-PAIR-MATCH")) and _machine_pass(_state_result(mft, "MFT-FV5-08-NO-WRITEBACK"))


def _clause_3(root: Path, summary: JsonValue) -> bool:
    del summary
    mft = _mft(root)
    if mft is None or not _machine_pass(_state_result(mft, "MFT-FV5-02-EXPOSURE-REQUIRED")):
        return False
    case = _safety_case(mft, "MFT-FV5-12-PROBE-KEY-INVARIANCE")
    assertions = _value(case, "assertions")
    families = next(
        (item for item in assertions if isinstance(item, dict) and item.get("field") == "candidate_families"),
        None,
    ) if isinstance(assertions, list) else None
    return _case_pass(case) and isinstance(families, dict) and families.get("actual") == ["full_history", "rag_frozen", "bot_style", "reflexion_style"]


def _clause_4(root: Path, summary: JsonValue) -> bool:
    del summary
    mft = _mft(root)
    provenance = _report(root, "answer_call_provenance_report.json")
    return mft is not None and _case_pass(_safety_case(mft, "MFT-FV5-13-ANSWER-CALL-PROVENANCE")) and _value(provenance, "mft_status") == "pass"


def _clause_5(root: Path, summary: JsonValue) -> bool:
    del summary
    mft = _mft(root)
    return mft is not None and _machine_pass(_state_result(mft, "MFT-FV5-03-TRISTATE")) and _machine_pass(_state_result(mft, "MFT-FV5-04-FAIL-OPEN")) and _case_pass(_safety_case(mft, "MFT-FV5-15-ELIGIBILITY-STATES"))


def _clause_6(root: Path, summary: JsonValue) -> bool:
    del summary
    policy = _report(root, "policy_schema_hashes.json")
    mft = _mft(root)
    header = _value(policy, "header")
    hashes = _value(header, "config_schema_hashes")
    expected = {
        "domain_contract": _schema_hash(FilterPolicyIdentity),
        "mft": _schema_hash(MergedMftReport),
        "search_config_schema": _schema_hash(SearchConfig),
    }
    return (
        mft is not None
        and isinstance(hashes, dict)
        and all(hashes.get(name) == digest for name, digest in expected.items())
        and hashes.get("search_config") == mft.get("search_config_hash")
    )


def _clause_7(root: Path, summary: JsonValue) -> bool:
    del summary
    archive = _value(_report(root, "archive_validation_report.json"), "report")
    mft = _mft(root)
    header = _value(_report(root, "policy_schema_hashes.json"), "header")
    return (
        isinstance(archive, dict)
        and mft is not None
        and archive.get("archive_valid") is True
        and archive.get("provider_calls_issued") == 0
        and archive.get("implementation_commit") == _value(header, "implementation_commit")
        and archive.get("search_config_hash") == mft.get("search_config_hash")
        and archive.get("freeze_id") == "phase12-filter-v5-build-freeze-v1"
        and archive.get("run_id") == "filter-v5-build-synthetic"
    )


def _clause_8(root: Path, summary: JsonValue) -> bool:
    del summary
    mft = _mft(root)
    counts = _value(mft, "execution_counts")
    return (
        mft is not None
        and mft.get("ordered_test_ids") == list(MFT_IDS)
        and mft.get("all_passed") is True
        and isinstance(counts, list)
        and [(item.get("test_id"), item.get("count")) for item in counts if isinstance(item, dict)]
        == [(test_id, 1) for test_id in MFT_IDS]
    )


def _clause_9(root: Path, summary: JsonValue) -> bool:
    del summary
    readiness = _value(_report(root, "bct_readiness_report.json"), "report")
    families = _value(readiness, "family_statuses")
    return (
        isinstance(readiness, dict)
        and readiness.get("software_interface_status") == "ready"
        and readiness.get("execution_status") == "blocked"
        and isinstance(families, list)
        and [(item.get("test_id"), item.get("status")) for item in families if isinstance(item, dict)]
        == [(test_id, "not_executed") for test_id in BCT_TEST_IDS]
    )


def _schema_hash(model: type[BaseModel]) -> str:
    return sha256_bytes(canonical_json_bytes(model.model_json_schema()))


LEDGER_CHECKS: Final = LedgerChecks(
    checks=(
        _clause_1, _clause_2, _clause_3, _clause_4, _clause_5, _clause_6,
        _clause_7, _clause_8, _clause_9,
        lambda root, summary: clause_10(_report, root, summary),
        lambda root, summary: clause_11(_report, root, summary),
        lambda root, summary: clause_12(_report, root, summary),
    ),
    descriptions=(
        ("ledger-01-versioned-domain", "versioned domain schema boundary"),
        ("ledger-02-read-only-pair", "isolated matched read-only pair"),
        ("ledger-03-native-adapters", "native adapter exposure semantics"),
        ("ledger-04-answer-provenance", "answer-call provenance relations"),
        ("ledger-05-routing", "eligibility witness and routing"),
        ("ledger-06-configuration", "strict configuration registry"),
        ("ledger-07-archive", "archive logging reconciliation"),
        ("ledger-08-mft", "exact MFT execution"),
        ("ledger-09-bct", "behavioral readiness interfaces"),
        ("ledger-10-validation", "validation gate evidence"),
        ("ledger-11-evidence", "tracked evidence graph"),
        ("ledger-12-terminal", "terminal metadata availability"),
    ),
)
