"""Logging schema readers."""

from typing import Any, Mapping

from memcontam.logging.schema import LOGGING_V3, RunMetadata, TrialLog
from memcontam.logging.schema_v3 import Phase12Record, parse_log_record_v3


def parse_log_record(record: Mapping[str, Any]) -> TrialLog | RunMetadata | Phase12Record:
    if record.get("schema_version") == LOGGING_V3 or any(
        key in record for key in ("metadata_kind", "trial_kind", "record_type")
    ):
        return parse_log_record_v3(record)
    if "run_metadata_id" in record and "git_commit" in record:
        return RunMetadata.model_validate(record)
    return TrialLog.model_validate(record)


__all__ = ["parse_log_record", "parse_log_record_v3"]
