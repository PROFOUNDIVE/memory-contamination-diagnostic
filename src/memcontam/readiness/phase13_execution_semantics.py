from __future__ import annotations

import hashlib
import json
from typing import Final

from memcontam.readiness.phase13_execution_models import ExecutionRegistry


PARTITION_SHA256: Final = "a31b731244f5c56b4aafa5ed83bbe720c8623563cfbd800e7a478a0025aff4ba"
IDENTITIES: Final = {
    "provider_id": "openai-responses-v1",
    "model_snapshot_id": "gpt-4o-2024-11-20",
    "decoding_contract_id": "phase13-decoding-zero-v1",
    "prompt_contract_id": "baseline-fidelity-v2-prompts",
    "tool_contract_id": "text-only-equal-availability-v1",
    "parser_contract_id": "phase13-task-parsers-v1",
    "verifier_contract_id": "phase13-task-verifiers-v1",
    "resource_contract_id": "phase13-resource-envelope-v1",
    "session_contract_id": "paired-isolated-session-v1",
    "failure_contract_id": "baseline-fidelity-v2-failure-taxonomy",
    "retry_contract_id": "four-transport-attempts-v1",
    "checkpoint_serializer_id": "phase12-native-checkpoint-v1",
    "task_stream_contract_id": "phase13-calibration-v2-rotations-v1",
    "native_capacity_registry_id": "phase13-native-capacity-v1",
}
ARMS: Final = (
    ("Clean", "phase13-matched-native-branch-v1", "clean-no-insertion-v1"),
    ("Correct", "phase13-matched-native-branch-v1", "correct-control-v1"),
    ("Irrelevant", "phase13-matched-native-branch-v1", "irrelevant-control-v1"),
    ("Contam", "phase13-matched-native-branch-v1", "canonical-false-candidate-v1"),
)
CAPACITIES: Final = (
    ("fh_bounded", "fh-bounded-10000-v1", "oldest_first_truncation", 10000, True),
    ("rag_frozen", "rag-frozen-corpus-v1", "frozen_corpus_no_online_admission", 3, True),
    ("bot_style", "bot-active-capacity-3-v1", "bounded_template_eviction", 3, True),
    ("reflexion_style", "reflexion-active-capacity-3-v1", "bounded_reflection_eviction", 3, True),
)
COMPONENTS: Final = (
    ("prefix-burn-init-calls", "prefix", "phase13-clean-prefix-owner-v1", "burn_init", 6, 9),
    ("execution-trial-calls", "execution", "phase13-h10-execution-owner-v1", "trial", 25, 37),
)
WINDOWS: Final = (
    ("accuracy-h2-sensitivity", 2, 1, "verified_accuracy", "prespecified_sensitivity", "descriptive_no_inferential_family", "prefix_view", 0),
    ("recurrence-h2-descriptive", 2, 1, "recurrence", "descriptive", "estimation_only", "prefix_view", 0),
    ("accuracy-h5-primary", 5, 4, "verified_accuracy", "confirmatory_primary", "primary_holm_family", "prefix_view", 0),
    ("recurrence-h5-secondary", 5, 4, "recurrence", "confirmatory_secondary", "estimation_only", "prefix_view", 0),
    ("persistence-h5-secondary", 5, 4, "persistence", "confirmatory_secondary", "estimation_only", "prefix_view", 0),
    ("propagation-h5-conditional", 5, 4, "propagation", "descriptive", "descriptive_no_inferential_family", "prefix_view", 0),
    ("collapse-h5-exploratory", 5, 4, "collapse_like", "exploratory", "descriptive_no_inferential_family", "prefix_view", 0),
    ("accuracy-h10-sensitivity", 10, 9, "verified_accuracy", "prespecified_sensitivity", "descriptive_no_inferential_family", "source_execution", 1),
    ("recurrence-h10-descriptive", 10, 9, "recurrence", "descriptive", "estimation_only", "source_execution", 0),
    ("persistence-h10-descriptive", 10, 9, "persistence", "descriptive", "estimation_only", "source_execution", 0),
    ("propagation-h10-conditional", 10, 9, "propagation", "descriptive", "descriptive_no_inferential_family", "source_execution", 0),
    ("collapse-h10-exploratory", 10, 9, "collapse_like", "exploratory", "descriptive_no_inferential_family", "source_execution", 0),
)
TASKS: Final = ("game24", "math_equation_balancer", "word_sorting")
BASELINE_CALLS: Final = {
    "fh_bounded": (1, 1),
    "rag_frozen": (1, 1),
    "bot_style": (2, 3),
    "reflexion_style": (2, 4),
}


def ordered_stream_hash(sample_ids: list[str]) -> str:
    return hashlib.sha256(
        (json.dumps({"ordered_sample_ids": sample_ids}, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()


def validate_exact_inventory(registry: ExecutionRegistry) -> str | None:
    if registry.identities.model_dump() != IDENTITIES:
        return "EXECUTION_IDENTITY_INVALID"
    if tuple((row.arm_key, row.branch_constructor_id, row.candidate_or_control_id) for row in registry.memory_arms) != ARMS:
        return "ARM_REGISTRY_INVALID"
    if registry.prefix_owner_id == registry.execution_owner_id:
        return "OWNER_IDENTITY_INVALID"
    if tuple((row.component_id, row.owner_kind, row.owner_id, row.phase, row.nominal_calls_per_activation, row.raw_maximum_calls_per_activation) for row in registry.call_components) != COMPONENTS:
        return "CALL_COMPONENT_INVALID"
    if tuple((row.baseline, row.capacity_contract_id, row.transition_policy, row.configured_limit, row.insertion_supported) for row in registry.native_capacities) != CAPACITIES:
        return "NATIVE_CAPACITY_INVALID"
    projection = tuple((row.analysis_window_id, row.window_length, row.event_time_end, row.outcome_family, row.evidence_status, row.multiplicity_status, row.realization_disposition, row.provider_execution_multiplicity) for row in registry.analysis_windows)
    if projection != WINDOWS:
        return "WINDOW_INVENTORY_INVALID"
    expected_templates = tuple(
        row
        for task in TASKS
        for row in (
            *(
                (f"{task}-{baseline}-{arm.lower()}", "phase13-h10-execution-owner-v1", task, baseline, arm, *BASELINE_CALLS[baseline], 10, 12)
                for baseline in BASELINE_CALLS
                for arm in ("Clean", "Correct", "Irrelevant", "Contam")
            ),
            (f"{task}-nomem-singleton", "phase13-h10-execution-owner-v1", task, "nomem", "star_NoMem", 1, 1, 10, 12),
        )
    )
    templates = tuple(
        (row.template_id, row.owner_id, row.task, row.baseline, row.arm_key, row.nominal_semantic_calls_per_trial, row.raw_maximum_semantic_calls_per_trial, row.main_seed_multiplicity, row.calibration_seed_multiplicity)
        for row in registry.execution_templates
    )
    if templates != expected_templates:
        return "EXECUTION_TEMPLATE_INVALID"
    return None
