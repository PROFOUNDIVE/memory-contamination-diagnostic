from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, fields, replace

from memcontam.experiment.phase12.filter_challenge.assessment import route_assessment
from memcontam.experiment.phase12.filter_challenge.executor import (
    ActivationContext,
    activation_decision,
    build_control_cache_key,
    build_shared_assessment_key,
    consume_routing,
)
from memcontam.experiment.phase12.filter_challenge.executor_types import (
    ControlCacheKey,
    PairAuditEvidence,
    PairingIdentity,
)
from memcontam.experiment.phase12.filter_challenge.mft_safety_types import (
    CONTROL_CACHE_FIELDS,
    PAIR_ADAPTER,
    ROUTABILITY_ADAPTER,
    GateEvidence,
    MftIdentity,
    assertion,
    candidate,
    pairing_identity,
)


def gate_shadow_share(mutated: bool) -> GateEvidence:
    identity, challenge_candidate = pairing_identity(), candidate()
    key = build_shared_assessment_key(
        identity, challenge_candidate, "candidate-v1", "probe-config-hash"
    )
    evidence = PairAuditEvidence(
        assessment_id="assessment-1", adapter_name="scripted",
        paired_execution_identity=PAIR_ADAPTER.validate_python(
            {"paired_execution_identity_status": "matched", "pair_id": "assessment-1"}
        ),
        control_cache_key=build_control_cache_key(identity), shared_assessment_key=key,
        control_from_cache=False, control_answer_call_id="control", challenge_answer_call_id="challenge",
        control_displaced_noncandidate_entry_ids=(), challenge_displaced_noncandidate_entry_ids=(),
    )
    consumed = consume_routing("assessment-1", route_assessment("contradicted"), key, evidence)
    routing = (f"{consumed.contam.effect}:{consumed.contam.route_target or 'none'}",
               f"{consumed.filter.effect}:{consumed.filter.route_target or 'none'}")
    if mutated:
        routing = (routing[0], "shadow:none")
    key_hash = hashlib.sha256(json.dumps(asdict(key), sort_keys=True).encode()).hexdigest()
    return GateEvidence(
        (MftIdentity(field="assessment_id", value=evidence.assessment_id),
         MftIdentity(field="shared_assessment_key_hash", value=key_hash)),
        (assertion("assessment_identity", ("assessment-1", "assessment-1"),
                   (evidence.assessment_id, evidence.assessment_id)),
         assertion("routing_consumption", ("shadow:none", "apply:quarantine"), routing)),
    )


def _cache_mutations(identity: PairingIdentity) -> tuple[PairingIdentity, ...]:
    return (
        replace(identity, source_checkpoint_hash="changed"), replace(identity, baseline_family="rag_frozen"),
        replace(identity, rag_mode="frozen"), replace(identity, probe_id="changed"),
        replace(identity, prompt_payload_hash="changed"), replace(identity, replicate_seed_contract="seed_coupled"),
        replace(identity, replicate_id=1), replace(identity, paired_seed_replay_id="changed"),
        replace(identity, model_snapshot="changed"), replace(identity, decoding_contract_hash="changed"),
        replace(identity, fidelity_label="changed"), replace(identity, tool_mode="changed"),
        replace(identity, tool_permissions_hash="changed"), replace(identity, raw_parser_version="changed"),
        replace(identity, canonicalizer_version="changed"), replace(identity, verifier_version="changed"),
        replace(identity, base_prompt_hash="changed"), replace(identity, formatter_version="changed"),
        replace(identity, context_budget_capacity_hash="changed"),
        replace(identity, retriever_index_capacity_hash="changed"),
        replace(identity, noncandidate_memory_hash="changed"),
        replace(identity, resource_retry_limit_hash="changed"),
    )


def gate_control_cache(mutated: bool) -> GateEvidence:
    identity = pairing_identity()
    key = build_control_cache_key(identity)
    equal_identity = replace(identity, source_checkpoint_hash="mutated") if mutated else replace(identity)
    changed_keys = tuple(build_control_cache_key(item) for item in _cache_mutations(identity))
    return GateEvidence(
        (MftIdentity(field="cache_key_type", value=type(key).__name__),),
        (assertion("cache_key_fields", CONTROL_CACHE_FIELDS,
                   tuple(field.name for field in fields(ControlCacheKey))),
         assertion("equal_identity_key_equality", ("true",),
                   (str(key == build_control_cache_key(equal_identity)).lower(),)),
         assertion("all_field_mutation_sensitivity", (str(len(CONTROL_CACHE_FIELDS)),),
                   (str(sum(changed != key for changed in changed_keys)),)),
         assertion("mutated_keys_remain_distinct", (str(len(CONTROL_CACHE_FIELDS)),),
                   (str(len(set(changed_keys))),))),
    )


def gate_activation(mutated: bool) -> GateEvidence:
    context = ActivationContext("tau", "branch-later", ("grandfathered",), "candidate-1", "filter")
    base = candidate()
    grandfathered = base.model_copy(update={"candidate_entry_id": "grandfathered"})
    root = base.model_copy(update={"source_checkpoint_id": "tau"})
    later = base.model_copy(update={"candidate_entry_id": "later", "source_checkpoint_id": "branch-later"})
    unsupported = root.model_copy(update={
        "candidate_entry_id": "unsupported",
        "routability": ROUTABILITY_ADAPTER.validate_python(
            {"routability": "unsupported", "reason_code": "PROBE_MAPPING_UNSUPPORTED"}
        ),
    })
    paths = (
        activation_decision(context, grandfathered, later_native_write=False).status,
        activation_decision(context, root, later_native_write=False).status,
        activation_decision(context, later, later_native_write=True).status,
        activation_decision(context, unsupported, later_native_write=False).status,
    )
    if mutated:
        paths = (paths[0], paths[1], "not_assessed", paths[3])
    return GateEvidence(
        (MftIdentity(field="activation_checkpoint_id", value="tau"),
         MftIdentity(field="evolved_checkpoint_id", value="branch-later")),
        (assertion("activation_paths", ("grandfathered", "assess", "assess", "not_evaluable"), paths),),
    )
