from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Final

from memcontam.rag.phase12_corpus import CleanCorpus, build_branch_corpora

from .phase13_core_datasets import CANONICAL_CORE_ARTIFACT_SHA256
from .phase13_legacy_rag_models import PackageManifest, SerializedDocument
from .phase13_legacy_rag_serialization import hash_json
from .phase13_legacy_rag_validation_checks import IndexCheckSource, validate_indices
from .phase13_main_checkpoint import TaskSeedOrders, _canonical_hash, _expected_registry
from .phase13_main_preloaded_resources import PreloadedMainResources
from .phase13_main_production import _execution_template_id
from .phase13_v3_entrypoint import EntrypointError, SelectedExecutionV3

TASKS: Final = ("game24", "math_equation_balancer", "word_sorting", "mmlu_pro_engineering", "mmlu_pro_physics")
LEGACY: Final = TASKS[:3]
RESOURCE_PATHS: Final = {
    "common_checkpoint_registry": "data/phase13/main/mr_p4/main_a_common_checkpoint_registry_v1.json",
    "task_seed_orders": "data/phase13/main/mr_p4/task_seed_orders_v1.json",
    "observability_packet": "data/phase13/observability/registration_packet_v2.json",
    "candidate_registry": "data/phase12/registries/candidate_registry_v2.json",
    "legacy_dc_rs_intervention_registry": "data/phase13/main/legacy_dc_rs_intervention_registry_v2.json",
    "main_new_mcq_authority_selection": "data/phase13/rag/new_mcq/authority_selection_v1.json",
    "main_new_mcq_intervention_registry": "data/phase13/rag/new_mcq/intervention_registry_v1.json",
    "legacy_rag_seal": "data/phase13/rag/legacy_seal_v2.json",
    "legacy_rag_manifest": "data/phase13/rag/legacy_v2/manifest.json",
    "activated_policy": "data/phase13/main/cost_envelope_v3/activated_policy_v3.json",
    "base_inputs": "data/phase13/main/cost_envelope_v3/base_inputs_v3.json",
    "cost_witness": "data/phase13/main/cost_envelope_v3/cost_witness_v3.json",
    **{f"task_{task}": f"data/phase13/main/{task}_main_v1.jsonl" for task in LEGACY},
    **{f"task_{task}": f"data/phase13/core/materialized/{task}.jsonl" for task in TASKS[3:]},
    **{f"rag_corpus_{task}": f"data/phase13/rag/legacy_v2/{task}/corpus.json" for task in LEGACY},
    **{f"rag_index_{task}": f"data/phase13/rag/legacy_v2/{task}/indices.json" for task in LEGACY},
}


def validate_resource_contract(selected: SelectedExecutionV3) -> PreloadedMainResources:
    if {row.role: row.path for row in selected.package.resources} != RESOURCE_PATHS:
        raise EntrypointError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
    loaded = PreloadedMainResources(selected)
    registry = loaded.checkpoint_registry
    orders_raw = selected.resource("task_seed_orders")
    orders = TaskSeedOrders.model_validate_json(orders_raw)
    expected_registry = _expected_registry(orders.model_dump(mode="json", exclude={"orders_hash"}), selected.resource_binding("task_seed_orders").sha256)
    if (set(registry.tasks) != set(TASKS) or set(orders.tasks) != set(TASKS)
        or orders.concrete_seed_ids != tuple(range(10))
        or registry.model_dump(mode="json", exclude={"registry_hash"}) != expected_registry
        or registry.registry_hash != _canonical_hash(expected_registry)
        or orders.orders_hash != _canonical_hash(orders.model_dump(mode="json", exclude={"orders_hash"}))):
        raise EntrypointError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
    for task in TASKS:
        rows = loaded.tasks(task)
        sample_ids = tuple(row.sample_id for row in rows)
        if (len(set(sample_ids)) != len(sample_ids) or set(sample_ids) != set(orders.tasks[task].sample_ids)
            or any(row.task_name != task for row in rows)
            or tuple(seed.seed for seed in orders.tasks[task].seeds) != tuple(range(10))):
            raise EntrypointError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
        expected_core_hash = next((value for name, value in CANONICAL_CORE_ARTIFACT_SHA256.items() if name == task), None)
        if expected_core_hash is not None and selected.resource_binding("task_" + task).sha256 != expected_core_hash:
            raise EntrypointError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
        for seed in orders.tasks[task].seeds:
            expected_order = tuple(sorted(sample_ids, key=lambda sample: hashlib.sha256(
                f"sha256_task_seed_v1\0{task}\0{seed.seed}\0{sample}".encode()).digest()))
            if seed.ordered_sample_ids != expected_order or seed.order_sha256 != hashlib.sha256(
                json.dumps(expected_order, separators=(",", ":")).encode()).hexdigest():
                raise EntrypointError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
    _rag(loaded)
    _production(loaded)
    return loaded


def _rag(loaded: PreloadedMainResources) -> None:
    selected = loaded.selected
    raw = selected.resource("legacy_rag_manifest")
    manifest = PackageManifest.model_validate_json(raw)
    seal = json.loads(selected.resource("legacy_rag_seal"))
    if (seal["manifest_sha256"] != selected.resource_binding("legacy_rag_manifest").sha256
        or manifest.materialization_profile != "production_bge_m3"):
        raise EntrypointError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
    triplets = {row.task: row for row in loaded.candidate_registry.triplets}
    for task in LEGACY:
        corpus, index = loaded.legacy_bundles(task)
        for role, filename in (("rag_corpus_", "corpus.json"), ("rag_index_", "indices.json")):
            if selected.resource_binding(role + task).sha256 != manifest.artifact_hashes[f"{task}/{filename}"]:
                raise EntrypointError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
        clean = CleanCorpus.from_documents([{"id": row.document_id, "text": row.text} for row in corpus.clean_documents],
                                          corpus_id=f"phase13_legacy_rag_v1::{task}")
        expected = build_branch_corpora(clean, triplets[task])
        if (corpus.triplet_artifact_hash != hash_json(asdict(triplets[task]))
            or corpus.triplet_registry.sha256 != selected.resource_binding("candidate_registry").sha256
            or index.embedding_runtime.embedding_library_version != "5.6.0"):
            raise EntrypointError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
        for branch, actual in corpus.branches.items():
            value = expected.branches[branch]
            if (actual.documents != tuple(SerializedDocument.model_validate(row.payload()) for row in value.documents)
                or actual.active_document_ids != value.active_document_ids or actual.serialization_id != value.serialization_id):
                raise EntrypointError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
        validate_indices(IndexCheckSource(corpus.task_id, corpus, index))


def _production(loaded: PreloadedMainResources) -> None:
    package = loaded.selected.package
    costs = {row.unit_id: row.projected_krw for row in loaded.selected.costs.resources.proof.projected_krw}
    prefixes = {row.unit_id: row for row in package.production if row.kind == "CLEAN_PREFIX"}
    packet_hash = loaded.selected.resource_binding("observability_packet").sha256
    checkpoint_hash = loaded.selected.resource_binding("common_checkpoint_registry").sha256
    for sequence, unit in enumerate(package.production):
        if (unit.task not in TASKS or unit.seed not in range(10)
            or unit.memory_baseline not in {None, "fh_bounded", "rag_frozen", "bot_style", "reflexion_style", "dc_rs"}
            or unit.execution_template_id != _execution_template_id(unit)):
            raise EntrypointError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
        seed = loaded.checkpoint_registry.tasks[unit.task].seeds[unit.seed]
        identity = ["phase13-main-a-disjoint-unit-id-v1", unit.kind, unit.seed, unit.task, unit.memory_baseline, unit.arm]
        if (unit.sequence != sequence or unit.projected_cost_krw != costs[unit.unit_id]
            or unit.unit_id != hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest()
            or unit.checkpoint_registry_sha256 != checkpoint_hash or unit.registration_packet_sha256 != packet_hash
            or unit.ordered_sample_ids_sha256 != seed.suffix_sample_ids_sha256):
            raise EntrypointError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
        if unit.kind == "MEMORY_BEARING":
            prefix = prefixes.get(unit.prefix_unit_id or "")
            if prefix is None or (prefix.task, prefix.seed, prefix.memory_baseline) != (unit.task, unit.seed, unit.memory_baseline):
                raise EntrypointError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
