from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Final, assert_never

from pydantic import TypeAdapter

from memcontam.baselines import dynamic_cheatsheet_phase12 as dc
from memcontam.experiment.phase13_dc_rs_runtime import serialize
from memcontam.memory.checkpoint_v3 import NativeEntry, NativeState
from memcontam.memory.embeddings import BgeM3EmbeddingProvider
from memcontam.tasks.base import TaskInstance
from memcontam.verifiers.math_equation_balancer import _parse_input

from .phase13_authority_files import read_regular_nofollow
from .phase13_diagnostic_cli_models import NativeInput
from .phase13_diagnostic_proposal_models import (
    PreparationMetadata, PreparedProposal, Prerequisite, ProposalProjection, ProposalUnit, Task, Task11Proposal,
)
from .phase13_cost_policy_models import Sha256
from .phase13_legacy_rag_audit import _canonical_signature_bytes
from .phase13_live_diagnostic_models import LiveDiagnosticError, LiveDiagnosticScope
from .phase13_shared_budget_models import FilePin

ARMS: Final = ("clean", "correct", "irrelevant", "contam")
GROUPS: Final = tuple((task, baseline) for task in ("math_equation_balancer", "word_sorting")
                     for baseline in ("bot_style", "dc_rs"))
DC_ORDER: Final = ("retrieve", "curate_complete_cheatsheet", "generate",
                   "append_raw_interaction", "persist_complete_cheatsheet")


class InventoryInput(ProposalProjection):
    task: Task
    task_instance: TaskInstance
    canonical_signature: Sha256


class EquationInput(ProposalProjection):
    input: str


class WordsInput(ProposalProjection):
    words: tuple[str, ...]


def validate_input_contracts(proposal: Task11Proposal) -> None:
    resources = proposal.resources
    if (resources.embedding != f"{BgeM3EmbeddingProvider.MODEL_ID}@{BgeM3EmbeddingProvider.REVISION}"
        or resources.dimension != BgeM3EmbeddingProvider.VECTOR_DIMENSION
        or resources.normalize != BgeM3EmbeddingProvider.NORMALIZE_EMBEDDINGS):
        raise LiveDiagnosticError("DIAGNOSTIC_PROPOSAL_EMBEDDING_CONTRACT_MISMATCH")
    for item in proposal.inputs:
        match item.task:
            case "math_equation_balancer":
                equation = EquationInput.model_validate_json(json.dumps(item.task_instance.input))
                parsed = _parse_input(equation.input)
                if parsed is None:
                    raise LiveDiagnosticError("DIAGNOSTIC_PROPOSAL_INPUT_SIGNATURE_UNPROVEN")
                reduced: dict[str, int | list[int] | list[str]] = {
                    "ordered_operands": list(parsed[0]), "target_value": parsed[1]}
            case "word_sorting":
                words = WordsInput.model_validate_json(json.dumps(item.task_instance.input))
                reduced = {"tokens": sorted(unicodedata.normalize("NFC", word) for word in words.words)}
            case unreachable:
                assert_never(unreachable)
        if hashlib.sha256(_canonical_signature_bytes(reduced)).hexdigest() != item.canonical_signature:
            raise LiveDiagnosticError("DIAGNOSTIC_PROPOSAL_INPUT_SIGNATURE_MISMATCH")
        inventory = TypeAdapter(tuple[InventoryInput | None, ...]).validate_json(item.inventory_binding.read())
        if not 0 <= item.inventory_index < len(inventory):
            raise LiveDiagnosticError("DIAGNOSTIC_PROPOSAL_INPUT_INVENTORY_MISMATCH")
        selected = inventory[item.inventory_index]
        if (selected is None or (selected.task, selected.task_instance, selected.canonical_signature)
            != (item.task, item.task_instance, item.canonical_signature)):
            raise LiveDiagnosticError("DIAGNOSTIC_PROPOSAL_INPUT_INVENTORY_MISMATCH")


def validate_dc_payload(unit: ProposalUnit, native: NativeInput, scope: LiveDiagnosticScope) -> None:
    render = unit.render
    if render is None or render.query is None or render.response is None or render.candidate_id is None:
        raise LiveDiagnosticError("DIAGNOSTIC_PREPARATION_DC_PAYLOAD_FACT_REQUIRED")
    state = NativeState.from_mapping(json.loads(scope.state_utf8))
    selected = tuple(entry for entry in state.entries if isinstance(entry, NativeEntry)
                     and entry.entry_id == native.branch.injected_root_id)
    if len(selected) != 1:
        raise LiveDiagnosticError("DIAGNOSTIC_PREPARATION_DC_PAYLOAD_FACT_REQUIRED")
    entry = selected[0]
    expected = dc._archive_native(dc.MemoryEntry(entry_id=render.candidate_id, memory_type="dc_rs_io_pair",
        content=render.query, metadata={"generated_output": render.response, "render_id": render.render_id}))
    if ((entry.entry_id, entry.semantic_kind, entry.native_component, entry.schema_version,
         entry.content, entry.content_hash, entry.render_id)
        != (expected.entry_id, expected.semantic_kind, expected.native_component, expected.schema_version,
            expected.content, expected.content_hash, expected.render_id)
        or hashlib.sha256(entry.content.encode()).hexdigest() != render.serialized_content_sha256
        or native.branch.candidate_triplet_id != render.triplet_id
        or state.native_state.get("injected_root_id") != native.branch.injected_root_id):
        raise LiveDiagnosticError("DIAGNOSTIC_PREPARATION_DC_PAYLOAD_MISMATCH")
    restored = dc.DcRsStateV3(
        archive=[dc.MemoryEntry.model_validate(row) for row in state.native_state.get("archive", ())],
        strategies=[entry for entry in state.entries if isinstance(entry, NativeEntry)
                    and entry.native_component == "strategy"],
        strategy_history=[NativeEntry.from_mapping(row) for row in state.native_state.get("strategy_history", ())],
        injected_root_id=state.native_state.get("injected_root_id"),
        allow_unparented_strategies=state.native_state.get("allow_unparented_strategies", False))
    if serialize(restored) != state:
        raise LiveDiagnosticError("DIAGNOSTIC_PREPARATION_DC_ARCHIVE_MISMATCH")


def validate_proposal(proposal: Task11Proposal) -> None:
    validate_input_contracts(proposal)
    if (tuple((row.task, row.baseline, row.arm, row.seed) for row in proposal.ordinary_units)
        != tuple((task, baseline, arm, 0) for task, baseline in GROUPS for arm in ARMS)
        or tuple((row.task, row.baseline, row.seed) for row in proposal.checkpoint_groups)
        != tuple((task, baseline, 0) for task, baseline in GROUPS)
        or tuple(row.task for row in proposal.inputs) != ("math_equation_balancer", "word_sorting")):
        raise LiveDiagnosticError("DIAGNOSTIC_PROPOSAL_SCHEDULE_MISMATCH")
    dc = proposal.dc_contract
    if dc.native_order != DC_ORDER or dc.renderer_id != "legacy-dc-rs-canonical-renderer-v2":
        raise LiveDiagnosticError("DIAGNOSTIC_PROPOSAL_DC_CONTRACT_MISMATCH")
    for group in proposal.checkpoint_groups:
        expected = f"proposed16:{group.task}:{group.baseline}:seed0"
        if group.group_id != expected or group.arm_order != ARMS:
            raise LiveDiagnosticError("DIAGNOSTIC_PROPOSAL_GROUP_MISMATCH")
    for row in proposal.ordinary_units:
        item = next(item for item in proposal.inputs if item.task == row.task)
        group = next((group for group in proposal.checkpoint_groups if group.group_id == row.group_id), None)
        if group is None:
            raise LiveDiagnosticError("DIAGNOSTIC_PROPOSAL_GROUP_MISMATCH")
        if (row.unit_id != f"{row.group_id}:{row.arm}" or (group.task, group.baseline) != (row.task, row.baseline)
            or row.root_count != int(row.arm != "clean")):
            raise LiveDiagnosticError("DIAGNOSTIC_PROPOSAL_GROUP_MISMATCH")
        if (row.sample_id != item.task_instance.sample_id or row.sample_id != group.suffix_sample_id
            or row.inventory_index != item.inventory_index or row.input_binding != item.inventory_binding
            or item.task_instance.task_name != row.task):
            raise LiveDiagnosticError("DIAGNOSTIC_PROPOSAL_COMMON_INPUT_MISMATCH")
        render = row.render
        if (row.arm == "clean") != (render is None):
            raise LiveDiagnosticError("DIAGNOSTIC_PROPOSAL_RENDER_MISMATCH")
        if render is not None:
            if row.baseline == "bot_style" and render.carrier != "thought_template":
                raise LiveDiagnosticError("DIAGNOSTIC_PROPOSAL_RENDER_MISMATCH")
            if row.baseline == "dc_rs":
                if (render.carrier != "raw_interaction" or render.native_semantic_kind != "dc_rs_io_pair"
                    or render.serialization_id != "dc-rs-input-then-response-json-v1"
                    or render.renderer_id != dc.renderer_id or render.family_binding != dc.family
                    or any(value is None for value in (render.triplet_id, render.candidate_id, render.render_id,
                                                       render.query, render.response))):
                    raise LiveDiagnosticError("DIAGNOSTIC_PROPOSAL_DC_CONTRACT_MISMATCH")
                content = json.dumps({"input": render.query, "raw_output": render.response},
                                     sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                if (hashlib.sha256((render.query or "").encode()).hexdigest() != render.query_sha256
                    or hashlib.sha256((render.response or "").encode()).hexdigest() != render.response_sha256
                    or hashlib.sha256(content.encode()).hexdigest() != render.serialized_content_sha256):
                    raise LiveDiagnosticError("DIAGNOSTIC_PROPOSAL_DC_BYTES_MISMATCH")


def prepare_proposal(path: Path, metadata: PreparationMetadata | None) -> PreparedProposal:
    raw = read_regular_nofollow(path)
    pin = FilePin(path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest())
    proposal = Task11Proposal.model_validate_json(raw)
    validate_proposal(proposal)
    pending: list[Prerequisite] = []
    if metadata is None:
        for index, group in enumerate(proposal.checkpoint_groups):
            for field in ("prefix_id", "prefix_sample_ids", "checkpoint_id", "checkpoint_index", "state_sha256", "state_binding"):
                if getattr(group, field) is None:
                    pending.append(Prerequisite(path=f"checkpoint_groups.{index}.{field}", code="NATIVE_PREFIX_FACT_REQUIRED"))
        for index, unit in enumerate(proposal.ordinary_units):
            if unit.render is not None:
                for field in ("renderer_id", "render_id"):
                    if getattr(unit.render, field) is None:
                        pending.append(Prerequisite(path=f"ordinary_units.{index}.render.{field}", code="NATIVE_RENDER_FACT_REQUIRED"))
        for field in ("actual_cache_binding", "intended_runtime_binding"):
            if getattr(proposal.resources, field) is None:
                pending.append(Prerequisite(path=f"resources.{field}", code="RUNTIME_RESOURCE_BINDING_REQUIRED"))
        for field in ("subject", "groups", "renders", "subject.operations", "subject.runtime_registration",
                      "subject.binding.source_files", "subject.children", "subject.embedding_cache"):
            pending.append(Prerequisite(path=f"preparation.{field}", code="TYPED_PREPARATION_METADATA_REQUIRED"))
    else:
        validate_metadata(proposal, pin, metadata)
    return PreparedProposal(status="PENDING_PREREQUISITES" if pending else "PREPARED_NOT_ADMITTED",
        proposal=pin, schedule=proposal.ordinary_units, prerequisites=tuple(pending),
        subject=None if metadata is None else metadata.subject)


def validate_metadata(proposal: Task11Proposal, pin: FilePin, metadata: PreparationMetadata) -> None:
    if metadata.proposal != pin:
        raise LiveDiagnosticError("DIAGNOSTIC_PREPARATION_PROPOSAL_MISMATCH")
    subject = metadata.subject
    sources = {row.path: row.sha256 for row in subject.binding.source_files}
    witnesses = [pin, *(row.evidence for row in metadata.groups), *(row.state_binding for row in metadata.groups),
                 *(row.evidence for row in metadata.renders), *(item.inventory_binding for item in proposal.inputs),
                 proposal.dc_contract.family,
                 *(unit.render.family_binding for unit in proposal.ordinary_units if unit.render is not None)]
    if any(sources.get(row.path) != row.sha256 for row in witnesses):
        raise LiveDiagnosticError("DIAGNOSTIC_PREPARATION_EVIDENCE_UNBOUND")
    for witness in witnesses:
        witness.read()
    if (tuple(row.group_id for row in metadata.groups) != tuple(row.group_id for row in proposal.checkpoint_groups)
        or len({row.unit_id for row in metadata.renders}) != len(metadata.renders)
        or any(row.unit_id not in {unit.unit_id for unit in proposal.ordinary_units if unit.render is not None}
               for row in metadata.renders)):
        raise LiveDiagnosticError("DIAGNOSTIC_PREPARATION_GROUP_MISMATCH")
    natives = tuple((scope, NativeInput.model_validate_json(scope.input_utf8)) for scope in subject.scopes)
    for group, facts in zip(proposal.checkpoint_groups, metadata.groups, strict=True):
        for field in ("prefix_id", "prefix_sample_ids", "checkpoint_id", "checkpoint_index", "state_binding"):
            frozen = getattr(group, field)
            if frozen is not None and frozen != getattr(facts, field):
                raise LiveDiagnosticError("DIAGNOSTIC_PREPARATION_FROZEN_FACT_MISMATCH")
        if group.state_sha256 is not None and group.state_sha256 != facts.state_binding.sha256:
            raise LiveDiagnosticError("DIAGNOSTIC_PREPARATION_FROZEN_FACT_MISMATCH")
    expected = {(row.task, row.baseline, row.arm, row.seed) for row in proposal.ordinary_units}
    if { (native.task_name, native.baseline, native.arm, native.trajectory_seed) for _, native in natives} != expected:
        raise LiveDiagnosticError("DIAGNOSTIC_PREPARATION_SCHEDULE_MISMATCH")
    for row in proposal.ordinary_units:
        scope, native = next((scope, native) for scope, native in natives
            if (native.task_name, native.baseline, native.arm) == (row.task, row.baseline, row.arm))
        item = next(item for item in proposal.inputs if item.task == row.task)
        facts = next(facts for facts in metadata.groups if facts.group_id == row.group_id)
        branch = native.branch
        if (native.tasks != (item.task_instance,) or native.model != proposal.resources.native_model
            or (branch.arm, branch.root_count, branch.prefix_identity, branch.source_identity, branch.checkpoint_index)
            != (row.arm, row.root_count, facts.prefix_id, facts.source_identity, facts.checkpoint_index)
            or (row.arm == "clean" and scope.state_sha256 != facts.state_binding.sha256)
            or row.sample_id in facts.prefix_sample_ids):
            raise LiveDiagnosticError("DIAGNOSTIC_PREPARATION_NATIVE_FACT_MISMATCH")
        if row.prefix_id is not None and row.prefix_id != facts.prefix_id:
            raise LiveDiagnosticError("DIAGNOSTIC_PREPARATION_FROZEN_FACT_MISMATCH")
        if row.checkpoint_id is not None and row.checkpoint_id != facts.checkpoint_id:
            raise LiveDiagnosticError("DIAGNOSTIC_PREPARATION_FROZEN_FACT_MISMATCH")
        if row.source_state_sha256 is not None and row.source_state_sha256 != facts.state_binding.sha256:
            raise LiveDiagnosticError("DIAGNOSTIC_PREPARATION_FROZEN_FACT_MISMATCH")
        if row.render is not None:
            render = row.render
            supplied = next((fact for fact in metadata.renders if fact.unit_id == row.unit_id), None)
            if supplied is not None and ((render.renderer_id is not None and supplied.renderer_id != render.renderer_id)
                                        or (render.render_id is not None and supplied.render_id != render.render_id)):
                raise LiveDiagnosticError("DIAGNOSTIC_PREPARATION_FROZEN_FACT_MISMATCH")
            if (render.render_id is None or render.renderer_id is None) and supplied is None:
                raise LiveDiagnosticError("DIAGNOSTIC_PREPARATION_RENDER_FACT_REQUIRED")
            render_id = render.render_id if render.render_id is not None else supplied.render_id if supplied else None
            if branch.native_render_id != render_id:
                raise LiveDiagnosticError("DIAGNOSTIC_PREPARATION_RENDER_MISMATCH")
            if render.triplet_id is not None and branch.candidate_triplet_id != render.triplet_id:
                raise LiveDiagnosticError("DIAGNOSTIC_PREPARATION_RENDER_MISMATCH")
            if row.baseline == "dc_rs":
                validate_dc_payload(row, native, scope)
    context = subject.verify()
    for resource in (proposal.resources.actual_cache_binding, proposal.resources.intended_runtime_binding):
        if resource is not None and resource not in context.contract.resource_files:
            raise LiveDiagnosticError("DIAGNOSTIC_PREPARATION_RESOURCE_MISMATCH")
