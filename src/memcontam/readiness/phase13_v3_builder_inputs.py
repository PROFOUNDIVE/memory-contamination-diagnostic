from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Final, TypeVar

from pydantic import BaseModel, ValidationError

from .phase13_execution_contract import CORE_MAIN_REGISTRY
from .phase13_main_checkpoint import CommonCheckpointRegistry, TaskSeedOrders, _expected_registry, _canonical_hash
from .phase13_main_production import ProductionObject, _object, _execution_template_id, _stages
from .phase13_main_resource_contract import RESOURCE_PATHS
from .phase13_v3_builder_models import FirstFreeze, MRP4Manifest
from .phase13_v3_cost import activate_policy, build_witness, freeze_base
from .phase13_v3_cost_binding import MRP4Costs
from .phase13_v3_cost_models import CostUnit, PrefreezeBindings, StageOccurrences, canonical_bytes, digest
from .phase13_v3_publication import ArtifactError, OUTPUT_PATHS
from .phase13_v3_resource_files import ClosureError, FileBinding, read_files
from .phase13_v3_source_closure import ResourceClosure, _rows_hash

PREFIX: Final = "data/phase13/main/"
STATIC_PATHS: Final = tuple(sorted({path for role, path in RESOURCE_PATHS.items() if role not in {"activated_policy", "base_inputs", "cost_witness"}} | {
    "src/memcontam/readiness/data/mmlu_pro_dc_selection_v1.json",
    *(PREFIX + "mr_p4/corrected_v1/" + name for name in ("common_task_spec_contract_v1.json", "answer_payload_contract_v1.json", "game24_task_prompt_v1.txt", "meb_task_prompt_v1.txt", "word_sorting_task_prompt_v1.txt", "mmlu_task_prompt_v1.txt")),
}))
STAGES: Final = {
    "full_history_generate": "FH_generation", "rag_generate": "RAG_generation",
    "bot_problem_distill": "BoT_problem_distillation", "bot_instantiate_solve": "BoT_solve",
    "bot_thought_distill": "BoT_thought_distillation", "reflexion_generate": "Reflexion_actor_generation",
    "reflexion_reflect": "Reflexion_reflection", "dc_rs_generate": "DC_RS_generation",
    "dc_rs_synthesize": "DC_RS_writer_synthesis", "no_memory_generate": "NoMem_generation",
}
ModelT = TypeVar("ModelT", bound=BaseModel)


def artifact_raw(output: Path, relative: str) -> bytes:
    try:
        row, = read_files(output, (relative,))
        return row.raw
    except ClosureError as error:
        cause = error.__cause__
        if isinstance(cause, FileNotFoundError) or (cause is not None and isinstance(cause.__cause__, FileNotFoundError)):
            raise ArtifactError("MAIN_PREDECESSOR_MISSING") from error
        raise


def parse_artifact(raw: bytes, model: type[ModelT]) -> ModelT:
    try:
        result = model.model_validate_json(raw)
    except ValidationError as error:
        raise ArtifactError("MAIN_ARTIFACT_BINDING_MISMATCH") from error
    if canonical_bytes(result) != raw:
        raise ArtifactError("MAIN_ARTIFACT_BINDING_MISMATCH")
    return result


def binding(path: str, raw: bytes) -> FileBinding:
    return FileBinding(path=path, size=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def resource_closure(rows: tuple[FileBinding, ...]) -> ResourceClosure:
    ordered = tuple(sorted(rows, key=lambda row: row.path.encode()))
    return ResourceClosure(rows=ordered, resource_closure_sha256=_rows_hash(ordered))


def first_freeze(repository: Path) -> FirstFreeze:
    resources = {row.binding.path: row for row in read_files(repository, STATIC_PATHS)}
    orders_raw = resources[RESOURCE_PATHS["task_seed_orders"]].raw
    orders = TaskSeedOrders.model_validate_json(orders_raw)
    registry = CommonCheckpointRegistry.model_validate_json(resources[RESOURCE_PATHS["common_checkpoint_registry"]].raw)
    expected = _expected_registry(orders.model_dump(mode="json", exclude={"orders_hash"}), hashlib.sha256(orders_raw).hexdigest())
    if (set(orders.tasks) != set(CORE_MAIN_REGISTRY.tasks) or orders.concrete_seed_ids != tuple(range(10))
        or orders.orders_hash != _canonical_hash(orders.model_dump(mode="json", exclude={"orders_hash"}))
        or registry.registry_hash != _canonical_hash(expected)
        or registry.model_dump(mode="json", exclude={"registry_hash"}) != expected):
        raise ArtifactError("MAIN_MR_P4_FIRST_FREEZE_MISMATCH")
    for task, task_orders in orders.tasks.items():
        source = resources.get(task_orders.source.path)
        if source is None or source.binding.sha256 != task_orders.source.sha256:
            raise ArtifactError("MAIN_MR_P4_FIRST_FREEZE_MISMATCH")
        if task in CORE_MAIN_REGISTRY.tasks[:3]:
            samples = tuple(json.loads(line)["sample_id"] for line in source.raw.splitlines())
        else:
            samples = tuple(f"{task}:{item}" for item in json.loads(source.raw)["tasks"][task]["question_ids"])
        if samples != task_orders.sample_ids or tuple(seed.seed for seed in task_orders.seeds) != tuple(range(10)):
            raise ArtifactError("MAIN_MR_P4_FIRST_FREEZE_MISMATCH")
        for seed in task_orders.seeds:
            order = tuple(sorted(samples, key=lambda sample: hashlib.sha256(f"sha256_task_seed_v1\0{task}\0{seed.seed}\0{sample}".encode()).digest()))
            if seed.ordered_sample_ids != order or seed.order_sha256 != hashlib.sha256(json.dumps(order, separators=(",", ":")).encode()).hexdigest():
                raise ArtifactError("MAIN_MR_P4_FIRST_FREEZE_MISMATCH")
    return FirstFreeze(concrete_seed_ids=tuple(range(10)), orders=orders, registry=registry, capacity=8192)


def production(first: FirstFreeze, resources: tuple[FileBinding, ...]) -> tuple[ProductionObject, ...]:
    by_path = {row.path: row.sha256 for row in resources}
    units: list[ProductionObject] = []
    arms = CORE_MAIN_REGISTRY.arms
    for seed in first.concrete_seed_ids:
        for task in CORE_MAIN_REGISTRY.tasks:
            for baseline in CORE_MAIN_REGISTRY.memory_baselines:
                if (task, baseline) in CORE_MAIN_REGISTRY.current_main_excluded_cells:
                    continue
                prefix = _object(len(units), "CLEAN_PREFIX", seed, task, baseline, "NOT_APPLICABLE", None)
                units.append(prefix)
                for arm in arms[seed % 4:] + arms[:seed % 4]:
                    units.append(_object(len(units), "MEMORY_BEARING", seed, task, baseline, arm, prefix.unit_id))
            units.append(_object(len(units), "NO_MEMORY_SINGLETON", seed, task, None, "NOT_APPLICABLE", None))
    return tuple(replace(unit, execution_template_id=_execution_template_id(unit),
                         ordered_sample_ids_sha256=first.registry.tasks[unit.task].seeds[unit.seed].suffix_sample_ids_sha256,
                         registration_packet_sha256=by_path[RESOURCE_PATHS["observability_packet"]],
                         checkpoint_registry_sha256=by_path[RESOURCE_PATHS["common_checkpoint_registry"]]) for unit in units)


def phase4_costs(manifest: MRP4Manifest) -> MRP4Costs:
    units = production(manifest.first_freeze, manifest.resources)
    cost_units = tuple(sorted((CostUnit(unit_id=unit.unit_id, stages=tuple(StageOccurrences(stage_id=STAGES[stage], calls=calls)
                       for stage, calls in _stages(unit))) for unit in units), key=lambda unit: unit.unit_id))
    governed = {row.path: row.sha256 for row in manifest.governed_source.rows}
    def source_hash(names: tuple[str, ...]) -> str:
        return hashlib.sha256("\0".join(governed[name] for name in names).encode()).hexdigest()
    runtime = digest(manifest.runtime_identity)
    bindings = PrefreezeBindings(package_cells_hash=hashlib.sha256("\0".join(sorted(unit.unit_id for unit in units)).encode()).hexdigest(),
        seed_checkpoint_hash=digest(manifest.first_freeze),
        prefix_ownership_hash=hashlib.sha256("\0".join(sorted(f"{unit.unit_id}:{unit.prefix_unit_id}" for unit in units)).encode()).hexdigest(),
        stage_occurrences_hash=hashlib.sha256(b"".join(canonical_bytes(unit) for unit in cost_units)).hexdigest(),
        governed_source_hash=manifest.governed_source.governed_tree_sha256, runtime_hash=runtime,
        request_compiler_hash=source_hash(("src/memcontam/readiness/phase13_v3_request.py", "src/memcontam/clients/openai_responses.py")),
        serializer_hash=source_hash(("src/memcontam/readiness/phase13_v3_cost_models.py",)),
        tokenizer_hash=hashlib.sha256((dict(manifest.runtime_identity.versions)["tiktoken"] + runtime).encode()).hexdigest())
    policy = activate_policy(manifest.authority)
    base = freeze_base(policy, bindings, cost_units)
    return MRP4Costs(policy=policy, base=base, witness=build_witness(base))


def generated_resources(repository: Path, output: Path, paths: tuple[str, ...]) -> ResourceClosure:
    rows: list[FileBinding] = []
    for path in sorted(paths):
        relative = path.removeprefix(PREFIX)
        if relative in OUTPUT_PATHS:
            rows.append(binding(path, artifact_raw(output, relative)))
        else:
            row, = read_files(repository, (path,))
            rows.append(row.binding)
    return resource_closure(tuple(rows))
