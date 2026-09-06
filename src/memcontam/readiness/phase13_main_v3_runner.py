from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

from memcontam.memory.checkpoint_v3 import NativeState, Phase12Checkpoint, serialize_checkpoint

from .phase13_main_preloaded_resources import PreloadedMainResources
from .phase13_main_production import ProductionObject
from .phase13_main_production_backend import OrdinaryRuntimeRequest, _memory_baseline, _ordinary_arm
from .phase13_main_request_client import MainRequestClientV3
from .phase13_main_request_dispatch import (
    CompiledProvider,
    DispatchTechnicalFailureV3,
    ProductionRequestDispatcherV3,
    production_provider,
)
from .phase13_main_request_recovery import require_known_costs
from .phase13_v3_cost_actual import reconcile_actual
from .phase13_v3_entrypoint import EntrypointError, SelectedExecutionV3
from .phase13_v3_entrypoint_paths import PrivateLedger, private_ledger
from .phase13_v3_request import PackageBindingV3, ParentTrajectoryV3, RequestKeyV3, Stage
from .phase13_v3_terminal_ledger import TerminalLedgerV3
from .phase13_v3_terminal_models import LedgerBindingV3

STAGE_NAMES: dict[str, Stage] = dict(zip(("FH_generation", "RAG_generation", "BoT_problem_distillation", "BoT_solve",
    "BoT_thought_distillation", "Reflexion_actor_generation", "Reflexion_reflection", "DC_RS_generation",
    "DC_RS_writer_synthesis", "NoMem_generation"), ("full_history_generate", "rag_generate", "bot_problem_distill",
    "bot_instantiate_solve", "bot_thought_distill", "reflexion_generate", "reflexion_reflect", "dc_rs_generate",
    "dc_rs_synthesize", "no_memory_generate"), strict=True))


@dataclass(frozen=True, slots=True)
class V3RunStatus:
    session_state: str
    completed_count: int
    terminal_technical_missing_count: int
    pending_count: int
    provider_calls_issued: int


@dataclass(frozen=True, slots=True)
class V3MainRun:
    selected: SelectedExecutionV3
    private: PrivateLedger
    ledger: TerminalLedgerV3
    lease: ExitStack

    def close(self) -> None:
        self.lease.close()

    @classmethod
    def open(cls, selected: SelectedExecutionV3, directory: Path, *, create: bool) -> V3MainRun:
        with ExitStack() as lease:
            lease.callback(selected.close)
            selected.preflight(selected.repository_root)
            private = lease.enter_context(private_ledger(directory, create=create))
            unit_ids: list[str] = []
            for unit in selected.costs.resources.phase4.base.units:
                ordinals: dict[Stage, int] = {}
                for group in unit.stages:
                    stage = STAGE_NAMES[group.stage_id]
                    start = ordinals.get(stage, 0)
                    unit_ids.extend(RequestKeyV3(parent_id=unit.unit_id, stage=stage, ordinal=ordinal).dispatch_id
                                    for ordinal in range(start, start + group.calls))
                    ordinals[stage] = start + group.calls
            binding = LedgerBindingV3(schema_version="phase13_main_run_ledger_v3", unit_ids=tuple(unit_ids),
                package_sha256=selected.package_sha256, authorization_sha256=selected.authorization_sha256)
            ledger = (TerminalLedgerV3.create_guarded(private, binding.model_dump(mode="json")) if create
                      else TerminalLedgerV3.open_guarded(private, binding))
            with private.connect() as connection:
                if create:
                    connection.execute("CREATE TABLE parents (unit_id TEXT PRIMARY KEY, raw BLOB, sha256 TEXT)")
                    connection.executemany("INSERT INTO parents VALUES (?, NULL, NULL)",
                        ((unit.unit_id,) for unit in selected.package.production))
                if {row[0] for row in connection.execute("SELECT unit_id FROM parents")} != set(selected.package.final_order.unit_ids):
                    raise EntrypointError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
            return cls(selected, private, ledger, lease.pop_all())

    def dispatcher(self, factory: Callable[[PackageBindingV3], CompiledProvider] = production_provider) -> ProductionRequestDispatcherV3:
        def checked_factory(binding: PackageBindingV3) -> CompiledProvider:
            self.selected.preflight(self.selected.repository_root)
            self.private.check()
            return factory(binding)
        return ProductionRequestDispatcherV3(self.ledger,
            PackageBindingV3(package_sha256=self.selected.package_sha256, authorization_sha256=self.selected.authorization_sha256),
            tuple(ParentTrajectoryV3(parent_id=unit.unit_id, kind=unit.kind, prefix_parent_id=unit.prefix_unit_id)
                  for unit in self.selected.package.production), provider_factory=checked_factory)

    def status(self) -> V3RunStatus:
        failed = self.dispatcher().terminal_parents
        with self.private.connect() as connection:
            for unit_id, raw, checksum in connection.execute("SELECT * FROM parents WHERE raw IS NOT NULL"):
                if hashlib.sha256(raw).hexdigest() != checksum or json.loads(raw)["unit_id"] != unit_id:
                    raise EntrypointError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
            completed = connection.execute("SELECT count(*) FROM parents WHERE raw IS NOT NULL").fetchone()[0]
        attempts = sum(json.loads(raw)["kind"] == "ATTEMPT_STARTED" for raw in self.ledger.rows())
        pending = len(self.selected.package.production) - completed - len(failed)
        return V3RunStatus("COMPLETED" if pending == 0 else "READY", completed, len(failed), pending, attempts)

    def execute(self, cache: Path, *, max_units: int | None, tranche_ceiling_krw: int,
                provider_factory: Callable[[PackageBindingV3], CompiledProvider] = production_provider) -> V3RunStatus:
        from .phase13_main_live_runtime import ProductionMainRuntime

        dispatcher = self.dispatcher(provider_factory)
        self.status()
        dispatcher.recover()
        attempted = 0
        projected = 0
        for unit in self.selected.package.production:
            with self.private.connect() as connection:
                complete = connection.execute("SELECT raw FROM parents WHERE unit_id=?", (unit.unit_id,)).fetchone()[0]
            if complete is not None or unit.unit_id in dispatcher.terminal_parents:
                continue
            if max_units is not None and attempted >= max_units:
                break
            require_known_costs(self.ledger)
            for cost_unit in self.selected.costs.resources.phase4.base.units:
                if cost_unit.unit_id != unit.unit_id:
                    continue
                for group in cost_unit.stages:
                    for ordinal in range(group.calls):
                        key = RequestKeyV3(parent_id=unit.unit_id, stage=STAGE_NAMES[group.stage_id], ordinal=ordinal)
                        if self.ledger.state(key.dispatch_id).kind == "COMPLETED":
                            raise EntrypointError("MAIN_RUN_IN_FLIGHT_RECONCILIATION_REQUIRED")
            realized = sum(reconcile_actual(state.attempted_cost).realized_krw
                           for state in self.ledger.states().values() if state.attempted_cost is not None)
            if (projected + unit.projected_cost_krw > tranche_ceiling_krw
                or realized + unit.projected_cost_krw > 450000):
                break
            client = MainRequestClientV3(dispatcher, unit.unit_id, lambda: self.selected.preflight(self.selected.repository_root))
            runtime = ProductionMainRuntime(self.selected.repository_root, cache, client=client,
                                            resources=PreloadedMainResources(self.selected))
            runtime.preflight((unit,))
            try:
                checkpoint = None
                if unit.kind == "CLEAN_PREFIX":
                    output = runtime.execute_prefix(unit)
                    checkpoint = output.checkpoint.canonical_bytes.decode()
                    dispatch = output.dispatch
                else:
                    dispatch = runtime.execute_ordinary(OrdinaryRuntimeRequest(unit,
                        "nomem" if unit.memory_baseline is None else _memory_baseline(unit.memory_baseline),
                        "clean" if unit.memory_baseline is None else _ordinary_arm(unit.arm), unit.arm,
                        unit.prefix_unit_id, self.checkpoint(unit)))
                raw = json.dumps({"unit_id": unit.unit_id, "checkpoint": checkpoint, "evidence": dispatch.evidence,
                    "provider_calls": [call.model_dump(mode="json") for call in dispatch.provider_calls],
                    "realized_cost_krw": dispatch.realized_cost_krw}, sort_keys=True, allow_nan=False).encode()
                with self.private.connect() as connection:
                    connection.execute("UPDATE parents SET raw=?, sha256=? WHERE unit_id=? AND raw IS NULL",
                                       (raw, hashlib.sha256(raw).hexdigest(), unit.unit_id))
            except DispatchTechnicalFailureV3:
                attempted += 1
                projected += unit.projected_cost_krw
                continue
            attempted += 1
            projected += unit.projected_cost_krw
        return self.status()

    def checkpoint(self, unit: ProductionObject) -> Phase12Checkpoint | None:
        if unit.prefix_unit_id is None:
            return None
        with self.private.connect() as connection:
            raw, expected = connection.execute("SELECT raw, sha256 FROM parents WHERE unit_id=?", (unit.prefix_unit_id,)).fetchone()
        if raw is None or hashlib.sha256(raw).hexdigest() != expected:
            raise EntrypointError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
        payload = json.loads(raw)
        if payload["unit_id"] != unit.prefix_unit_id or payload["checkpoint"] is None:
            raise EntrypointError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
        return serialize_checkpoint(NativeState.from_mapping(json.loads(payload["checkpoint"])))
