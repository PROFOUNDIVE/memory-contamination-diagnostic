from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from memcontam.config.phase12 import (
    Phase12ConfigError,
    build_candidate_template_set,
    load_phase12_config,
    resolve_phase12_config,
)
from memcontam.contamination.phase12.registry import load_candidate_registry
from memcontam.contamination.phase12.renderers import RendererRegistry
from memcontam.experiment.phase12.branching import BranchSet, build_matched_branches
from memcontam.experiment.phase12.contracts import (
    MemoryArmExecutionKey,
    PrefixExecutionKey,
    PrefixTemplateSpec,
    RouteCandidateId,
    RunTemplateSpec,
)
from memcontam.experiment.phase12.planner import (
    PlanningError,
    build_conditional_call_scope_registry,
    generate_candidate_route_registries,
    validate_exploratory_activation,
    validate_route_selection,
)
from memcontam.experiment.phase12.contracts import (
    CodeMatrixPlan,
    ExploratoryActivationManifest,
    FidelityCertificate,
    MftManifest,
    PilotBManifest,
    RouteFeasibilityReport,
    RouteSelectionManifest,
    SeedAllocationManifest,
    SelectedPackageResourceManifest,
)
from memcontam.experiment.phase12.prefix_runner import (
    PrefixEventLedger,
    PrefixRunSpec,
    PrefixStep,
    PrefixTask,
    run_clean_prefix,
)
from memcontam.experiment.phase12.suffix_runner import (
    SuffixEventLedger,
    SuffixStep,
    SuffixWriterFactory,
    run_matched_suffix,
)
from memcontam.logging.schema_v3 import (
    BaseSensitivityCellRef,
    CheckpointEvent,
    MemoryArmExecutionKey as LogMemoryArmExecutionKey,
    PreRouteRunMetadata,
    PrefixExecutionKey as LogPrefixExecutionKey,
    parse_log_record_v3,
)
from memcontam.logging.writer_v3 import Phase12RunWriter
from memcontam.memory.admission import AdmissionContext
from memcontam.memory.cards_v3 import MEMORY_CARD_V3, MemoryCardEnvelopeV3, canonical_content_hash
from memcontam.memory.checkpoint_v3 import NativeState
from memcontam.manifests.archive_validation import ArchiveValidationReport, validate_archive
from memcontam.readiness.scientific_admission import (
    AdmissionDenied,
    CertificateBundle,
    ScientificRunRequest,
    evaluate_scientific_admission,
)
from memcontam.tasks.base import TaskInstance


_CANDIDATES = {"3w", "5w"}
_PUBLIC_STREAMS = (
    "trials.jsonl",
    "calls.jsonl",
    "tool_events.jsonl",
    "retrieval_events.jsonl",
    "context_events.jsonl",
    "failures.jsonl",
    "memory_events.jsonl",
    "admission_events.jsonl",
    "intervention_events.jsonl",
    "checkpoint_events.jsonl",
    "eligibility_events.jsonl",
)
_WRITERS = {
    "fh_bounded": ("full_history_transcript", "fh_appender", "full_history_generate", "history"),
    "rag_frozen": ("rag_document", "rag_corpus_loader", "rag_corpus_load", "corpus"),
    "bot_style": ("thought_template", "bot_buffer_manager", "bot_thought_distill", "buffer"),
    "reflexion_style": (
        "verbal_reflection",
        "reflexion_reflector",
        "reflexion_reflect",
        "reflections",
    ),
}


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    phase12 = subparsers.add_parser("phase12")
    commands = phase12.add_subparsers(dest="phase12_command", required=True)

    validate = commands.add_parser("validate")
    validate.add_argument("--config", type=Path, required=True)

    plan = commands.add_parser("plan")
    plan.add_argument("--config", type=Path, required=True)

    for name in ("run-prefix", "run-branch"):
        run = commands.add_parser(name)
        run.add_argument("--replay")
        run.add_argument("--fixture-root", type=Path, default=_default_fixture_root())
        run.add_argument("--run-root", type=Path, default=Path("runs"))
        run.add_argument("--run-id", default="phase12-replay")
        run.add_argument("--candidate", default="3w")
        run.add_argument("--mode", default="text_only")
        run.add_argument("--run-family", default="readiness")
        run.add_argument("--scientific", action="store_true")
        run.add_argument("--scientific-result", choices=("true", "false"), default="false")
        run.add_argument("--admission-bundle", type=Path)
        run.add_argument("--admission-only", action="store_true")

    aggregate = commands.add_parser("aggregate")
    _add_replay_or_run_dir(aggregate)

    archive = commands.add_parser("validate-archive")
    _add_replay_or_run_dir(archive)


def run(args: argparse.Namespace) -> None:
    if args.phase12_command == "validate":
        _validate_config(args.config)
        print(f"valid phase12 config: {args.config}")
    elif args.phase12_command == "plan":
        print(json.dumps(_plan(args.config), sort_keys=True))
    elif args.phase12_command == "run-prefix":
        decision = _validate_run_request(args)
        result = _admission_result(decision) if args.admission_only else _run_prefix(args)
        print(json.dumps(result, sort_keys=True))
    elif args.phase12_command == "run-branch":
        decision = _validate_run_request(args)
        result = _admission_result(decision) if args.admission_only else _run_branch(args)
        print(json.dumps(result, sort_keys=True))
    elif args.phase12_command == "aggregate":
        print(json.dumps(_aggregate(args), sort_keys=True))
    elif args.phase12_command == "validate-archive":
        print(json.dumps(_validate_archive(args), sort_keys=True))
    else:
        raise SystemExit(f"unsupported phase12 command: {args.phase12_command}")


def _add_replay_or_run_dir(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--replay")
    source.add_argument("--run-dir", type=Path)
    parser.add_argument("--fixture-root", type=Path, default=_default_fixture_root())


def _default_fixture_root() -> Path:
    return Path(__file__).resolve().parents[4] / "tests" / "fixtures" / "phase12"


def _validate_config(path: Path) -> Any:
    try:
        return resolve_phase12_config(load_phase12_config(path))
    except Phase12ConfigError as error:
        raise SystemExit(str(error)) from error


def _plan(path: Path) -> dict[str, Any]:
    config = _validate_config(path)
    candidates: tuple[RouteCandidateId, ...] = ("3w", "5w")
    template_sets = tuple(
        build_candidate_template_set(config, candidate) for candidate in candidates
    )
    scopes = build_conditional_call_scope_registry(template_sets, frozen_at="non-scientific-replay")
    registries = generate_candidate_route_registries(template_sets, scopes)
    return {
        "candidate_routes": [template_set.candidate_route for template_set in template_sets],
        "registry_ids": [registry.registry_id for registry in registries],
        "scientific_result": False,
    }


def _validate_run_request(args: argparse.Namespace):
    scientific_result = args.scientific or args.scientific_result == "true"
    if args.replay and scientific_result:
        raise SystemExit("phase12 readiness gate not activated")
    if args.mode != "text_only" and not args.admission_only:
        raise SystemExit(f"unsupported phase12 mode: {args.mode}")
    if args.candidate not in _CANDIDATES:
        raise SystemExit(f"unsupported phase12 candidate: {args.candidate}")
    if not scientific_result and args.admission_bundle is not None:
        raise SystemExit("ADMISSION_EVIDENCE_FORBIDDEN")
    if not scientific_result and args.mode == "text_only":
        return None
    try:
        request, certificates, archive, route, activation = _load_admission_evidence(
            args, scientific_result
        )
        return evaluate_scientific_admission(request, certificates, archive, route, activation)
    except AdmissionDenied as error:
        raise SystemExit(error.code) from error


def _admission_result(decision: Any) -> dict[str, Any]:
    return {
        "admitted": decision is not None,
        "scientific_admission_ref": None if decision is None else decision.scientific_admission_ref,
    }


def _load_admission_evidence(
    args: argparse.Namespace, scientific_result: bool
) -> tuple[Any, CertificateBundle, ArchiveValidationReport, Any, Any]:
    if args.admission_bundle is None:
        if scientific_result:
            raise SystemExit("SCIENTIFIC_ADMISSION_REQUIRED")
        return (
            ScientificRunRequest(args.run_family, args.candidate, args.mode, False, None, None),
            CertificateBundle.empty(),
            ArchiveValidationReport(True, 0),
            None,
            None,
        )
    try:
        from memcontam.readiness.phase12_certificate import load_p12i

        payload = json.loads(args.admission_bundle.read_text(encoding="utf-8"))
        request = ScientificRunRequest(
            args.run_family,
            args.candidate,
            args.mode,
            scientific_result,
            payload.get("trajectory_seed"),
            payload.get("abstract_seed_slot"),
            _manifest_id(payload, "route_selection_manifest"),
            _manifest_id(payload, "seed_allocation_manifest"),
            _manifest_id(payload, "exploratory_activation_manifest"),
        )
        certificates = CertificateBundle(
            FidelityCertificate.model_validate(payload["bfv2_certificate"]),
            load_p12i(payload["p12i_certificate"]),
            payload["p12i_artifacts"],
        )
        archive_root = payload.get("archive_root")
        archive = (
            validate_archive(Path(archive_root))
            if isinstance(archive_root, str) and archive_root
            else ArchiveValidationReport(False, 0)
        )
        route = _validated_route(payload)
        activation = _validated_activation(payload, route)
        return request, certificates, archive, route, activation
    except AdmissionDenied:
        raise
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise SystemExit("ADMISSION_EVIDENCE_INVALID") from error


def _manifest_id(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    return value.get("manifest_id") if isinstance(value, dict) else None


def _validated_route(payload: dict[str, Any]):
    selection_payload = payload.get("route_selection_manifest")
    allocation_payload = payload.get("seed_allocation_manifest")
    if selection_payload is None or allocation_payload is None:
        return None
    try:
        return validate_route_selection(
            tuple(
                RouteFeasibilityReport.model_validate(item)
                for item in payload["feasibility_reports"]
            ),
            PilotBManifest.model_validate(payload["pilot_b_manifest"]),
            MftManifest.model_validate(payload["mft_manifest"]),
            RouteSelectionManifest.model_validate(selection_payload),
            SeedAllocationManifest.model_validate(allocation_payload),
        )
    except PlanningError as error:
        raise AdmissionDenied(error.code) from error


def _validated_activation(payload: dict[str, Any], route: Any):
    activation_payload = payload.get("exploratory_activation_manifest")
    if activation_payload is None:
        return None
    if route is None:
        return None
    try:
        plan = CodeMatrixPlan.model_validate(payload["exploratory_plan"])
        resource_payload = payload.get("selected_package_resource_manifest")
        if (
            not isinstance(resource_payload, dict)
            or resource_payload.get("mandatory_package_status") != "fully_resourced"
        ):
            raise AdmissionDenied("EXPLORATORY_RESOURCE_RESERVATION_NOT_PASS")
        resource = SelectedPackageResourceManifest.model_validate(resource_payload)
        activation = ExploratoryActivationManifest.model_validate(activation_payload)
        if resource.mandatory_package_status != "fully_resourced":
            raise AdmissionDenied("EXPLORATORY_RESOURCE_RESERVATION_NOT_PASS")
        if plan.estimated_exploratory_calls > resource.exploratory_call_budget:
            raise AdmissionDenied("EXPLORATORY_BUDGET_INSUFFICIENT")
        if (
            resource.exploratory_call_budget + resource.reproducibility_reserve
            > resource.remaining_call_capacity
        ):
            raise AdmissionDenied("REPRODUCIBILITY_RESERVE_INSUFFICIENT")
        return validate_exploratory_activation(plan, resource, activation, route)
    except PlanningError as error:
        raise AdmissionDenied(error.code) from error


def _run_prefix(args: argparse.Namespace) -> dict[str, Any]:
    fixture = _load_replay_fixture(args.fixture_root, args.replay)
    result = _build_prefix(fixture)
    writer = _open_writer(args.run_root / args.run_id, fixture, prefix=True)
    _write_prefix_result(writer, result)
    writer.finalize()
    _write_sidecars(args.run_root / args.run_id, fixture, "run-prefix")
    return {"prefix_run_id": result.prefix_run_id, "run_dir": str(args.run_root / args.run_id)}


def _run_branch(args: argparse.Namespace) -> dict[str, Any]:
    fixture = _load_replay_fixture(args.fixture_root, args.replay)
    prefix = _build_prefix(fixture)
    baseline = prefix.checkpoint.state.baseline
    registry_path = (
        Path(__file__).resolve().parents[4]
        / "data"
        / "phase12"
        / "registries"
        / "candidate_registry_v1.json"
    )
    branches = build_matched_branches(
        prefix.checkpoint,
        load_candidate_registry(registry_path).triplets[0],
        RendererRegistry.native(),
        _admission_context(baseline, prefix.checkpoint.state.entries),
    )
    suffix = _suffix_tasks(fixture)
    spec = _suffix_spec(baseline)
    factory = SuffixWriterFactory(
        {
            arm: _ReplaySuffixPolicy()
            for arm in ("clean", "correct", "irrelevant", "contam", "filter")
        }
    )
    suffix_result = run_matched_suffix(branches, suffix, spec, factory, seed=fixture["seed"])
    writer = _open_writer(args.run_root / args.run_id, fixture, prefix=False)
    _write_prefix_result(writer, prefix)
    _write_suffix_result(writer, suffix_result, factory)
    if isinstance(branches, BranchSet):
        for label in branches.audit_labels:
            writer.append_audit_label(asdict(label))
    writer.finalize()
    _write_sidecars(args.run_root / args.run_id, fixture, "run-branch")
    return {"pair_id": suffix_result.pair_id, "run_dir": str(args.run_root / args.run_id)}


def _build_prefix(fixture: dict[str, Any]):
    baseline = "fh_bounded"
    checkpoint = fixture["baseline_prefixes"][baseline]["checkpoint"]
    template = PrefixTemplateSpec(
        prefix_template_key=f"replay:{baseline}",
        execution_key=PrefixExecutionKey(kind="branch_free_prefix"),
        model_snapshot="replay",
        evidence_layer="build",
        task_family="phase12-replay",
        baseline_condition_id=baseline,
        sensitivity_cell_ref={"kind": "base", "cell_id": "base"},
        prompt_version="replay",
        tool_contract_hash="replay",
        corpus_version="replay",
        capacity_contract_id="replay",
        artifact_hash="replay",
    )
    spec = PrefixRunSpec(
        template=template,
        tasks=tuple(
            PrefixTask(row["absolute_trial_index"], row["task_id"], str(row["input"]))
            for row in fixture["burn_in_task_sequence"]
        ),
    )
    return run_clean_prefix(
        spec,
        seed=fixture["seed"],
        policy=_ReplayPrefixPolicy(checkpoint),
        writer=PrefixEventLedger(),
    )


class _ReplayPrefixPolicy:
    def __init__(self, checkpoint: dict[str, Any]) -> None:
        self._checkpoint = checkpoint

    def initial_state(self, spec: PrefixRunSpec, seed: int) -> NativeState:
        del spec, seed
        return NativeState(self._checkpoint["baseline"], (), self._checkpoint["native_state"])

    def execute(self, task: PrefixTask, state: NativeState, seed: int, trial_id: str) -> PrefixStep:
        del seed, trial_id
        return PrefixStep(
            NativeState(
                state.baseline,
                tuple(self._checkpoint["entries"][: task.absolute_trial_index]),
                state.native_state,
            )
        )


class _ReplaySuffixPolicy:
    def execute(
        self, task: TaskInstance, state: NativeState, seed: int, trial_id: str
    ) -> SuffixStep:
        del task, seed, trial_id
        return SuffixStep(state)


def _admission_context(baseline: str, entries: tuple[str | Any, ...]) -> AdmissionContext:
    semantic_kind, writer_id, writer_stage, native_component = _WRITERS[baseline]
    envelopes = tuple(
        MemoryCardEnvelopeV3(
            entry_id=str(entry),
            baseline=baseline,
            semantic_kind=semantic_kind,
            schema_version=MEMORY_CARD_V3,
            writer_id=writer_id,
            writer_event_id=f"replay-{entry}",
            writer_stage=writer_stage,
            created_trial_id=None if baseline == "rag_frozen" else f"trial-{entry}",
            source_trial_ids=() if baseline == "rag_frozen" else (f"trial-{entry}",),
            source_outcome=None,
            trial_support_ids=() if baseline == "rag_frozen" else (f"trial-{entry}",),
            memory_support_ids=(),
            direct_parent_ids=(),
            version_predecessor_id=None,
            order_key=index,
            native_component=native_component,
            content=f"replay {entry}",
            content_hash=canonical_content_hash(f"replay {entry}"),
        )
        for index, entry in enumerate(entries, start=1)
    )
    return AdmissionContext(
        writer_event_ids=frozenset(envelope.writer_event_id for envelope in envelopes),
        trial_record_ids=frozenset(
            trial_id for envelope in envelopes for trial_id in envelope.trial_support_ids
        ),
        evidence_envelopes=envelopes,
    )


def _suffix_tasks(fixture: dict[str, Any]) -> tuple[TaskInstance, ...]:
    checkpoint_index = fixture["selected_checkpoint"]["checkpoint_index"]
    return tuple(
        TaskInstance(
            sample_id=row["task_id"],
            task_name="phase12_replay",
            input={"replay_input": row["input"]},
            metadata={
                "absolute_trial_index": checkpoint_index + index,
                "event_time": row["event_time"],
            },
        )
        for index, row in enumerate(fixture["suffix"], start=1)
    )


def _suffix_spec(baseline: str) -> RunTemplateSpec:
    return RunTemplateSpec(
        run_template_id="phase12-replay",
        layer="core",
        population_layer="core",
        run_family="readiness",
        analysis_status="primary",
        model_snapshot="replay",
        evidence_layer="build",
        task_family="phase12-replay",
        baseline_condition_id=baseline,
        execution_key=MemoryArmExecutionKey(kind="memory_arm", arm="clean"),
        sensitivity_cell_ref={"kind": "base", "cell_id": "base"},
        contamination_type="core",
        horizon=1,
        prefix_template_key_or_none=f"replay:{baseline}",
        candidate_and_control_ids=("replay",),
        corpus_index_filter_versions={"corpus": "replay"},
        prompt_version="replay",
        tool_contract_hash="replay",
        artifact_hash="replay",
    )


def _open_writer(run_dir: Path, fixture: dict[str, Any], *, prefix: bool) -> Phase12RunWriter:
    execution_key = (
        LogPrefixExecutionKey(kind="branch_free_prefix")
        if prefix
        else LogMemoryArmExecutionKey(kind="memory_arm", arm="clean")
    )
    metadata = PreRouteRunMetadata(
        protocol_version="phase12_primary_v1",
        evidence_layer="build",
        run_family="readiness",
        run_template_id="phase12-replay",
        prefix_template_key_or_none="replay:fh_bounded",
        task_family="phase12-replay",
        baseline_condition_id="fh_bounded",
        execution_key=execution_key,
        protocol_index_or_none=None if prefix else "clean",
        trajectory_seed=fixture["seed"],
        abstract_seed_slot_or_none=None,
        sensitivity_cell_ref=BaseSensitivityCellRef(kind="base", cell_id="base"),
        metric_registry_version="replay",
        embedding_contract_hash="replay",
        tool_contract_hash="replay",
        candidate_registry_version="replay",
        split_manifest_version="replay",
        behavior_registry_version="replay",
        run_template_registry_version="replay",
        rerun_policy_version="replay",
        metadata_kind="pre_route",
        scientific_result=False,
        scientific_admission_ref_or_none=None,
    )
    return Phase12RunWriter.open(run_dir, metadata)


def _write_prefix_result(writer: Phase12RunWriter, result: Any) -> None:
    trial_ids = {event.checkpoint_index: event.trial_id for event in result.checkpoint_events}
    for trial in result.trials:
        trial_id = trial_ids[trial.absolute_trial_index]
        assert isinstance(trial_id, str)
        writer.append_trial(trial_id, trial)
    for event in (*result.admission_events, *result.checkpoint_events):
        writer.append_event(event.model_copy(update={"run_id": writer.run_dir.name}))


def _write_suffix_result(
    writer: Phase12RunWriter, result: Any, factory: SuffixWriterFactory
) -> None:
    for run in result.runs:
        ledger = factory._writers[run.arm]
        assert isinstance(ledger, SuffixEventLedger)
        checkpoint_events = [event for event in ledger.events if isinstance(event, CheckpointEvent)]
        for trial in run.trials:
            event = next(
                event
                for event in checkpoint_events
                if event.checkpoint_index == trial.absolute_trial_index
            )
            assert isinstance(event.trial_id, str)
            writer.append_trial(event.trial_id, trial)
    for arm in ("clean", "correct", "irrelevant", "contam", "filter"):
        ledger = factory._writers[arm]
        assert isinstance(ledger, SuffixEventLedger)
        for ledger_event in ledger.events:
            writer.append_event(ledger_event.model_copy(update={"run_id": writer.run_dir.name}))


def _write_sidecars(run_dir: Path, fixture: dict[str, Any], command: str) -> None:
    (run_dir / "resolved_config.json").write_text(
        json.dumps(
            {"command": command, "fixture_id": fixture["fixture_id"], "scientific_result": False},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (run_dir / "provider_profile.json").write_text(
        json.dumps({"provider": "replay", "provider_profile_id": "phase12-local"}, sort_keys=True),
        encoding="utf-8",
    )


def _aggregate(args: argparse.Namespace) -> dict[str, Any]:
    if args.replay:
        fixture = _load_replay_fixture(args.fixture_root, args.replay)
        accuracy = fixture["seed_accuracy"]
        averages = {
            arm: sum(seed[arm] for seed in accuracy.values()) / len(accuracy)
            for arm in accuracy["s1"]
        }
        return {
            "clean_minus_contam": averages["clean"] - averages["contam"],
            "clean_minus_filter": averages["clean"] - averages["filter"],
            "correct_minus_contam": averages["correct"] - averages["contam"],
            "filter_minus_contam": averages["filter"] - averages["contam"],
            "irrelevant_minus_contam": averages["irrelevant"] - averages["contam"],
        }
    assert args.run_dir is not None
    rows = _jsonl(args.run_dir / "trials.jsonl")
    return {"run_dir": str(args.run_dir), "trial_count": len(rows)}


def _validate_archive(args: argparse.Namespace) -> dict[str, Any]:
    if args.replay:
        fixture = _load_replay_fixture(args.fixture_root, args.replay)
        expected = fixture.get("expected", {})
        return {
            "archive_valid": expected.get("archive_valid") is True,
            "resolved_edges": expected.get("resolved_edges", 0),
        }
    assert args.run_dir is not None
    manifest_path = args.run_dir / "public_artifact_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"phase12 archive manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise SystemExit("phase12 archive is not completed")
    for filename, artifact in manifest.get("artifacts", {}).items():
        path = args.run_dir / filename
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != artifact["sha256"]:
            raise SystemExit(f"phase12 archive hash mismatch: {filename}")
        if filename.endswith(".jsonl") and len(_jsonl(path)) != artifact["count"]:
            raise SystemExit(f"phase12 archive count mismatch: {filename}")
    parse_log_record_v3(
        json.loads((args.run_dir / "run.json").read_text(encoding="utf-8"))["run_metadata"]
    )
    for filename in _PUBLIC_STREAMS:
        for row in _jsonl(args.run_dir / filename):
            if filename != "calls.jsonl" and filename != "memory_events.jsonl":
                parse_log_record_v3({key: value for key, value in row.items() if key != "trial_id"})
    return {"archive_valid": True, "run_dir": str(args.run_dir)}


def _load_replay_fixture(root: Path, fixture_id: str) -> dict[str, Any]:
    fixtures: dict[str, dict[str, Any]] = {}
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for filename in manifest["files"]:
        payload = json.loads((root / filename).read_text(encoding="utf-8"))
        if isinstance(payload.get("fixture_id"), str):
            fixtures[payload["fixture_id"]] = payload
    try:
        fixture = fixtures[fixture_id]
    except KeyError as error:
        raise SystemExit(f"unknown phase12 replay fixture: {fixture_id}") from error
    for reference in fixture.get("compose", []):
        if reference not in fixtures:
            raise SystemExit(f"phase12 replay fixture reference missing: {reference}")
    return fixture


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
