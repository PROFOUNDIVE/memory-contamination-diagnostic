from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Final

from pydantic import TypeAdapter

from .phase13_authority_files import authority_directory
from .phase13_main_checkpoint import ArtifactIdentity
from .phase13_v3_builder_inputs import PREFIX, artifact_raw
from .phase13_v3_publication import ArtifactError, OUTPUT_PATHS
from .phase13_v3_resource_files import read_files
from .phase13_v3_source_closure import _git

RECEIPTS: Final = {
    ".omo/evidence/phase13-initial-head.txt": "ad82ecc044e9e7456ee05fbca21b6135d5ffd2f67d28ce395ccc93b3eb49eb72",
    ".omo/evidence/phase13-initial-status.bin": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    ".omo/evidence/phase13-historical-baseline.json": "7b892e10d4e44cff4115ebf23810dbdfff4757910cbc6307c4f54be8ccc52ce0",
}
_BASELINE: Final[TypeAdapter[tuple[ArtifactIdentity, ...]]] = TypeAdapter(tuple[ArtifactIdentity, ...])
ALLOW: Final = (
    "src/memcontam/readiness/phase13_*.py", "src/memcontam/clients/openai_responses.py",
    "src/memcontam/experiment/phase13_ordinary_runtime.py", "src/memcontam/verifiers/math_equation_balancer.py",
    "scripts/build_phase13_corrected_main_closure.py", "scripts/diagnose_phase13_mr_p5_closure.py",
    "scripts/build_phase13_main_registries.py", "tests/phase13_v3_fixtures.py", "tests/test_phase13_*.py",
    "tests/test_bot_retrieval_decision.py", ".omo/evidence/*", *(PREFIX + name for name in OUTPUT_PATHS),
    "tests/test_baseline_execution_outcomes.py", "tests/test_baseline_policy_compatibility.py",
    "tests/test_baseline_source_contract_replay.py", "tests/test_cli_run.py",
    "tests/test_exact_lineage_contract.py", "tests/test_final_answer_parser.py",
    "tests/test_full_history_faithful.py", "tests/test_no_memory.py", "tests/test_phase12_end_to_end.py",
    "tests/test_phase12_filter_v5_authority_transition.py", "tests/test_phase12_filter_v5_bct_authorization.py",
    "tests/test_phase12_filter_v5_bct_waiting.py", "tests/test_phase12_filter_v5_evidence_security.py",
    "tests/test_phase12_filter_v5_final_verifier_modes.py", "tests/test_phase12_filter_v5_freeze_a.py",
    "tests/test_phase12_filter_v5_methods_lock.py", "tests/test_phase12_filter_v5_plan_digest.py",
    "tests/test_phase12_filter_v5_rootless_closure.py", "tests/test_phase12_filter_v5_rootless_execution.py",
    "tests/test_phase12_filter_v5_rootless_legacy_fence.py", "tests/test_phase12_filter_v5_rootless_live_manifests.py",
    "tests/test_phase12_filter_v5_rootless_process_races.py", "tests/test_phase12_filter_v5_rootless_task7_rehearsal.py",
    "tests/test_phase12_filter_v5_scope.py", "tests/test_phase12_filter_v5_terminal_semantics.py",
    "tests/test_phase12_integration_certificate.py", "tests/test_phase12_scientific_admission.py",
    "tests/test_gitignore_contract.py", "tests/test_rag_runner.py", "tests/test_reflexion_faithful.py",
    "tests/test_resolved_config.py", "tests/test_retrieval_rag.py", "tests/test_tool_augmented_no_memory.py",
    "tests/test_phase12_externalized_provenance.py",
    "tests/fixtures/phase12_externalized_provenance_registry_v1.json",
    "tests/fixtures/tracked_ignored_provenance_baseline_v1.json",
    "tests/fixtures/prompts/reflexion_generate.json", "tests/fixtures/prompts/reflexion_reflect.json",
    "tests/fixtures/prompts/rag_generate.json", "tests/fixtures/baseline_fidelity_v2_semantic_call_hashes.json",
    "data/phase13/common_capacity_corrected_v2.json",
    "data/phase13/common_capacity_status_corrected_v2.json",
    "data/phase13/observability/manifest_v1.json",
    "data/phase13/observability/fixture_v1.json",
    "data/phase13/observability/registration_packet_v1.json",
    "data/phase13/observability/registration_packet_v2.json",
    "data/phase13/main/legacy_dc_rs_intervention_registry_v1.json",
    "data/phase12/registries/candidate_registry_v2.json",
    "data/phase12/registries/hidden_audit_registry_v2.json",
    "data/phase13/main/legacy_dc_rs_intervention_registry_v2.json",
    "data/phase13/rag/legacy_seal_v2.json",
    "data/phase13/rag/legacy_v2/*",
    "src/memcontam/baselines/bot_phase12.py",
    "src/memcontam/baselines/bot_runtime.py",
    "src/memcontam/baselines/bot_write.py",
    "src/memcontam/baselines/dynamic_cheatsheet_phase12.py",
    "src/memcontam/baselines/reflexion_adapter.py",
    "src/memcontam/baselines/reflexion_phase12.py",
    "src/memcontam/baselines/retrieval_rag_phase12.py",
    "src/memcontam/contamination/phase12/renderers.py",
    "src/memcontam/contamination/phase12/certification.py",
    "src/memcontam/contamination/phase12/models.py",
    "src/memcontam/contamination/phase12/registry.py",
    "src/memcontam/contamination/phase13_legacy_dc_rs.py",
    "src/memcontam/contamination/phase13_v2_applicability.py",
    "src/memcontam/evaluation/phase12_observables.py",
    "src/memcontam/evaluation/phase13_observability.py",
    "src/memcontam/evaluation/phase13_observability_lineage.py",
    "src/memcontam/evaluation/phase13_observability_registration.py",
    "src/memcontam/evaluation/phase13_observability_sequence.py",
    "src/memcontam/experiment/phase12/branching.py",
    "src/memcontam/experiment/phase12/game24_runner.py",
    "src/memcontam/experiment/phase12/runtime_registry.py",
    "src/memcontam/experiment/phase13_dc_rs_runtime.py",
    "src/memcontam/experiment/phase13_dc_rs_validation.py",
    "src/memcontam/memory/checkpoint_v3.py",
    "tests/phase13_corrective_identity.py",
    "tests/phase13_runner_safety_fixture.py",
    "tests/test_phase12_bot.py",
    "tests/test_phase12_reflexion.py",
    "tests/test_bot_style.py",
    "tests/test_openai_responses_client.py",
    "tests/provider_denial/sitecustomize.py",
)


def audit_scope(repository: Path, output: Path, compare: Path | None) -> None:
    if compare is not None:
        for name in OUTPUT_PATHS:
            if artifact_raw(output, name) != artifact_raw(compare, name):
                raise ArtifactError("MAIN_ARTIFACT_BINDING_MISMATCH")
        actual = {path.relative_to(compare).as_posix() for path in compare.rglob("*") if not path.is_dir()}
        if actual != set(OUTPUT_PATHS):
            raise ArtifactError("MAIN_ARTIFACT_BINDING_MISMATCH")
        return
    receipts = {row.binding.path: row for row in read_files(repository, tuple(RECEIPTS))}
    if any(receipts[name].binding.sha256 != expected for name, expected in RECEIPTS.items()):
        raise ArtifactError("MAIN_ARTIFACT_BINDING_MISMATCH")
    baseline = _BASELINE.validate_json(receipts[".omo/evidence/phase13-historical-baseline.json"].raw)
    if len(baseline) != 147:
        raise ArtifactError("MAIN_ARTIFACT_BINDING_MISMATCH")
    current = {row.binding.path: row.binding.sha256 for row in read_files(repository, tuple(row.path for row in baseline))}
    if any(current[row.path] != row.sha256 for row in baseline):
        raise ArtifactError("MAIN_ARTIFACT_BINDING_MISMATCH")
    initial = receipts[".omo/evidence/phase13-initial-head.txt"].raw.decode().strip()
    with authority_directory(repository) as directory:
        diff = _git(directory, ("diff", "--name-status", "--no-renames", initial, "HEAD")).decode()
        for line in diff.splitlines():
            status, path = line.split("\t", 1)
            if status not in {"A", "M"} or not any(fnmatch.fnmatchcase(path, rule) for rule in ALLOW):
                raise ArtifactError("MAIN_GOVERNED_SOURCE_DRIFT")
        for filename in ("pyproject.toml", "requirements.lock", "requirements-dev.lock"):
            row, = read_files(repository, (filename,))
            if row.raw != _git(directory, ("show", f"{initial}:{filename}")):
                raise ArtifactError("MAIN_GOVERNED_SOURCE_DRIFT")
    baseline_names = {row.path.removeprefix(PREFIX) for row in baseline if row.path.startswith(PREFIX)}
    actual = {path.relative_to(output).as_posix() for path in output.rglob("*") if not path.is_dir()}
    if actual != baseline_names | set(OUTPUT_PATHS) | {
        "legacy_dc_rs_intervention_registry_v1.json",
        "legacy_dc_rs_intervention_registry_v2.json",
    }:
        raise ArtifactError("MAIN_ARTIFACT_BINDING_MISMATCH")
    read_files(output, tuple(actual))
