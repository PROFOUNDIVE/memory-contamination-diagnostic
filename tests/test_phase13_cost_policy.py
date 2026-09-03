from __future__ import annotations

import importlib
import importlib.util
import hashlib
import json
import shutil
from pathlib import Path
from types import ModuleType

import pytest

from memcontam.clients.base import LLMResponse


ROOT = Path(__file__).resolve().parents[1]


def _canonical_hash(payload: dict[str, object], field: str) -> str:
    canonical = {key: value for key, value in payload.items() if key != field}
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _copy_complete_policy_fixture(tmp_path: Path) -> Path:
    package = tmp_path / "data/phase13/main/cost_envelope_v2"
    package.parent.mkdir(parents=True)
    shutil.copytree(ROOT / "data/phase13/main/cost_envelope_v2", package)
    for relative in (
        Path("data/phase13/main/post_cutoff_package_selection_v2.json"),
        Path("data/phase13/common_capacity_v1.json"),
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    return package


def _policy_module() -> ModuleType:
    assert importlib.util.find_spec("memcontam.readiness.phase13_cost_policy") is not None, (
        "the approved cost policy belongs in memcontam.readiness.phase13_cost_policy"
    )
    return importlib.import_module("memcontam.readiness.phase13_cost_policy")


class _Client:
    def __init__(self, *, attempts: int = 1) -> None:
        self.attempts = attempts
        self.configs: list[dict[str, object]] = []

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        config: dict[str, object],
    ) -> LLMResponse:
        del messages, model
        self.configs.append(dict(config))
        return LLMResponse(
            content="{}",
            raw={"attempts": self.attempts, "cost_usd": 0.0},
            token_usage={"prompt_tokens": 1, "completion_tokens": 1},
            latency_ms=0,
        )


def test_approved_cost_policy_package_reconstructs_exact_bound() -> None:
    policy = _policy_module()

    report = policy.validate_cost_policy_package(ROOT)

    assert report.policy_id == "phase13_main_a_cost_envelope_v2_450k_w8192_a512_s384_t1"
    assert report.total_budget_ceiling_krw == 500_000
    assert report.reserve_fraction == "0.10"
    assert report.core_authorization_gate_krw == 450_000
    assert report.cmax_main_krw == 444_256
    assert report.margin_krw == 5_744
    assert report.writer_cap == 8192
    assert report.common_capacity_tokens == 8192
    assert report.maximum_transport_attempts == 1
    assert report.execution_envelope_registry_id == "CORE_EXECUTION_ENVELOPE_REGISTRY_V2"
    assert report.execution_envelope_registry_sha256 == (
        "4dec48f105c8d4730706d1d99d05bb14bab96a8e643811db1ebdd26e612590d5"
    )
    assert report.terminal_failure_contract_sha256 == (
        "9bbcdd9dd1686af034f7c0d2114ac86d5837a07de0cc6ba8fef7940bbc822b75"
    )
    assert report.cost_envelope_sha256 == (
        "6de377752cd80e45147a8b47aa83828f2921363b564c44004ac90650dac65cf2"
    )
    assert report.activation_status == "PENDING_CONTROLLED_EXTERNAL_AUTHORITY_WRITE"


def test_cost_policy_module_main_emits_validation_report(
    capsys: pytest.CaptureFixture[str],
) -> None:
    policy = _policy_module()

    policy.main()

    report = json.loads(capsys.readouterr().out)
    assert report["cmax_main_krw"] == 444_256
    assert report["activation_status"] == "PENDING_CONTROLLED_EXTERNAL_AUTHORITY_WRITE"


def test_cost_policy_exposes_case_b_one_owner_prefix_decomposition() -> None:
    policy = _policy_module()

    bundle = policy.load_cost_policy_bundle(ROOT)

    assert bundle.proof.reconciliation.case == "B"
    assert bundle.proof.reconciliation.prefix_ownership_instances == 230
    assert bundle.proof.reconciliation.prefix_semantic_calls == 430
    assert bundle.proof.semantic_calls == 108_930
    assert sum(stage.prefix_calls for stage in bundle.registry.stages) == 430
    assert all(stage.calls == stage.suffix_calls + stage.prefix_calls for stage in bundle.registry.stages)


def test_cost_proof_binds_the_same_package_selection_as_mr_p5() -> None:
    policy = _policy_module()
    bundle = policy.load_cost_policy_bundle(ROOT)
    package = json.loads(
        (ROOT / "data/phase13/main/mr_p5/execution_package_v1.json").read_text(
            encoding="utf-8"
        )
    )
    package_selection = next(
        artifact for artifact in package["artifacts"] if artifact["role"] == "package_selection"
    )

    assert bundle.proof.package_selection_path == package_selection["path"]
    assert bundle.proof.package_selection_sha256 == package_selection["sha256"]


def test_cost_policy_binds_stage_caps_and_single_transport_attempt() -> None:
    policy = _policy_module()
    client = _Client()
    bound = policy.CostPolicyClient(client, policy.load_cost_policy_bundle(ROOT))

    bound.chat(
        [{"role": "user", "content": "distill this task"}],
        "replay",
        {"method_stage": "bot_problem_distill", "max_output_tokens": 4096},
    )

    assert client.configs == [
        {
            "method_stage": "bot_problem_distill",
            "max_output_tokens": 384,
            "_phase13_maximum_input_tokens": 1177,
            "_phase13_execution_envelope_id": "CORE_EXECUTION_ENVELOPE_REGISTRY_V2",
            "_phase13_execution_envelope_sha256": (
                "58e1ebda33a63fba4cb5289d21531298a7803a765b3525214d45700bc993cc22"
            ),
            "_phase13_maximum_transport_attempts": 1,
            "_phase13_failure_contract_id": "CORE_TRANSPORT_ATTEMPT_CONTRACT_V2",
            "_phase13_failure_contract_sha256": (
                "1ee66fcb795f97d483c2ef976133ee61dbd5108c9dae851c2c2786ff496d788f"
            ),
            "_phase13_terminal_failure_contract_id": (
                "CORE_TERMINAL_TECHNICAL_MISSINGNESS_V1"
            ),
            "_phase13_terminal_failure_contract_sha256": (
                "9bbcdd9dd1686af034f7c0d2114ac86d5837a07de0cc6ba8fef7940bbc822b75"
            ),
            "_phase13_rate_card_sha256": (
                "50975b67dce4c59ba9267c3234a873076137ded5078aa3e8b5c9a2fad4ff3e06"
            ),
        }
    ]


def test_cost_policy_rejects_input_overflow_before_dispatch() -> None:
    policy = _policy_module()
    client = _Client()
    bound = policy.CostPolicyClient(client, policy.load_cost_policy_bundle(ROOT))

    with pytest.raises(policy.Phase13CostPolicyError, match="INPUT_ENVELOPE_EXCEEDED"):
        bound.chat(
            [{"role": "user", "content": "token " * 400}],
            "replay",
            {"method_stage": "rag_generate"},
        )

    assert client.configs == []


def test_cost_policy_rejects_more_than_one_observed_transport_attempt() -> None:
    policy = _policy_module()
    bound = policy.CostPolicyClient(_Client(attempts=2), policy.load_cost_policy_bundle(ROOT))

    with pytest.raises(policy.Phase13CostPolicyError, match="TRANSPORT_ATTEMPT_EXCEEDED"):
        bound.chat(
            [{"role": "user", "content": "solve"}],
            "replay",
            {"method_stage": "no_memory_generate"},
        )


def test_cost_policy_validator_rejects_tampered_cost_proof(tmp_path: Path) -> None:
    policy = _policy_module()
    package = tmp_path / "data/phase13/main/cost_envelope_v2"
    package.parent.mkdir(parents=True)
    shutil.copytree(ROOT / "data/phase13/main/cost_envelope_v2", package)
    proof_path = package / "cost_proof_corrected_v2.json"
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof["cmax_main_krw"] = 442_129
    proof_path.write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(policy.Phase13CostPolicyError, match="ARTIFACT_HASH_MISMATCH"):
        policy.validate_cost_policy_package(tmp_path)


def test_cost_policy_validator_rejects_self_consistent_noncanonical_path(
    tmp_path: Path,
) -> None:
    policy = _policy_module()
    package = _copy_complete_policy_fixture(tmp_path)
    alternate = tmp_path / "alternate_stage_registry.json"
    shutil.copyfile(package / "stage_envelope_registry_v1.json", alternate)
    manifest_path = package / "candidate_manifest_corrected_v2.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["stage_envelope_registry"]["path"] = alternate.name
    manifest["manifest_hash"] = _canonical_hash(manifest, "manifest_hash")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(policy.Phase13CostPolicyError, match="CANONICAL_PATH_MISMATCH"):
        policy.validate_cost_policy_package(tmp_path)


def test_cost_policy_validator_rejects_self_consistent_noncanonical_stage(
    tmp_path: Path,
) -> None:
    policy = _policy_module()
    package = _copy_complete_policy_fixture(tmp_path)
    registry_path = package / "stage_envelope_registry_corrected_v2.json"
    proof_path = package / "cost_proof_corrected_v2.json"
    manifest_path = package / "candidate_manifest_corrected_v2.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    registry["stages"][1]["semantic_stage_id"] = "noncanonical_generate"
    registry["registry_hash"] = _canonical_hash(registry, "registry_hash")
    proof["stage_costs"][1]["semantic_stage_id"] = "noncanonical_generate"
    proof["stage_envelope_registry_hash"] = registry["registry_hash"]
    proof["proof_hash"] = _canonical_hash(proof, "proof_hash")
    registry_path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    proof_path.write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
    manifest["artifacts"]["stage_envelope_registry"]["sha256"] = hashlib.sha256(
        registry_path.read_bytes()
    ).hexdigest()
    manifest["artifacts"]["cost_proof"]["sha256"] = hashlib.sha256(
        proof_path.read_bytes()
    ).hexdigest()
    manifest["manifest_hash"] = _canonical_hash(manifest, "manifest_hash")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(policy.Phase13CostPolicyError, match="CANONICAL_STAGE_MISMATCH"):
        policy.validate_cost_policy_package(tmp_path)


def test_cost_policy_validator_rejects_self_consistent_unsafe_handoff(
    tmp_path: Path,
) -> None:
    policy = _policy_module()
    package = _copy_complete_policy_fixture(tmp_path)
    handoff_path = package / "controlled_external_write_v1.json"
    manifest_path = package / "candidate_manifest_corrected_v2.json"
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    handoff["post_write_stop_conditions"]["main_execution_permitted"] = True
    handoff_path.write_text(json.dumps(handoff, indent=2) + "\n", encoding="utf-8")
    manifest["artifacts"]["controlled_external_write"]["sha256"] = hashlib.sha256(
        handoff_path.read_bytes()
    ).hexdigest()
    manifest["manifest_hash"] = _canonical_hash(manifest, "manifest_hash")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(policy.Phase13CostPolicyError, match="CONTROLLED_HANDOFF_MISMATCH"):
        policy.validate_cost_policy_package(tmp_path)
