from __future__ import annotations

import hashlib
import re

from memcontam.readiness.phase13_cost_policy_models import Capacity
from memcontam.readiness.phase13_v3_authority_models import (
    ROUTED_DOCUMENTS,
    AuthorityStage,
    RetryAllocationRegistry,
    TerminalContract,
    TransportAttemptContract,
    V3Registry,
)


def canonical_block(raw: bytes, identity: str) -> bytes:
    pattern = rb"(?m)^BEGIN_" + identity.encode() + rb"\n.*?^END_" + identity.encode() + rb"(?=\n|$)"
    blocks = re.findall(pattern, raw, re.DOTALL)
    if len(blocks) != 1:
        raise ValueError(identity)
    return blocks[0]


def parse_registry(raw: bytes) -> V3Registry:
    block = canonical_block(raw, "CORE_EXECUTION_ENVELOPE_REGISTRY_V4")
    lines = block.decode("utf-8").splitlines()[1:-1]
    separator = lines.index("semantic_stage_id|max_output_tokens|max_input_tokens_per_attempt")
    metadata = dict(line.split("=", 1) for line in lines[:separator])
    stages = []
    for line in lines[separator + 1:]:
        stage, output_limit, input_limit = line.split("|")
        stages.append(AuthorityStage(stage_id=stage, maximum_output_tokens=int(output_limit), maximum_input_tokens=int(input_limit)))
    return V3Registry.model_validate({
        "registry_id": metadata["registry_id"],
        "sha256": hashlib.sha256(block).hexdigest(),
        "transport_contract_id": metadata["transport_contract_id"],
        "transport_contract_sha256": metadata["transport_contract_sha256"],
        "retry_allocation_registry_id": metadata["retry_allocation_registry_id"],
        "retry_allocation_registry_sha256": metadata["retry_allocation_registry_sha256"],
        "terminal_missingness_contract_id": metadata["terminal_missingness_contract_id"],
        "terminal_missingness_contract_sha256": metadata["terminal_missingness_contract_sha256"],
        "per_request_timeout_seconds": int(metadata["per_request_timeout_seconds"]),
        "default_max_transport_attempts": int(metadata["default_max_transport_attempts"]),
        "entitled_eligible_max_transport_attempts": int(metadata["entitled_eligible_max_transport_attempts"]),
        "maximum_retries_after_initial_attempt": int(metadata["maximum_retries_after_initial_attempt"]),
        "retry_budget_krw": int(metadata["retry_budget_krw"]),
        "stages": tuple(stages),
    })


def parse_terminal(raw: bytes) -> TerminalContract:
    block = canonical_block(raw, "CORE_TERMINAL_TECHNICAL_MISSINGNESS_V2")
    fields = tuple(line.split("=", 1) for line in block.decode("utf-8").splitlines()[1:-1])
    return TerminalContract.model_validate({
        "contract_id": fields[0][1], "sha256": hashlib.sha256(block).hexdigest(),
        "intermediate_triggers": tuple(value for key, value in fields if key == "intermediate_trigger"),
        "terminal_triggers": tuple(value for key, value in fields if key == "terminal_trigger"),
        "propagation": tuple(value for key, value in fields if key == "propagation"),
    })


def parse_retry(raw: bytes) -> RetryAllocationRegistry:
    block = canonical_block(raw, "CORE_RETRY_ALLOCATION_REGISTRY_V1")
    metadata = dict(
        line.split("=", 1) for line in block.decode("utf-8").splitlines()[1:-1]
    )
    return RetryAllocationRegistry.model_validate({
        "registry_id": metadata["registry_id"],
        "sha256": hashlib.sha256(block).hexdigest(),
        "retry_budget_krw": int(metadata["retry_budget_krw"]),
        "authorization_gate_krw": int(metadata["authorization_gate_krw"]),
        "retry_slots_per_entitled_request": int(metadata["retry_slots_per_entitled_request"]),
    })


def parse_transport(raw: bytes) -> TransportAttemptContract:
    block = canonical_block(raw, "CORE_TRANSPORT_ATTEMPT_CONTRACT_V3")
    fields = tuple(
        line.split("=", 1) for line in block.decode("utf-8").splitlines()[1:-1]
    )
    metadata = dict(fields)
    return TransportAttemptContract.model_validate({
        "contract_id": metadata["contract_id"],
        "sha256": hashlib.sha256(block).hexdigest(),
        "default_max_attempts": int(metadata["default_max_attempts"]),
        "entitled_eligible_max_attempts": int(metadata["entitled_eligible_max_attempts"]),
        "maximum_retries_after_initial_attempt": int(metadata["maximum_retries_after_initial_attempt"]),
        "eligible_failures": tuple(value for key, value in fields if key == "eligible_failure"),
        "ineligible_failures": tuple(value for key, value in fields if key == "ineligible_failure"),
    })


def validate_router(raw: bytes) -> None:
    text = raw.decode("utf-8")
    section = text.split("## Authority\n", 1)[1].split("## Ownership", 1)[0]
    labels = ("Theoretical authority", "Baseline authority", "Contamination protocol authority",
              "Narrow post-cutoff Main-A addendum", "Experiment-design authority",
              "Corrective scientific decision authority for separately versioned corrected Main-A only")
    for label, (_, filename) in zip(labels, ROUTED_DOCUMENTS[:6], strict=True):
        targets = re.findall(r"(?m)^\* " + re.escape(label) + r": `([^`]+)`$", section)
        if targets != [filename]:
            raise ValueError(label)
    prefix = "* **충돌 시 우선순위는 "
    precedence = [line for line in section.splitlines() if line.startswith(prefix)]
    canonical = (prefix + "Theory → Baseline → Contamination Protocol → `"
                 + ROUTED_DOCUMENTS[3][1]
                 + "`(명시된 narrow scope에 한함) → Experiment Design이다.**")
    if precedence != [canonical]:
        raise ValueError("precedence")


def parse_capacity(raw: bytes) -> Capacity:
    text = raw.decode("utf-8")
    fields = {}
    for source, target in (("O_writer_reg", "writer_max_output_tokens"),
                           ("B_DC_feasible", "B_DC_feasible"),
                           ("B_mem_tokens", "B_mem_tokens"), ("L_DC_tokens", "L_DC_tokens")):
        values = re.findall(r"`" + source + r"=(\d+)`", text)
        if not values or set(values) != {"8192"}:
            raise ValueError(source)
        fields[target] = int(values[0])
    return Capacity.model_validate(fields)
