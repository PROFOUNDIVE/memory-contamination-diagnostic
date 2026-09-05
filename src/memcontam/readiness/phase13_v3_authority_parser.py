from __future__ import annotations

import hashlib
import re

from memcontam.readiness.phase13_cost_policy_models import Capacity
from memcontam.readiness.phase13_v3_authority_models import (
    ROUTED_DOCUMENTS, AuthorityStage, TerminalContract, V3Registry,
)


def canonical_block(raw: bytes, identity: str) -> bytes:
    pattern = rb"(?m)^BEGIN_" + identity.encode() + rb"\n.*?^END_" + identity.encode() + rb"(?=\n|$)"
    blocks = re.findall(pattern, raw, re.DOTALL)
    if len(blocks) != 1:
        raise ValueError(identity)
    return blocks[0]


def parse_registry(raw: bytes) -> V3Registry:
    block = canonical_block(raw, "CORE_EXECUTION_ENVELOPE_REGISTRY_V3")
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
        "per_request_timeout_seconds": int(metadata["per_request_timeout_seconds"]),
        "max_transport_attempts": int(metadata["max_transport_attempts"]),
        "transport_retries": int(metadata["transport_retries"]), "T": int(metadata["T"]),
        "stages": tuple(stages),
    })


def parse_terminal(raw: bytes) -> TerminalContract:
    block = canonical_block(raw, "CORE_TERMINAL_TECHNICAL_MISSINGNESS_V1")
    fields = tuple(line.split("=", 1) for line in block.decode("utf-8").splitlines()[1:-1])
    return TerminalContract.model_validate({
        "contract_id": fields[0][1], "sha256": hashlib.sha256(block).hexdigest(),
        "triggers": tuple(value for key, value in fields if key == "trigger"),
        "propagation": tuple(value for key, value in fields if key == "propagation"),
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
