from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Final, Literal

from pydantic import Field

from memcontam.readiness.phase13_cost_policy_models import Sha256
from memcontam.readiness.phase13_v3_authority_models import FrozenModel, V3Identity
from memcontam.readiness.phase13_v3_terminal_models import CompiledRequestV3


Stage = Literal["full_history_generate", "rag_generate", "bot_problem_distill",
                "bot_instantiate_solve", "bot_thought_distill", "reflexion_generate",
                "reflexion_reflect", "dc_rs_generate", "dc_rs_synthesize", "no_memory_generate"]
STAGES: Final[dict[Stage, tuple[int, int]]] = {
    "full_history_generate": (9330, 512), "rag_generate": (378, 512),
    "bot_problem_distill": (1177, 384), "bot_instantiate_solve": (1949, 512),
    "bot_thought_distill": (2545, 384), "reflexion_generate": (2282, 512),
    "reflexion_reflect": (3349, 384), "dc_rs_generate": (9212, 512),
    "dc_rs_synthesize": (13521, 8192), "no_memory_generate": (1160, 512),
}


class PackageBindingV3(FrozenModel):
    identity: V3Identity
    package_sha256: Sha256
    authorization_sha256: Sha256


class ParentTrajectoryV3(FrozenModel):
    parent_id: Sha256
    kind: Literal["CLEAN_PREFIX", "MEMORY_BEARING", "NO_MEMORY_SINGLETON"]
    prefix_parent_id: Sha256 | None = None


class RequestKeyV3(FrozenModel):
    parent_id: Sha256
    stage: Stage
    ordinal: int = Field(ge=0)

    @property
    def dispatch_id(self) -> str:
        return hashlib.sha256(json.dumps(
            ["phase13-dispatch-request-v3", self.parent_id, self.stage, self.ordinal],
            separators=(",", ":"),
        ).encode()).hexdigest()


class MessageV3(FrozenModel):
    role: Literal["system", "developer", "user", "assistant"]
    content: str


class RequestMaterialV3(FrozenModel):
    messages: tuple[MessageV3, ...]
    native_state: bytes
    temperature: float = 0.0
    top_p: float = 1.0


@dataclass(frozen=True, slots=True)
class CompiledProviderRequestV3:
    binding: PackageBindingV3
    key: RequestKeyV3
    material: RequestMaterialV3
    request_bytes: bytes
    token_count: int

    @property
    def native_state(self) -> bytes:
        return self.material.native_state

    @property
    def evidence(self) -> CompiledRequestV3:
        return CompiledRequestV3(
            stage=self.key.stage, token_count=self.token_count,
            compiled_request_hash=hashlib.sha256(self.request_bytes).hexdigest(),
            immutable_input_hash=hashlib.sha256(input_bytes(self.material)).hexdigest(),
            native_state_hash=hashlib.sha256(self.native_state).hexdigest(),
        )


def input_bytes(material: RequestMaterialV3) -> bytes:
    return json.dumps([message.model_dump() for message in material.messages],
                      sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def compile_request_bytes(key: RequestKeyV3, material: RequestMaterialV3) -> bytes:
    return json.dumps({
        "model": "gpt-5.6-luna", "input": json.loads(input_bytes(material)),
        "temperature": material.temperature, "top_p": material.top_p,
        "max_output_tokens": STAGES[key.stage][1], "service_tier": "default",
        "store": False, "tools": [],
        "reasoning": {"mode": "standard", "effort": "none", "context": "current_turn"},
    }, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
