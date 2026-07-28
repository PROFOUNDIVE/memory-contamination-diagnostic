from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from memcontam.clients.base import LLMResponse
from memcontam.experiment.phase12.filter_challenge.contracts import (
    CandidateExposureRecord,
    ChallengeCandidate,
)


@dataclass(frozen=True, slots=True)
class ChallengeCallRequest:
    call_id: str
    prompt: str
    model: str


@dataclass(frozen=True, slots=True)
class ChallengeCallResponse:
    call_id: str
    response: LLMResponse


@dataclass(frozen=True, slots=True)
class ParsedAnswer:
    answer_call_id: str
    parser_status: Literal["parsed_raw", "parse_failed"]
    parsed_output: str | None


@dataclass(frozen=True, slots=True)
class VerifiedAnswer:
    answer_call_id: str
    verifier_status: Literal["success", "failed"]
    is_correct: bool | None


class FrozenCheckpoint(Protocol):
    def snapshot_id(self) -> str: ...


class ChallengeCallClient(Protocol):
    def answer(self, request: ChallengeCallRequest) -> ChallengeCallResponse: ...


class AnswerCallObserver(Protocol):
    def observe(self, response: ChallengeCallResponse) -> None: ...


class AnswerParser(Protocol):
    def parse(self, response: ChallengeCallResponse) -> ParsedAnswer: ...


class AnswerVerifier(Protocol):
    def verify(self, answer: ParsedAnswer) -> VerifiedAnswer: ...


class ChallengeAdapter(Protocol):
    def execute(
        self, checkpoint: FrozenCheckpoint, candidate: ChallengeCandidate
    ) -> CandidateExposureRecord: ...
