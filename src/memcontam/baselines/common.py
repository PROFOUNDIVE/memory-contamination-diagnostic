from __future__ import annotations


FINAL_ANSWER_INSTRUCTION = "Return exactly one non-empty terminal line: final: <answer>."
FINAL_ANSWER_PROMPT_VERSION = "terminal_final_line_prompt_v2"
FINAL_ANSWER_PARSER_VERSION = "terminal_final_line_v1"


def parse_final_answer(response: str) -> str:
    """Return the answer from one non-empty terminal ``final:`` line."""
    if not isinstance(response, str):
        raise ValueError("response must be a string")
    lines = response.splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    final_lines = [line for line in lines if line.lower().startswith("final:")]
    if len(final_lines) != 1 or not lines or lines[-1] != final_lines[0]:
        raise ValueError("response must contain exactly one terminal final: line")
    answer = final_lines[0][len("final:") :].strip()
    if not answer:
        raise ValueError("final answer must be non-empty")
    return answer
