from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

from memcontam.verifiers.math_equation_balancer import verify_answer as verify_meb
from memcontam.verifiers.word_sorting import verify_words


MEB_SPEC = {"target": "2 + 5 = 7", "target_value": 7}
WORD_GOLD = ["apple", "banana", "pear"]

MEB_CASES = [
    ("correct equation", "2 + 5 = 7", MEB_SPEC, True, "ok"),
    ("correct numeric only", "7", MEB_SPEC, True, "ok"),
    ("correct whitespace", "  7  ", MEB_SPEC, True, "ok"),
    ("wrong equation", "2 + 5 = 8", MEB_SPEC, False, "wrong_answer"),
    ("wrong number only", "8", MEB_SPEC, False, "wrong_answer"),
    ("malformed empty", "", MEB_SPEC, False, "malformed_answer"),
    ("malformed non-string", None, MEB_SPEC, False, "malformed_answer"),
]

WORD_CASES = [
    ("correct order", ["apple", "banana", "pear"], WORD_GOLD, True, "ok"),
    ("wrong order", ["pear", "apple", "banana"], WORD_GOLD, False, "wrong_order"),
    ("malformed not list", "apple banana pear", WORD_GOLD, False, "malformed_answer"),
    ("malformed empty list", [], WORD_GOLD, False, "malformed_answer"),
    ("malformed non-string element", ["apple", 123, "pear"], WORD_GOLD, False, "malformed_answer"),
]


def _format_result(description: str, result) -> str:
    return (
        f"[{description}]\n"
        f"  is_correct={result.is_correct}\n"
        f"  parsed_answer={result.parsed_answer!r}\n"
        f"  reason={result.reason!r}\n"
        f"  metadata={result.metadata}\n"
    )


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    evidence_dir = project_root / ".sisyphus" / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    happy_lines: list[str] = [f"T3 verifier happy-path evidence @ {datetime.now(timezone.utc).isoformat()}\n"]
    negative_lines: list[str] = [f"T3 verifier negative-path evidence @ {datetime.now(timezone.utc).isoformat()}\n"]

    failures: list[str] = []

    for description, answer, spec, expected_correct, expected_reason in MEB_CASES:
        result = verify_meb(answer, spec)
        line = _format_result(f"MEB: {description}", result)
        if result.is_correct:
            happy_lines.append(line)
        else:
            negative_lines.append(line)

        if result.is_correct != expected_correct:
            failures.append(f"MEB '{description}': expected is_correct={expected_correct}, got {result.is_correct}")
        if result.reason != expected_reason:
            failures.append(f"MEB '{description}': expected reason={expected_reason!r}, got {result.reason!r}")
        if not result.reason:
            failures.append(f"MEB '{description}': reason is empty")

    for description, answer, gold, expected_correct, expected_reason in WORD_CASES:
        result = verify_words(answer, gold)
        line = _format_result(f"WordSorting: {description}", result)
        if result.is_correct:
            happy_lines.append(line)
        else:
            negative_lines.append(line)

        if result.is_correct != expected_correct:
            failures.append(f"WordSorting '{description}': expected is_correct={expected_correct}, got {result.is_correct}")
        if result.reason != expected_reason:
            failures.append(f"WordSorting '{description}': expected reason={expected_reason!r}, got {result.reason!r}")
        if not result.reason:
            failures.append(f"WordSorting '{description}': reason is empty")

    (evidence_dir / "task-T3-verifier-happy.txt").write_text("\n".join(happy_lines), encoding="utf-8")
    (evidence_dir / "task-T3-verifier-negative.txt").write_text("\n".join(negative_lines), encoding="utf-8")

    if failures:
        print("FAILURES:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("T3 verifier QA passed.")
    print(f"  happy evidence: {evidence_dir / 'task-T3-verifier-happy.txt'}")
    print(f"  negative evidence: {evidence_dir / 'task-T3-verifier-negative.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
