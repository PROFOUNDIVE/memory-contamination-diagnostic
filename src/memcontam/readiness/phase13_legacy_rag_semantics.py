from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal


SemanticStratum = Literal["A", "B", "C"]


@dataclass(frozen=True, slots=True)
class SemanticRecord:
    record_id: str
    stratum: SemanticStratum
    title: str
    body: str


@dataclass(frozen=True, slots=True)
class UnknownSemanticTaskError(ValueError):
    task: str

    def __str__(self) -> str:
        return f"unknown legacy RAG semantic task: {self.task}"


_GAME24: Final = (
    SemanticRecord("G24-A01", "A", "Use every supplied occurrence",
                   "A valid Game24 expression uses every supplied number occurrence exactly once.\n"
                   "A repeated numeric value represents repeated occurrences; multiplicity must be preserved."),
    SemanticRecord("G24-A02", "A", "Allowed operations",
                   "Use only addition, subtraction, multiplication, and division between supplied values or their intermediate results.\n"
                   "Do not use concatenation, exponentiation, factorials, or newly introduced numeric constants."),
    SemanticRecord("G24-A03", "A", "Parenthesization",
                   "Parentheses may determine the order of arithmetic evaluation.\n"
                   "The validity of the expression is determined by its exact arithmetic structure, not by a bag of operations."),
    SemanticRecord("G24-A04", "A", "Exact target",
                   "The final value of the complete expression must be exactly 24 under exact arithmetic.\n"
                   "An expression that is merely close to 24 is incorrect."),
    SemanticRecord("G24-A05", "A", "Intermediate values",
                   "Negative and fractional intermediate values are allowed when they arise from the permitted operations.\n"
                   "Division by zero is never allowed."),
    SemanticRecord("G24-A06", "A", "No hidden number reuse",
                   "A subexpression cannot reuse a supplied number occurrence that has already been consumed elsewhere in the expression.\n"
                   "The complete expression must account for the original multiset exactly."),
    SemanticRecord("G24-B01", "B", "Duplicate numbers",
                   "When the input contains duplicate values, treat them as distinct occurrences with the stated multiplicity.\n"
                   "Using one occurrence twice is invalid even when the repeated values are numerically identical."),
    SemanticRecord("G24-B02", "B", "Division by zero",
                   "Any expression tree containing division by an intermediate value equal to zero is invalid regardless of its other branches."),
    SemanticRecord("G24-B03", "B", "Noncommutative operations",
                   "Subtraction and division are order-sensitive.\n"
                   "Swapping their left and right operands can change both the intermediate value and final validity."),
    SemanticRecord("G24-B04", "B", "Fractional paths",
                   "Do not reject a candidate merely because an intermediate result is fractional.\n"
                   "A fractional path is valid if every operation is defined and the complete exact value is 24."),
    SemanticRecord("G24-B05", "B", "Correct value with wrong multiplicity",
                   "Reaching 24 is insufficient when a supplied occurrence is omitted, duplicated, or replaced.\n"
                   "Both arithmetic equality and exact occurrence use are required."),
    SemanticRecord("G24-B06", "B", "Syntactic shortcuts",
                   "A notation that implicitly concatenates digits or introduces an operation outside the registered four operators does not satisfy the Game24 task even if its numerical value is 24."),
    SemanticRecord("G24-C01", "C", "Factor the target",
                   "A useful search strategy is to look for intermediate factors of 24, such as 2 and 12, 3 and 8, or 4 and 6, and test whether disjoint subsets of the supplied occurrences can produce them."),
    SemanticRecord("G24-C02", "C", "Pair subexpressions",
                   "Partitioning the four occurrences into small subexpressions can reduce search complexity.\n"
                   "After constructing each subexpression, combine them only if their occurrence sets are disjoint."),
    SemanticRecord("G24-C03", "C", "Create a small adjustment",
                   "Differences or ratios that produce a small adjustment such as 1 can help convert another intermediate value into 24.\n"
                   "Always verify that the adjustment itself uses only unused supplied occurrences."),
    SemanticRecord("G24-C04", "C", "Check subtraction routes",
                   "When direct multiplication overshoots the target, test whether a remaining subexpression can be subtracted to reach 24."),
    SemanticRecord("G24-C05", "C", "Preserve exact divisibility when useful",
                   "When considering division, prefer branches whose denominator is nonzero and whose resulting exact value can still combine naturally with the remaining occurrences.\n"
                   "This is a search heuristic, not a validity requirement."),
    SemanticRecord("G24-C06", "C", "Final audit",
                   "Before accepting a candidate, audit both conditions independently: every supplied occurrence appears exactly once, and exact evaluation of the complete expression equals 24."),
)

_MATH_EQUATION_BALANCER: Final = (
    SemanticRecord("MEB-A01", "A", "Fill every operator slot",
                   "A MathEquationBalancer instance supplies an ordered sequence of operands with question-mark operator slots.\n"
                   "A valid response fills every registered operator slot with an allowed operator."),
    SemanticRecord("MEB-A02", "A", "Preserve operand order",
                   "The supplied operands must remain in exactly their registered order.\n"
                   "Do not reorder, delete, duplicate, or replace an operand."),
    SemanticRecord("MEB-A03", "A", "Use the registered operator set",
                   "Each inserted operator must belong to the task's allowed set: addition, subtraction, multiplication, or division.\n"
                   "Do not introduce a different operation."),
    SemanticRecord("MEB-A04", "A", "Respect evaluation semantics",
                   "Evaluate the filled expression under the registered arithmetic semantics: multiplication and division have higher precedence than addition and subtraction, with left associativity within each precedence level."),
    SemanticRecord("MEB-A05", "A", "Return a complete filled equation",
                   "A scientifically valid response is a complete operator-filled equation matching the registered operand sequence and right-hand target.\n"
                   "The right-hand numeric target by itself is not a complete MathEquationBalancer response."),
    SemanticRecord("MEB-A06", "A", "Make the equation true",
                   "The filled left-hand expression must evaluate exactly to the registered right-hand target.\n"
                   "Every operator slot, operand, and target must satisfy the same equation."),
    SemanticRecord("MEB-B01", "B", "Bare target is insufficient",
                   "Returning only verifier_spec.target_value does not demonstrate an operator assignment and is invalid under the locked MathEquationBalancer task semantics."),
    SemanticRecord("MEB-B02", "B", "Alternative valid operator assignments",
                   "The frozen verifier_spec.target is the canonical construction solution used for deterministic worked examples.\n"
                   "Another complete operator assignment may also be correct if it preserves the registered operand order, uses only allowed operators, and makes the full equation exactly true."),
    SemanticRecord("MEB-B03", "B", "Precedence changes results",
                   "For a sequence containing both multiplication or division and addition or subtraction, evaluate multiplication and division first under the registered precedence rule.\n"
                   "Do not evaluate all operators strictly from left to right."),
    SemanticRecord("MEB-B04", "B", "Division by zero",
                   "An inserted division operator is invalid whenever its denominator evaluates to zero under the registered expression structure."),
    SemanticRecord("MEB-B05", "B", "Negative values",
                   "A negative intermediate value or negative registered target is not inherently invalid.\n"
                   "Correctness depends on the exact value of the complete filled equation."),
    SemanticRecord("MEB-B06", "B", "No unregistered parentheses",
                   "The current canonical MathEquationBalancer response grammar does not add parentheses that are absent from the registered operator-slot instance.\n"
                   "Solve the equation by filling the existing slots under the registered precedence law."),
    SemanticRecord("MEB-C01", "C", "Compare target scale",
                   "Use the magnitude and sign of the right-hand target to prioritize plausible operator patterns.\n"
                   "Large positive targets often justify checking multiplication early, while this remains only a search heuristic."),
    SemanticRecord("MEB-C02", "C", "Account for precedence before choosing signs",
                   "When a multiplication or division is present, compute its local effect before deciding whether surrounding addition or subtraction can reach the target."),
    SemanticRecord("MEB-C03", "C", "Search operator tuples systematically",
                   "Treat the missing operators as an ordered tuple and test candidate tuples systematically rather than changing operand order."),
    SemanticRecord("MEB-C04", "C", "Use subtraction for controlled adjustment",
                   "After identifying a large intermediate term, subtraction can provide a controlled adjustment toward the target.\n"
                   "Verify the complete expression under precedence rather than reasoning from the adjustment alone."),
    SemanticRecord("MEB-C05", "C", "Check division exactly",
                   "When testing division, use exact arithmetic and reject zero denominators.\n"
                   "Do not rely on rounded floating-point agreement with the target."),
    SemanticRecord("MEB-C06", "C", "Final structural audit",
                   "Before accepting a response, verify the operand sequence, every inserted operator, the right-hand target, and exact evaluation of the complete equation."),
)

_WORD_SORTING: Final = (
    SemanticRecord("WS-A01", "A", "Preserve the token multiset",
                   "A valid WordSorting response contains exactly the supplied word occurrences.\n"
                   "Do not add, delete, substitute, or duplicate an occurrence."),
    SemanticRecord("WS-A02", "A", "Use the registered lexical comparator",
                   "Order the words according to the task's registered lexical sorting semantics.\n"
                   "The expected output is determined by that comparator, not by semantic category or word meaning."),
    SemanticRecord("WS-A03", "A", "Compare from the beginning",
                   "When two words differ, compare their characters from left to right until the first differing position determines their lexical order."),
    SemanticRecord("WS-A04", "A", "Preserve spelling",
                   "Sorting changes position, not the identity of a word token.\n"
                   "Preserve each supplied token's registered spelling."),
    SemanticRecord("WS-A05", "A", "Return the full sequence",
                   "The response must provide the complete sorted sequence.\n"
                   "A partial prefix or a statement naming only the first or last word is insufficient."),
    SemanticRecord("WS-A06", "A", "Multiplicity remains binding",
                   "If the same token occurs more than once, every occurrence must remain present in the sorted output with the same multiplicity."),
    SemanticRecord("WS-B01", "B", "Shared prefixes",
                   "Words sharing an initial prefix must be compared beyond that prefix using the registered comparator until their order is resolved."),
    SemanticRecord("WS-B02", "B", "Duplicate tokens",
                   "Duplicate input tokens remain duplicate output occurrences.\n"
                   "Deduplicating them changes the task instance and is invalid."),
    SemanticRecord("WS-B03", "B", "Meaning is irrelevant",
                   "Do not group or order words by semantic similarity, category, frequency, or perceived importance.\n"
                   "Only the registered lexical comparator controls order."),
    SemanticRecord("WS-B04", "B", "Exact token identity",
                   "Do not silently rewrite a word, change its characters, or substitute a synonym while sorting."),
    SemanticRecord("WS-B05", "B", "Local correctness is insufficient",
                   "Having several adjacent pairs in the correct order is insufficient if another pair remains out of order.\n"
                   "The complete output sequence must match the registered sorted sequence."),
    SemanticRecord("WS-B06", "B", "No omitted occurrence",
                   "A lexically correct sequence of only a subset of the input is still invalid because the original occurrence multiset was not preserved."),
    SemanticRecord("WS-C01", "C", "Group by early characters",
                   "A useful sorting strategy is to group words by their earliest differing characters before resolving finer within-group order."),
    SemanticRecord("WS-C02", "C", "Resolve one prefix group at a time",
                   "After grouping words with a common prefix, compare the next unresolved character position within that group."),
    SemanticRecord("WS-C03", "C", "Maintain a working ordered list",
                   "Insert each unsorted token into a working list at the position determined by lexical comparison with neighboring tokens."),
    SemanticRecord("WS-C04", "C", "Compare only until order is known",
                   "When comparing two tokens, stop at the first character position that resolves their lexical order unless the registered comparator requires further comparison."),
    SemanticRecord("WS-C05", "C", "Track multiplicity separately",
                   "When duplicates are possible, track occurrence counts while sorting so that no repeated token is accidentally dropped."),
    SemanticRecord("WS-C06", "C", "Final sequence audit",
                   "Before accepting the answer, check both that the complete sequence is lexically sorted and that its token multiset exactly equals the input multiset."),
)


def semantic_records(task: str) -> tuple[SemanticRecord, ...]:
    registries = {
        "game24": _GAME24,
        "math_equation_balancer": _MATH_EQUATION_BALANCER,
        "word_sorting": _WORD_SORTING,
    }
    try:
        return registries[task]
    except KeyError as error:
        raise UnknownSemanticTaskError(task) from error


def render_semantic_record(record: SemanticRecord) -> bytes:
    return f"Title: {record.title}\nRule: {record.body}".encode("utf-8")


__all__ = (
    "SemanticRecord",
    "UnknownSemanticTaskError",
    "render_semantic_record",
    "semantic_records",
)
