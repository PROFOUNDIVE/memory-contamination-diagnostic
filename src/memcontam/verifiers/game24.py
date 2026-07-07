from __future__ import annotations

import ast
import math
from collections import Counter
from typing import Any

from memcontam.logging.schema import VerifierResult

MAX_EXPRESSION_LENGTH = 500
MAX_AST_NODES = 64


def _evaluate(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        raise ValueError
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return float(node.value)
    if isinstance(node, ast.BinOp):
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
    raise ValueError


def _numbers_used(node: ast.AST) -> list[int]:
    numbers: list[int] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, bool):
            raise ValueError
        if isinstance(child, ast.Constant) and isinstance(child.value, int):
            numbers.append(child.value)
        elif isinstance(child, ast.Constant) and isinstance(child.value, float):
            if not child.value.is_integer():
                raise ValueError
            numbers.append(int(child.value))
    return numbers


def _unsupported_nodes(tree: ast.AST) -> list[ast.AST]:
    supported: tuple[type[Any], ...] = (
        ast.Expression,
        ast.BinOp,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Load,
    )
    return [node for node in ast.walk(tree) if not isinstance(node, supported)]


def verify_expression(expression: str, numbers: list[int], target: int = 24) -> VerifierResult:
    try:
        if len(expression) > MAX_EXPRESSION_LENGTH:
            raise ValueError
        tree = ast.parse(expression, mode="eval")
        if sum(1 for _node in ast.walk(tree)) > MAX_AST_NODES:
            raise ValueError
        if _unsupported_nodes(tree):
            raise ValueError
        used_numbers = _numbers_used(tree)
        if Counter(used_numbers) != Counter(numbers):
            return VerifierResult(
                is_correct=False,
                parsed_answer=expression,
                reason="numbers_used_do_not_match",
                metadata={"numbers_used": used_numbers, "expected_numbers": numbers},
            )
        value = _evaluate(tree)
    except (SyntaxError, ValueError, ZeroDivisionError, RecursionError, OverflowError):
        return VerifierResult(
            is_correct=False,
            parsed_answer=expression,
            reason="unsupported_expression",
        )

    if not math.isclose(value, target, rel_tol=0, abs_tol=1e-9):
        return VerifierResult(
            is_correct=False,
            parsed_answer=expression,
            reason="value_does_not_match_target",
            metadata={"value": value, "target": target},
        )

    return VerifierResult(
        is_correct=True,
        parsed_answer=expression,
        reason="ok",
        metadata={"value": int(value) if value.is_integer() else value, "target": target},
    )
