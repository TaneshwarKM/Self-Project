"""
calculator.py
Safe arithmetic evaluation tool. Uses Python's ast module to evaluate
only a whitelisted set of math operations — never raw eval() on
model-provided input, since that would be an arbitrary code execution
hole disguised as a calculator.
"""

from __future__ import annotations

import ast
import operator

CALCULATOR_SCHEMA = {
    "name": "calculator",
    "description": (
        "Evaluate a mathematical expression and return the numeric result. "
        "Supports +, -, *, /, //, %, **, parentheses, and unary minus. "
        "Use this for any arithmetic instead of computing it yourself."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "A mathematical expression, e.g. '(4 + 5) * 12 / 3'",
            }
        },
        "required": ["expression"],
    },
}

_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"Unsupported expression element: {ast.dump(node)}")


def run_calculator(expression: str) -> dict:
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree.body)
        return {"success": True, "result": result}
    except ZeroDivisionError:
        return {"success": False, "error": "Division by zero"}
    except Exception as exc:  # noqa: BLE001 — surface any parse/eval issue to the agent
        return {"success": False, "error": f"Could not evaluate expression: {exc}"}
