"""Safe evaluation of array expressions used in property definitions.

Example: ``"where(K <= 10, 0.1 * PERMX, PERMX)"``.

Only arithmetic, comparisons, ``&`` / ``|`` and a whitelist of NumPy functions
are allowed. Names resolve to property arrays or the index arrays I, J, K.
"""

from __future__ import annotations

import ast
import operator

import numpy as np

FUNCTIONS = {
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "minimum": np.minimum,
    "maximum": np.maximum,
    "where": np.where,
    "clip": np.clip,
}

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.BitAnd: np.logical_and,
    ast.BitOr: np.logical_or,
}
_UNARY = {ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Invert: np.logical_not}
_COMPARE = {
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}


def _parse(expr: str) -> ast.Expression:
    try:
        return ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"invalid expression {expr!r}: {exc.msg}") from None


def expression_names(expr: str) -> set[str]:
    """Names referenced by an expression, excluding function names."""
    tree = _parse(expr)
    funcs = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} - funcs


def evaluate(expr: str, variables: dict[str, np.ndarray]):
    """Evaluate ``expr`` with ``variables`` in scope."""

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in variables:
                raise ValueError(f"unknown name {node.id!r} in expression {expr!r}")
            return variables[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            return _BINOPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in _COMPARE:
            return _COMPARE[type(node.ops[0])](ev(node.left), ev(node.comparators[0]))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in FUNCTIONS
            and not node.keywords
        ):
            return FUNCTIONS[node.func.id](*[ev(a) for a in node.args])
        raise ValueError(f"unsupported syntax in expression {expr!r}: {ast.dump(node)[:60]}")

    return ev(_parse(expr))
