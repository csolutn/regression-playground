"""Safe evaluation of a student-written formula such as ``sin(x1) + 0.1 * x2**2``.

Only numbers, the allowed variable names, arithmetic and the functions in
FUNCTIONS are accepted, so nothing else in Python can be reached.
"""
import ast
import operator

import numpy as np

FUNCTIONS = {
    'abs': np.abs, 'sqrt': np.sqrt, 'exp': np.exp, 'log': np.log,
    'sin': np.sin, 'cos': np.cos, 'tan': np.tan, 'tanh': np.tanh,
    'sign': np.sign, 'floor': np.floor, 'round': np.round,
    'max': np.maximum, 'min': np.minimum,
}
CONSTANTS = {'pi': np.pi, 'e': np.e}
BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
}
UNARY_OPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}
MAX_LENGTH = 200


class ExpressionError(ValueError):
    """``kind`` is one of: syntax, unknown_name, not_allowed, not_finite."""

    def __init__(self, kind, name=''):
        super().__init__(f'{kind} {name}'.strip())
        self.kind, self.name = kind, name


def evaluate(text, variables):
    """Evaluate ``text`` with numpy arrays in ``variables`` (e.g. {'x': array})."""
    text = (text or '').strip().replace('^', '**')
    if not text:
        raise ExpressionError('syntax')
    if len(text) > MAX_LENGTH:
        raise ExpressionError('not_allowed')
    try:
        tree = ast.parse(text, mode='eval')
    except SyntaxError as exc:
        raise ExpressionError('syntax') from exc

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id in variables:
                return variables[node.id]
            if node.id in CONSTANTS:
                return CONSTANTS[node.id]
            raise ExpressionError('unknown_name', node.id)
        if isinstance(node, ast.BinOp) and type(node.op) in BIN_OPS:
            return BIN_OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY_OPS:
            return UNARY_OPS[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCTIONS and not node.keywords:
            return FUNCTIONS[node.func.id](*[ev(a) for a in node.args])
        raise ExpressionError('not_allowed')

    shape = np.shape(next(iter(variables.values())))
    with np.errstate(all='ignore'):
        try:
            result = np.broadcast_to(np.asarray(ev(tree), dtype=float), shape).copy()
        except ExpressionError:
            raise
        except (TypeError, ValueError, OverflowError, ZeroDivisionError) as exc:
            raise ExpressionError('not_allowed') from exc
    if not np.all(np.isfinite(result)):
        raise ExpressionError('not_finite')
    return result
