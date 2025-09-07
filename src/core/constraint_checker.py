from __future__ import annotations
from typing import List, Tuple
import sympy as sp
from .function_parser import ParsedProblem
from src.solvers.base import SolverResult

def evaluate_constraints(problem: ParsedProblem, results: List[SolverResult]) -> Tuple[List[str], List[str]]:
    eq_lines = []
    ineq_lines = []
    for r in results:
        subs = {sp.Symbol(k): v for k, v in r.point.items()}
        for c in problem.equality_constraints:
            lhs = float(c.lhs.subs(subs))
            rhs = float(c.rhs.subs(subs))
            eq_lines.append(f"{r.method}: {sp.simplify(c.lhs - c.rhs)} = {lhs - rhs:+.3e}")
        for c in problem.inequality_constraints:
            expr = c.lhs - c.rhs
            val = float(expr.subs(subs))
            ok = False
            if isinstance(c, sp.StrictLessThan):
                ok = val < 0
            elif isinstance(c, sp.StrictGreaterThan):
                ok = val > 0
            elif isinstance(c, sp.LessThan):
                ok = val <= 0
            elif isinstance(c, sp.GreaterThan):
                ok = val >= 0
            ineq_lines.append(f"{r.method}: {sp.simplify(expr)} = {val:+.3e} ({'OK' if ok else 'VIOLA'})")
    return eq_lines, ineq_lines
