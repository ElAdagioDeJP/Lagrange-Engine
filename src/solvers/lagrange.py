from __future__ import annotations
from typing import List, Dict, Any
import sympy as sp
import numpy as np
from .base import OptimizationEngine, SolverResult

class LagrangeSolver(OptimizationEngine):
    def __init__(self, objective: sp.Expr, variables: List[sp.Symbol], equality_constraints: List[sp.Eq]):
        super().__init__(objective, variables)
        self.equality_constraints = equality_constraints
        self.lambdas = sp.symbols(f"lambda0:{len(equality_constraints)}") if equality_constraints else []
        # Build Lagrangian
        self.L = objective + sum(l * (c.lhs - c.rhs) for l, c in zip(self.lambdas, equality_constraints))
        self.gradient = [sp.diff(self.L, v) for v in list(variables) + list(self.lambdas)]

    def solve(self, start):  # type: ignore[override]
        if len(start) != len(self.variables):
            raise ValueError("Start vector size mismatch")
        # Initial guesses for lambdas are zeros
        init_vals = list(start) + [0.0]*len(self.lambdas)
        symbols = list(self.variables) + list(self.lambdas)
        equations = [sp.diff(self.L, s) for s in symbols]
        solutions = sp.nsolve(equations, symbols, init_vals, tol=1e-12, maxsteps=200)
        x_sol = solutions[:len(self.variables)]
        point = {str(v): float(val) for v, val in zip(self.variables, x_sol)}
        return SolverResult(
            method='LagrangeMultipliers',
            point=point,
            objective_value=float(self._objective_callable(*x_sol)),
            iterations=1,
            converged=True,
            extra={'lambdas': {str(l): float(val) for l, val in zip(self.lambdas, solutions[len(self.variables):])}}
        )
