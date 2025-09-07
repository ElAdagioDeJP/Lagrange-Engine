from __future__ import annotations
from typing import List, Dict, Any
import sympy as sp
import numpy as np
from scipy.optimize import minimize
from .base import OptimizationEngine, SolverResult

class UnconstrainedMinimizer(OptimizationEngine):
    """Wrapper sobre scipy.optimize.minimize (BFGS por defecto)."""
    def __init__(self, objective: sp.Expr, variables: List[sp.Symbol], method: str = 'BFGS', tol: float | None = None):
        super().__init__(objective, variables)
        self.method = method
        self.tol = tol

    def _func(self, x: np.ndarray) -> float:
        return float(self._objective_callable(*x))

    def solve(self, start):  # type: ignore[override]
        res = minimize(self._func, np.array(start, dtype=float), method=self.method, tol=self.tol)
        point = {str(v): float(val) for v, val in zip(self.variables, res.x)}
        return SolverResult(
            method=f'Unconstrained-{self.method}',
            point=point,
            objective_value=float(res.fun),
            iterations=res.nit if hasattr(res, 'nit') else -1,
            converged=bool(res.success),
            extra={'message': res.message}
        )
