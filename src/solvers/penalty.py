from __future__ import annotations
from typing import List
import sympy as sp
import numpy as np
from .base import OptimizationEngine, SolverResult
from .unconstrained import UnconstrainedMinimizer


class PenaltyInequalitySolver(OptimizationEngine):
    """Resuelve problemas con desigualdades usando penalización cuadrática.

    Convención: cada restricción se normaliza a forma g(x) <= 0.
    Penalización: f_rho(x) = f(x) + rho * sum(max(0, g_i(x))**2)
    Se incrementa rho hasta que las violaciones caen bajo un umbral o se alcanza max_iter_outer.
    """

    def __init__(self, objective: sp.Expr, variables: List[sp.Symbol], inequalities: List[sp.Rel],
                 rho0: float = 10.0, rho_factor: float = 5.0, tol_cons: float = 1e-4, max_outer: int = 5):
        super().__init__(objective, variables)
        self.in_equalities = inequalities
        self.rho0 = rho0
        self.rho_factor = rho_factor
        self.tol_cons = tol_cons
        self.max_outer = max_outer

        # Construir expresiones g_i(x) <= 0
        self._g_exprs: List[sp.Expr] = []
        for c in inequalities:
            if isinstance(c, (sp.LessThan, sp.StrictLessThan)):
                expr = c.lhs - c.rhs
            elif isinstance(c, (sp.GreaterThan, sp.StrictGreaterThan)):
                expr = c.rhs - c.lhs
            else:
                # fallback genérico: lhs - rhs
                expr = c.lhs - c.rhs
            self._g_exprs.append(sp.simplify(expr))

    def _build_penalized(self, rho: float) -> sp.Expr:
        if not self._g_exprs:
            return self.objective
        penalty_terms = [sp.Max(0, g)**2 for g in self._g_exprs]
        return self.objective + rho * sum(penalty_terms)

    def _violations(self, point: np.ndarray) -> List[float]:
        vals = []
        for g in self._g_exprs:
            g_func = sp.lambdify(self.variables, g, 'numpy')
            vals.append(float(g_func(*point)))
        # queremos max(0, g(x))
        return [max(0.0, v) for v in vals]

    def solve(self, start):  # type: ignore[override]
        x_curr = np.array(start, dtype=float)
        history = []
        rho = self.rho0
        best_res = None
        for outer in range(1, self.max_outer + 1):
            pen_obj = self._build_penalized(rho)
            # Minimizar usando BFGS
            solver = UnconstrainedMinimizer(pen_obj, self.variables)
            res = solver.solve(x_curr)
            x_curr = np.array([res.point[str(v)] for v in self.variables])
            viols = self._violations(x_curr)
            max_viol = max(viols) if viols else 0.0
            history.append({'outer': outer, 'rho': rho, 'point': x_curr.copy(), 'max_violation': max_viol})
            best_res = res
            if max_viol < self.tol_cons:
                break
            rho *= self.rho_factor
        # Valor objetivo original en el punto final
        f_val = float(self._objective_callable(*x_curr))
        point = {str(v): float(val) for v, val in zip(self.variables, x_curr)}
        converged = (history[-1]['max_violation'] < self.tol_cons) if history else True
        return SolverResult(
            method='PenaltyInequalities',
            point=point,
            objective_value=f_val,
            iterations=len(history),
            converged=converged and (best_res.converged if best_res else True),
            extra={'outer_history': history, 'final_max_violation': history[-1]['max_violation'] if history else 0.0}
        )
