from __future__ import annotations
from typing import List, Dict, Any
import sympy as sp
import numpy as np
from .base import OptimizationEngine, SolverResult

class GradientDescentSolver(OptimizationEngine):
    def __init__(self, objective: sp.Expr, variables: List[sp.Symbol], learning_rate: float = 0.05, max_iter: int = 500, tol: float = 1e-6, adaptive_lr: bool = True):
        super().__init__(objective, variables)
        self.lr = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.adaptive_lr = adaptive_lr
        self.gradient = [sp.diff(objective, v) for v in variables]
        self._grad_callable = sp.lambdify(variables, self.gradient, 'numpy')

    def solve(self, start):  # type: ignore[override]
        x = np.array(start, dtype=float)
        history = []
        lr = self.lr
        prev_obj = float('inf')
        
        for it in range(1, self.max_iter + 1):
            grad_vals = np.array(self._grad_callable(*x), dtype=float)
            grad_norm = np.linalg.norm(grad_vals)
            current_obj = float(self._objective_callable(*x))
            
            history.append((it, x.copy(), grad_norm, current_obj))
            
            if grad_norm < self.tol:
                return SolverResult(
                    method='GradientDescent',
                    point={str(v): float(val) for v, val in zip(self.variables, x)},
                    objective_value=current_obj,
                    iterations=it,
                    converged=True,
                    extra={'history': history, 'final_lr': lr}
                )
            
            # Ajuste adaptativo de learning rate
            if self.adaptive_lr and it > 1:
                if current_obj > prev_obj:
                    lr *= 0.8  # Reducir si empeora
                else:
                    lr *= 1.01  # Aumentar ligeramente si mejora
                lr = max(1e-8, min(lr, 1.0))  # Limitar rango
            
            x = x - lr * grad_vals
            prev_obj = current_obj
            
        return SolverResult(
            method='GradientDescent',
            point={str(v): float(val) for v, val in zip(self.variables, x)},
            objective_value=float(self._objective_callable(*x)),
            iterations=self.max_iter,
            converged=False,
            extra={'history': history, 'final_lr': lr}
        )
