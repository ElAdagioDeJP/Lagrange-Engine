from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List
import sympy as sp
import numpy as np

@dataclass
class SolverResult:
    method: str
    point: Dict[str, float]
    objective_value: float
    iterations: int
    converged: bool
    extra: Dict[str, Any]

class OptimizationEngine(ABC):
    def __init__(self, objective: sp.Expr, variables: List[sp.Symbol]):
        self.objective = objective
        self.variables = variables
        self._objective_callable = sp.lambdify(variables, objective, 'numpy')

    @abstractmethod
    def solve(self, start: np.ndarray) -> SolverResult:
        ...
