from __future__ import annotations
from typing import List, Dict, Any
import sympy as sp
import numpy as np
from .base import OptimizationEngine, SolverResult


class VolumeOptimizer(OptimizationEngine):
    """Solver especializado para problemas de optimización de volumen.
    
    Resuelve problemas como:
    - Minimizar área superficial de un cilindro con volumen fijo
    - Maximizar volumen con restricciones de área superficial
    - Optimización de formas geométricas con restricciones de volumen
    """
    
    def __init__(self, objective: sp.Expr, variables: List[sp.Symbol], 
                 volume_constraint: sp.Expr = None, volume_value: float = None):
        super().__init__(objective, variables)
        self.volume_constraint = volume_constraint
        self.volume_value = volume_value
        
        # Detectar si es un problema de cilindro
        self.is_cylinder = self._detect_cylinder_problem()
        
        if self.is_cylinder:
            self._setup_cylinder_solver()
    
    def _detect_cylinder_problem(self) -> bool:
        """Detecta si el problema involucra un cilindro (variables r, h)"""
        var_names = [str(v) for v in self.variables]
        return 'r' in var_names and 'h' in var_names and len(self.variables) == 2
    
    def _setup_cylinder_solver(self):
        """Configura el solver para problemas de cilindro"""
        r, h = self.variables[0], self.variables[1]
        
        # Fórmulas del cilindro
        self.volume_formula = sp.pi * r**2 * h
        self.surface_area_formula = 2 * sp.pi * r**2 + 2 * sp.pi * r * h
        
        # Derivadas para análisis
        self.dV_dr = sp.diff(self.volume_formula, r)
        self.dV_dh = sp.diff(self.volume_formula, h)
        self.dA_dr = sp.diff(self.surface_area_formula, r)
        self.dA_dh = sp.diff(self.surface_area_formula, h)
    
    def solve_cylinder_min_surface(self, volume: float) -> SolverResult:
        """Resuelve el problema de minimizar área superficial con volumen fijo"""
        if not self.is_cylinder:
            raise ValueError("Este método solo funciona para problemas de cilindro")
        
        r, h = self.variables[0], self.variables[1]
        
        # Solución analítica para cilindro con volumen fijo
        # h = V / (π * r²)
        # A = 2πr² + 2πr * h = 2πr² + 2V/r
        # dA/dr = 4πr - 2V/r² = 0
        # 4πr = 2V/r²
        # 4πr³ = 2V
        # r³ = V/(2π)
        # r = (V/(2π))^(1/3)
        
        r_opt = (volume / (2 * sp.pi))**(1/3)
        h_opt = volume / (sp.pi * r_opt**2)
        
        point = {str(r): float(r_opt), str(h): float(h_opt)}
        surface_area = float(self.surface_area_formula.subs({r: r_opt, h: h_opt}))
        
        return SolverResult(
            method='CylinderAnalytical',
            point=point,
            objective_value=surface_area,
            iterations=1,
            converged=True,
            extra={
                'volume': volume,
                'radius': float(r_opt),
                'height': float(h_opt),
                'surface_area': surface_area,
                'volume_achieved': float(self.volume_formula.subs({r: r_opt, h: h_opt}))
            }
        )
    
    def solve_cylinder_max_volume(self, surface_area: float) -> SolverResult:
        """Resuelve el problema de maximizar volumen con área superficial fija"""
        if not self.is_cylinder:
            raise ValueError("Este método solo funciona para problemas de cilindro")
        
        r, h = self.variables[0], self.variables[1]
        
        # Solución analítica para cilindro con área superficial fija
        # A = 2πr² + 2πrh = constante
        # h = (A - 2πr²) / (2πr)
        # V = πr²h = πr² * (A - 2πr²) / (2πr) = r(A - 2πr²) / 2
        # V = (Ar - 2πr³) / 2
        # dV/dr = (A - 6πr²) / 2 = 0
        # A = 6πr²
        # r = sqrt(A/(6π))
        
        r_opt = sp.sqrt(surface_area / (6 * sp.pi))
        h_opt = (surface_area - 2 * sp.pi * r_opt**2) / (2 * sp.pi * r_opt)
        
        point = {str(r): float(r_opt), str(h): float(h_opt)}
        volume = float(self.volume_formula.subs({r: r_opt, h: h_opt}))
        
        return SolverResult(
            method='CylinderAnalytical',
            point=point,
            objective_value=volume,
            iterations=1,
            converged=True,
            extra={
                'surface_area': surface_area,
                'radius': float(r_opt),
                'height': float(h_opt),
                'volume': volume,
                'surface_area_achieved': float(self.surface_area_formula.subs({r: r_opt, h: h_opt}))
            }
        )
    
    def solve(self, start):  # type: ignore[override]
        """Método principal de resolución"""
        if self.is_cylinder and self.volume_value is not None:
            # Si es un problema de cilindro con volumen fijo, usar solución analítica
            return self.solve_cylinder_min_surface(self.volume_value)
        else:
            # Para otros casos, usar el solver genérico
            from .unconstrained import UnconstrainedMinimizer
            solver = UnconstrainedMinimizer(self.objective, self.variables)
            return solver.solve(start)
    
    def get_cylinder_analysis(self, point: Dict[str, float]) -> Dict[str, Any]:
        """Proporciona análisis detallado para un cilindro"""
        if not self.is_cylinder:
            return {}
        
        r_val = point.get('r', 0)
        h_val = point.get('h', 0)
        
        r, h = self.variables[0], self.variables[1]
        
        volume = float(self.volume_formula.subs({r: r_val, h: h_val}))
        surface_area = float(self.surface_area_formula.subs({r: r_val, h: h_val}))
        
        # Relación óptima: h = 2r para área mínima con volumen fijo
        optimal_ratio = 2.0
        actual_ratio = h_val / r_val if r_val > 0 else 0
        
        return {
            'volume': volume,
            'surface_area': surface_area,
            'radius': r_val,
            'height': h_val,
            'optimal_ratio': optimal_ratio,
            'actual_ratio': actual_ratio,
            'is_optimal_ratio': abs(actual_ratio - optimal_ratio) < 0.01,
            'efficiency': optimal_ratio / actual_ratio if actual_ratio > 0 else 0
        }
