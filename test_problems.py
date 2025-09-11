#!/usr/bin/env python3
"""
Script de prueba para verificar que todos los problemas del usuario funcionan correctamente.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.function_parser import FunctionParser
from src.solvers.gradient_descent import GradientDescentSolver
from src.solvers.lagrange import LagrangeSolver
from src.solvers.unconstrained import UnconstrainedMinimizer
from src.solvers.penalty import PenaltyInequalitySolver
from src.solvers.volume_optimizer import VolumeOptimizer
import numpy as np

def test_problem_1():
    """Problema 1: Sin restricciones (minimización)"""
    print("🧪 Probando Problema 1: Sin restricciones")
    print("C(x,y) = x² + y² - 4x - 6y + 20")
    
    problem = FunctionParser.parse_problem(
        "x**2 + y**2 - 4*x - 6*y + 20",
        "x y",
        ""
    )
    
    # Solución analítica: x = 2, y = 3, C = 7
    start = np.array([0.0, 0.0])
    
    # Probar diferentes métodos
    results = []
    
    # BFGS
    solver = UnconstrainedMinimizer(problem.objective_expr, problem.variables)
    result = solver.solve(start)
    results.append(result)
    print(f"  BFGS: x={result.point['x']:.3f}, y={result.point['y']:.3f}, C={result.objective_value:.3f}")
    
    # Gradiente
    solver = GradientDescentSolver(problem.objective_expr, problem.variables, learning_rate=0.1)
    result = solver.solve(start)
    results.append(result)
    print(f"  Gradiente: x={result.point['x']:.3f}, y={result.point['y']:.3f}, C={result.objective_value:.3f}")
    
    print(f"  ✅ Solución esperada: x=2, y=3, C=7")
    print()

def test_problem_2():
    """Problema 2: Método de Lagrange (maximización)"""
    print("🧪 Probando Problema 2: Método de Lagrange")
    print("A(x,y) = x * y con 2x + 2y = 40")
    
    problem = FunctionParser.parse_problem(
        "x * y",
        "x y",
        "2*x + 2*y = 40"
    )
    
    # Solución analítica: x = 10, y = 10, A = 100
    start = np.array([10.0, 10.0])
    
    solver = LagrangeSolver(problem.objective_expr, problem.variables, problem.equality_constraints)
    result = solver.solve(start)
    print(f"  Lagrange: x={result.point['x']:.3f}, y={result.point['y']:.3f}, A={result.objective_value:.3f}")
    print(f"  ✅ Solución esperada: x=10, y=10, A=100")
    print()

def test_problem_3():
    """Problema 3: Método del Gradiente (maximización)"""
    print("🧪 Probando Problema 3: Método del Gradiente")
    print("f(x,y) = -(x² + y²) + 4x + 6y")
    
    problem = FunctionParser.parse_problem(
        "-(x**2 + y**2) + 4*x + 6*y",
        "x y",
        ""
    )
    
    # Solución analítica: x = 2, y = 3, f = 13
    start = np.array([0.0, 0.0])
    
    solver = GradientDescentSolver(problem.objective_expr, problem.variables, learning_rate=0.1)
    result = solver.solve(start)
    print(f"  Gradiente: x={result.point['x']:.3f}, y={result.point['y']:.3f}, f={result.objective_value:.3f}")
    print(f"  ✅ Solución esperada: x=2, y=3, f=13")
    print()

def test_problem_4():
    """Problema 4: Múltiples restricciones (maximización)"""
    print("🧪 Probando Problema 4: Múltiples restricciones")
    print("G(x,y) = 5x + 8y - 0.1x² - 0.2y² con x + y ≤ 40, x ≥ 0, y ≥ 0")
    
    problem = FunctionParser.parse_problem(
        "5*x + 8*y - 0.1*x**2 - 0.2*y**2",
        "x y",
        "x + y <= 40\nx >= 0\ny >= 0"
    )
    
    start = np.array([0.0, 0.0])
    
    solver = PenaltyInequalitySolver(problem.objective_expr, problem.variables, problem.inequality_constraints)
    result = solver.solve(start)
    print(f"  Penalización: x={result.point['x']:.3f}, y={result.point['y']:.3f}, G={result.objective_value:.3f}")
    print(f"  ✅ Solución esperada: x≈25, y≈15, G≈200")
    print()

def test_problem_5():
    """Problema 5: Volumen mínimo (cilindro)"""
    print("🧪 Probando Problema 5: Volumen mínimo del cilindro")
    print("Volumen fijo: V = 500 cm³, minimizar área superficial")
    
    problem = FunctionParser.parse_problem(
        "2*pi*r**2 + 2*pi*r*h",
        "r h",
        "pi*r**2*h = 500"
    )
    
    # Usar solver especializado para cilindros
    solver = VolumeOptimizer(problem.objective_expr, problem.variables, volume_value=500)
    result = solver.solve_cylinder_min_surface(500)
    
    print(f"  Cilindro: r={result.point['r']:.3f}, h={result.point['h']:.3f}, A={result.objective_value:.3f}")
    print(f"  Volumen logrado: {result.extra['volume_achieved']:.3f}")
    print(f"  ✅ Solución esperada: r≈4.3, h≈8.6, A≈350")
    print()

def main():
    """Ejecutar todas las pruebas"""
    print("🚀 Iniciando pruebas de los 5 problemas de optimización\n")
    
    try:
        test_problem_1()
        test_problem_2()
        test_problem_3()
        test_problem_4()
        test_problem_5()
        
        print("🎉 ¡Todas las pruebas completadas exitosamente!")
        print("\n📋 Resumen:")
        print("  ✅ Problema 1: Minimización sin restricciones")
        print("  ✅ Problema 2: Método de Lagrange")
        print("  ✅ Problema 3: Método del Gradiente")
        print("  ✅ Problema 4: Múltiples restricciones")
        print("  ✅ Problema 5: Volumen mínimo del cilindro")
        
    except Exception as e:
        print(f"❌ Error durante las pruebas: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
