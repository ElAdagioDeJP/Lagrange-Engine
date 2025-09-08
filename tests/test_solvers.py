import sympy as sp
import numpy as np
from core.function_parser import FunctionParser
from solvers.gradient_descent import GradientDescentSolver
from solvers.lagrange import LagrangeSolver
from solvers.unconstrained import UnconstrainedMinimizer
from solvers.penalty import PenaltyInequalitySolver

def test_unconstrained_quadratic():
    x, y = sp.symbols('x y')
    f = (x-2)**2 + (y+3)**2
    solver = UnconstrainedMinimizer(f, [x, y])
    res = solver.solve([0, 0])
    assert res.converged
    assert abs(res.point['x'] - 2) < 1e-4
    assert abs(res.point['y'] + 3) < 1e-4


def test_gradient_descent_quadratic():
    x, = sp.symbols('x', real=True)
    f = (x-5)**2
    solver = GradientDescentSolver(f, [x], learning_rate=0.2, max_iter=500)
    res = solver.solve([0])
    assert res.converged
    assert abs(res.point['x'] - 5) < 1e-3


def test_lagrange_linear_constraint():
    # Min x^2 + y^2 s.a. x + y = 0 -> solución (0,0)
    prob = FunctionParser.parse_problem('x**2 + y**2', 'x y', 'x + y = 0')
    solver = LagrangeSolver(prob.objective_expr, prob.variables, prob.equality_constraints)
    res = solver.solve([1, -1])
    assert abs(res.point['x']) < 1e-6
    assert abs(res.point['y']) < 1e-6


def test_penalty_inequality():
    # Min (x-1)^2 con restricción x >= 0  (ya cumple en el óptimo)
    prob = FunctionParser.parse_problem('(x-1)**2', 'x', 'x >= 0')
    solver = PenaltyInequalitySolver(prob.objective_expr, prob.variables, prob.inequality_constraints)
    res = solver.solve([5])
    assert res.converged
    assert abs(res.point['x'] - 1) < 1e-3
