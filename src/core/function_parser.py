from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Callable, Any
import sympy as sp


@dataclass
class ParsedProblem:
    variables: List[sp.Symbol]
    objective_expr: sp.Expr
    equality_constraints: List[sp.Eq]
    inequality_constraints: List[sp.Rel]

    def objective_callable(self) -> Callable:
        return sp.lambdify(self.variables, self.objective_expr, 'numpy')


class FunctionParser:
    """Parse objective functions and constraints from user-provided text.

    Expected syntax examples:
        f(x,y) = x**2 + 3*y - sp.sin(x)
        constraints:
            x + y = 10
            x**2 <= 25
    """

    @staticmethod
    def parse_variables(var_text: str) -> List[sp.Symbol]:
        names = [v.strip() for v in var_text.replace(',', ' ').split() if v.strip()]
        if not names:
            raise ValueError("No variables provided")
        return [sp.Symbol(n) for n in names]

    @staticmethod
    def parse_objective(expr_text: str, variables: List[sp.Symbol]) -> sp.Expr:
        local_dict = {str(v): v for v in variables}
        local_dict.update({n: getattr(sp, n) for n in dir(sp) if not n.startswith('_')})
        return sp.sympify(expr_text, locals=local_dict)

    @staticmethod
    def parse_constraints(lines: List[str], variables: List[sp.Symbol]):
        eqs: List[sp.Eq] = []
        ineqs: List[sp.Rel] = []
        local_dict = {str(v): v for v in variables}
        local_dict.update({n: getattr(sp, n) for n in dir(sp) if not n.startswith('_')})

        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            if '==' in line:
                left, right = line.split('==', 1)
                eqs.append(sp.Eq(sp.sympify(left, locals=local_dict), sp.sympify(right, locals=local_dict)))
            elif '=' in line and '<' not in line and '>' not in line:
                left, right = line.split('=', 1)
                eqs.append(sp.Eq(sp.sympify(left, locals=local_dict), sp.sympify(right, locals=local_dict)))
            elif '<=' in line:
                left, right = line.split('<=', 1)
                ineqs.append(sp.Le(sp.sympify(left, locals=local_dict), sp.sympify(right, locals=local_dict)))
            elif '>=' in line:
                left, right = line.split('>=', 1)
                ineqs.append(sp.Ge(sp.sympify(left, locals=local_dict), sp.sympify(right, locals=local_dict)))
            elif '<' in line:
                left, right = line.split('<', 1)
                ineqs.append(sp.Lt(sp.sympify(left, locals=local_dict), sp.sympify(right, locals=local_dict)))
            elif '>' in line:
                left, right = line.split('>', 1)
                ineqs.append(sp.Gt(sp.sympify(left, locals=local_dict), sp.sympify(right, locals=local_dict)))
            else:
                raise ValueError(f"La restricción '{line}' no es válida. Asegúrese de que es una ecuación o inecuación (ej: 'g(x)=0', 'h(x)<=0').")
        return eqs, ineqs

    @classmethod
    def parse_problem(cls, objective: str, variables: str, constraints: str) -> ParsedProblem:
        vars_ = cls.parse_variables(variables)
        obj_expr = cls.parse_objective(objective, vars_)
        eqs, ineqs = cls.parse_constraints(constraints.splitlines(), vars_)
        return ParsedProblem(vars_, obj_expr, eqs, ineqs)
