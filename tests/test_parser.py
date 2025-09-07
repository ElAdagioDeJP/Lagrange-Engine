from core.function_parser import FunctionParser

def test_parse_simple():
    prob = FunctionParser.parse_problem('x**2 + y**2', 'x y', 'x + y = 0')
    assert len(prob.variables) == 2
    assert len(prob.equality_constraints) == 1
