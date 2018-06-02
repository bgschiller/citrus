import pulp
from .core import Problem, negate

def test_that_negate_produces_negated_variable():
    p = Problem('negation test', pulp.LpMinimize)
    x = p.make_var('x', cat=pulp.LpBinary)
    y = negate(x)
    p.setObjective(2 * x + y)
    p.solve()
    assert pulp.LpStatus[p.status] == 'Optimal'
    assert x.value() == 0
    assert y.value() == 1

# test that logical_and produces constrained value

# test that logical_or produces constrained value

# test that negate throws on non binary variable

# test that logical_and, logical_or throw on non-binary variable

# test Variable.from_lp_var still works in constraints.

# test that minimum is truly the min
