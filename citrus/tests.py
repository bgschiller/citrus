import pytest
import pulp
from .core import Problem, negate, logical_and, logical_or
from .errors import NonBinaryVariableError, CitrusError

def test_that_negate_produces_negated_variable():
    p = Problem('negation test', pulp.LpMinimize)
    x = p.make_var('x', cat=pulp.LpBinary)
    y = negate(x)
    z = negate(y)
    p.setObjective(2 * x + y)
    p.solve()
    assert pulp.LpStatus[p.status] == 'Optimal'
    assert x.value() == 0
    assert y.value() == 1
    assert z.value() == 0

def test_that_logical_and_produces_constrained_value():
    p = Problem('logical_and test', pulp.LpMinimize)
    t1 = p.make_var('t1', cat=pulp.LpBinary)
    t2 = p.make_var('t2', cat=pulp.LpBinary)
    f1 = p.make_var('f1', cat=pulp.LpBinary)
    f2 = p.make_var('f2', cat=pulp.LpBinary)

    tt = logical_and(t1, t2)
    tf = logical_and(t1, f1)
    ft = logical_and(f1, t1)
    ff = logical_and(f1, f2)
    p.addConstraint(t1 == 1)
    p.addConstraint(t2 == 1)
    p.addConstraint(f1 == 0)
    p.addConstraint(f2 == 0)
    p.solve()
    assert pulp.LpStatus[p.status] == 'Optimal'
    assert tt.value() == 1
    assert tf.value() == 0
    assert ft.value() == 0
    assert ff.value() == 0

def test_that_anding_with_itself_is_okay():
    p = Problem('anding with self', pulp.LpMinimize)
    t = p.make_var('t', cat=pulp.LpBinary)
    f = p.make_var('f', cat=pulp.LpBinary)

    tt = logical_and(t, t)
    tf = logical_and(t, f)
    ft = logical_and(f, t)
    ff = logical_and(f, f)
    p.addConstraint(t == 1)
    p.addConstraint(f == 0)
    p.solve()
    assert pulp.LpStatus[p.status] == 'Optimal'
    assert tt.value() == 1
    assert tf.value() == 0
    assert ft.value() == 0
    assert ff.value() == 0

def test_that_logical_or_produces_constrained_value():
    p = Problem('logical_or tests', pulp.LpMinimize)
    t = p.make_var('t', cat=pulp.LpBinary)
    f = p.make_var('f', cat=pulp.LpBinary)

    tt = logical_or(t, t)
    tf = logical_or(t, f)
    ft = logical_or(f, t)
    ff = logical_or(f, f)
    p.addConstraint(t == 1)
    p.addConstraint(f == 0)
    p.solve()
    assert pulp.LpStatus[p.status] == 'Optimal'
    assert tt.value() == 1
    assert tf.value() == 1
    assert ft.value() == 1
    assert ff.value() == 0

def test_that_funcs_throws_on_non_binary_variable():
    p = Problem('problem', pulp.LpMinimize)
    x = p.make_var('x', cat=pulp.LpInteger)
    y = p.make_var('y', cat=pulp.LpInteger)

    with pytest.raises(NonBinaryVariableError):
        logical_and(x, y)
    with pytest.raises(NonBinaryVariableError):
        logical_or(x, y)

def test_that_vars_from_diff_problems_raise_error():
    a = Problem('problem a', pulp.LpMinimize)
    b = Problem('problem b', pulp.LpMinimize)
    x = a.make_var('x', cat=pulp.LpBinary)
    y = b.make_var('y', cat=pulp.LpBinary)

    with pytest.raises(CitrusError):
        logical_or(x, y)
    with pytest.raises(CitrusError):
        logical_and(x, y)

# test Variable.from_lp_var still works in constraints.

# test that minimum is truly the min
