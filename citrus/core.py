from functools import reduce
import pulp
from .errors import assert_binary, assert_same_problem


class Problem(pulp.LpProblem):
    def __init__(self, *args ,**kwargs):
        super().__init__(*args, **kwargs)
        self._synth_var_ix = 0

    def make_var(self, *args, **kwargs):
        return Variable(*args, **kwargs, problem=self)

    def _synth_var(self):
        name = str(self._synth_var_ix)
        self._synth_var_ix += 1
        return name

    def dicts(self, *args, **kwargs):
        ds = pulp.LpVariable.dicts(*args, **kwargs)
        return self._walk_dicts(ds)

    def _walk_dicts(self, ds):
        if isinstance(ds, pulp.LpVariable):
            return Variable.from_lp_var(ds, self)
        if isinstance(ds, dict):
            return { k: self._walk_dicts(v) for k, v in ds.items() }
        raise ValueError('Expected a dict or LpVariable. received {}'.format(ds))

class Variable(pulp.LpVariable):
    def __init__(self, *args, **kwargs):
        self._problem = kwargs.pop('problem')
        super().__init__(*args, **kwargs)

    @classmethod
    def from_lp_var(cls, lp_var: pulp.LpVariable, problem: Problem):
        lp_var.__class__ = cls
        lp_var._problem = problem
        return lp_var

    def __or__(self, other):
        return logical_or(self, other)

    def __and__(self, other):
        return logical_and(self, other)

    def __xor__(self, other):
        return logical_xor(self, other)


def negate(x: Variable):
    assert_binary(x)
    problem = x._problem
    y = problem.make_var('(NOT {})_{}'.format(x.name, problem._synth_var()), cat=pulp.LpBinary)
    problem.addConstraint(y == 1 - x)
    return y

def logical_and(x: Variable, y: Variable):
    """
    produce a variable that represents x && y

    That variable can then be used in constraints and the objective func.
    """
    assert_same_problem(x, y)
    assert_binary(x)
    assert_binary(y)

    model = x._problem

    z = model.make_var('({} AND {})_{}'.format(x.name, y.name, model._synth_var()), cat=pulp.LpBinary)
    model.addConstraint(z >= x + y - 1)
    model.addConstraint(z <= x)
    model.addConstraint(z <= y)
    return z

def logical_or(x: Variable, y: Variable):
    assert_same_problem(x, y)
    assert_binary(x)
    assert_binary(y)

    model = x._problem

    z = model.make_var('({} AND {})_{}'.format(x.name, y.name, model._synth_var()), cat=pulp.LpBinary)
    model.addConstraint(z <= x + y)
    model.addConstraint(z >= x)
    model.addConstraint(z >= y)
    return z

def logical_xor(x: Variable, y: Variable):
    assert_same_problem(x, y)
    assert_binary(x)
    assert_binary(y)

    model = x._problem

    z = model.make_var('({} XOR {})_{}'.format(x.name, y.name, model._synth_var()), cat=pulp.LpBinary)
    model.addConstraint(z <= x + y)
    model.addConstraint(z >= x - y)
    model.addConstraint(z >= y - x)
    model.addConstraint(z <= 2 - x - y)

    return z

def implies(x: Variable, y: Variable):
    assert_same_problem(x, y)
    assert_binary(x)
    assert_binary(y)

    model = x._problem

    z = model.make_var('({} implies {})_{}'.format(x.name, y.name, model._synth_var()), cat=pulp.LpBinary)
    model.addConstraint(z <= 1 - x + y)
    model.addConstraint(z >= 1 - x)
    model.addConstraint(z >= y)
    return z


def minimum(*xs: Variable, name=None):
    if len(xs) == 1:
        return xs[0]
    reduce(assert_same_problem, xs)
    model = xs[0]._problem
    m = model.make_var('{}_{}'.format(name or 'min', model._synth_var()), cat=pulp.LpContinuous)
    for x in xs:
        model.addConstraint(m <= x)
    return m

def maximum(*xs: Variable, name=None):
    if len(xs) == 1:
        return xs[0]
    reduce(assert_same_problem, xs)
    model = xs[0]._problem
    m = model.make_var('{}_{}'.format(name or 'max', model._synth_var()), cat=pulp.LpContinuous)
    for x in xs:
        model.addConstraint(m >= x)
    return m
