from functools import reduce
import pulp
from .errors import assert_binary, assert_same_problem, MissingProblemReference


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

class AffineExpression(pulp.LpAffineExpression):
    def __init__(self, *args, **kwargs):
        if 'problem' in kwargs:
            problem = kwargs.pop('problem')
        elif len(args) > 0 and isinstance(args[0], (Variable, AffineExpression)):
            problem = args[0]._problem
        else:
            raise MissingProblemReference('Cannot create AffineExpression without a reference to the problem')
        super().__init__(*args, **kwargs)
        self._problem = problem

    def __abs__(self):
        return abs_value(self)

    def copy(self):
        return AffineExpression(self)

    def emptyCopy(self):
        return AffineExpression(problem=self._problem)


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

    def __abs__(self):
        return abs_value(self)

    # -- copied from LpVariable, but with a different constructor
    def __neg__(self):
        return - AffineExpression(self)

    def __add__(self, other):
        return AffineExpression(self) + other

    def __radd__(self, other):
        return AffineExpression(self) + other

    def __sub__(self, other):
        return AffineExpression(self) - other

    def __rsub__(self, other):
        return other - AffineExpression(self)

    def __mul__(self, other):
        return AffineExpression(self) * other

    def __rmul__(self, other):
        return AffineExpression(self) * other

    def __div__(self, other):
        return AffineExpression(self)/other

def abs_value(var: Variable) -> Variable:
    problem = var._problem
    z = problem.make_var(f'abs({var.name})_{problem._synth_var()}', cat=pulp.LpInteger)
    problem.addConstraint(var <= z, f'{var.name} <= abs({var.name}) _{problem._synth_var()}')
    problem.addConstraint(-z <= var, f'- abs({var.name}) <= {var.name} _{problem._synth_var()}')
    return z

def prefer_between(x: Variable, a: int, b: int) -> Variable:
    """
    Produces a variable that is
     - 0 when x lies within [a, b]
     - negative otherwise, and more negative with distance from [a, b]
    """
    return (b - a) - abs_value(x - a) - abs_value(x - b)


def negate(x: Variable):
    assert_binary(x)
    problem = x._problem
    y = problem.make_var('(NOT {})_{}'.format(x.name, problem._synth_var()), cat=pulp.LpBinary)
    problem.addConstraint(y == 1 - x)
    return y

def logical_and(*xs):
    """
    produce a variable that represents x && y

    That variable can then be used in constraints and the objective func.
    """
    if len(xs) == 1:
        return xs[0]

    assert_same_problem(*xs)
    for x in xs:
        assert_binary(x)

    model = x._problem

    name = '(' + '_AND_'.join(x.name for x in xs) + f')_{model._synth_var()}'
    z = model.make_var(name, cat=pulp.LpBinary)
    model.addConstraint(z >= pulp.lpSum(xs) - len(xs) + 1)
    for x in xs:
        model.addConstraint(z <= x)
    return z

def logical_or(*xs):
    if len(xs) == 1:
        return xs[0]

    assert_same_problem(*xs)
    for x in xs:
        assert_binary(x)

    model = x._problem

    name = '(' + '_OR_'.join(x.name for x in xs) + f')_{model._synth_var()}'
    z = model.make_var(name, cat=pulp.LpBinary)
    model.addConstraint(z <= pulp.lpSum(xs))
    for x in xs:
        model.addConstraint(z >= x)
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
    assert_same_problem(*xs)
    model = xs[0]._problem
    m = model.make_var('{}_{}'.format(name or 'min', model._synth_var()), cat=pulp.LpContinuous)
    for x in xs:
        model.addConstraint(m <= x)
    return m

def maximum(*xs: Variable, name=None):
    if len(xs) == 1:
        return xs[0]
    assert_same_problem(*xs)
    model = xs[0]._problem
    m = model.make_var('{}_{}'.format(name or 'max', model._synth_var()), cat=pulp.LpContinuous)
    for x in xs:
        model.addConstraint(m >= x)
    return m
