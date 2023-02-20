from functools import reduce
import pulp
from .errors import assert_binary, assert_same_problem, MissingProblemReference


class Problem(pulp.LpProblem):
    def __init__(self, *args ,**kwargs):
        super().__init__(*args, **kwargs)
        self._synth_var_ix = 0

    def make_var(self, *args, **kwargs):
        v = Variable(*args, **kwargs, problem=self)
        if len(v.name) > 255:
            # conforming .lp files have variables less than 255 characters long
            v.name = v.name[:40] + '...' + v.name[-40:]
        return v

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
        return logical_or([self, other])

    def __and__(self, other):
        return logical_and([self, other])

    def __xor__(self, other):
        return logical_xor([self, other])

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
    tag = problem._synth_var()
    z = problem.make_var(f'abs({var.name})_{tag}', cat=pulp.LpInteger)
    problem.addConstraint(var <= z, f'abs_support_1_{tag}')
    problem.addConstraint(-z <= var, f'abs_support_2_{tag}')
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
    tag = problem._synth_var()
    y = problem.make_var('(NOT {})_{}'.format(x.name, tag), cat=pulp.LpBinary)
    problem.addConstraint(y == 1 - x, f"negate_support_{tag}")
    return y

def logical_and(xs):
    """
    produce a variable that represents x && y

    That variable can then be used in constraints and the objective func.
    """
    xs = list(xs)
    if len(xs) == 1:
        return xs[0]

    assert_same_problem(xs)
    for x in xs:
        assert_binary(x)

    model = x._problem

    tag = model._synth_var()
    name = '(' + '_AND_'.join(x.name for x in xs) + f')_{tag}'
    z = model.make_var(name, cat=pulp.LpBinary)
    model.addConstraint(z >= pulp.lpSum(xs) - len(xs) + 1, f"logical_and_support_{tag}")
    for ix, x in enumerate(xs):
        model.addConstraint(z <= x, f"logical_and_support_{ix}_{tag}")
    return z

def logical_or(xs):
    xs = list(xs)
    if len(xs) == 1:
        return xs[0]

    assert_same_problem(xs)
    for x in xs:
        assert_binary(x)

    model = x._problem

    tag = model._synth_var()
    name = '(' + '_OR_'.join(x.name for x in xs) + f')_{tag}'
    z = model.make_var(name, cat=pulp.LpBinary)
    model.addConstraint(z <= pulp.lpSum(xs), f"logical_or_support_{tag}")
    for ix, x in enumerate(xs):
        model.addConstraint(z >= x, f"logical_or_support_{ix}_{tag}")
    return z

def logical_xor(x: Variable, y: Variable):
    assert_same_problem([x, y])
    assert_binary(x)
    assert_binary(y)

    model = x._problem

    tag = model._synth_var()
    z = model.make_var('({} XOR {})_{}'.format(x.name, y.name, tag), cat=pulp.LpBinary)
    model.addConstraint(z <= x + y, f"logical_xor_support_0_{tag}")
    model.addConstraint(z >= x - y, f"logical_xor_support_1_{tag}")
    model.addConstraint(z >= y - x, f"logical_xor_support_2_{tag}")
    model.addConstraint(z <= 2 - x - y, f"logical_xor_support_3_{tag}")

    return z

def implies(x: Variable, y: Variable):
    assert_same_problem([x, y])
    assert_binary(x)
    assert_binary(y)

    model = x._problem

    tag = model._synth_var()
    z = model.make_var('({} implies {})_{}'.format(x.name, y.name, tag), cat=pulp.LpBinary)
    model.addConstraint(z <= 1 - x + y, f"implies_support_0_{tag}")
    model.addConstraint(z >= 1 - x, f"implies_support_1_{tag}")
    model.addConstraint(z >= y, f"implies_support_2_{tag}")
    return z


def minimum(xs, name=None):
    xs = list(xs)
    if len(xs) == 1:
        return xs[0]
    assert_same_problem(xs)
    model = xs[0]._problem
    tag = model._synth_var()
    m = model.make_var('{}_{}'.format(name or 'min', tag), cat=pulp.LpContinuous)
    for ix, x in enumerate(xs):
        model.addConstraint(m <= x, f"min_support_{ix}_{tag}")
    return m

def maximum(xs, name=None):
    xs = list(xs)
    if len(xs) == 1:
        return xs[0]
    assert_same_problem(xs)
    model = xs[0]._problem
    tag = model._synth_var()
    m = model.make_var('{}_{}'.format(name or 'max', tag), cat=pulp.LpContinuous)
    for ix, x in enumerate(xs):
        model.addConstraint(m >= x, f"max_support_{ix}_{tag}")
    return m

def lpSum(xs):
    assert_same_problem(xs)
    total = pulp.lpSum(xs)
    return AffineExpression(xs, xs[0]._problem)
