from typing import Hashable, Iterable
import warnings
import hashlib
import inspect
import pulp
from .errors import assert_binary, assert_same_problem, MissingProblemReference

class NameMapping:
    def __init__(self):
        self._long_to_short = {}
        self._short_to_long = {}
    def get_short_name(self, var):
        return self._short_to_long[var]
    def get_long_name(self, name):
        return self._long_to_short[name]
    def create_short_name(self, long_name, prefix='x_'):
        short_name = self._long_to_short.get(long_name, None)
        if short_name is None:
            short_name = prefix + hashlib.sha1(repr(long_name).encode('utf-8')).hexdigest()
            self.add(long=long_name, short=short_name)
        else:
            warnings.warn(f"Long name {long_name} already has a short name {short_name}")
        return short_name
    def add(self, *, long, short):
        self._long_to_short[long] = short
        self._short_to_long[short] = long
    def remove(self, *, short):
        long = self._short_to_long.get(short, None)
        if long is not None:
            del self._long_to_short[long]
            del self._short_to_long[short]
    def __iter__(self):
        return iter(self._long_to_short.items())

VAR_SIGNATURE = inspect.signature(pulp.LpVariable)
DICTS_SIGNATURE = inspect.signature(pulp.LpVariable.dicts)

class Problem(pulp.LpProblem):
    def __init__(self, *args ,**kwargs):
        super().__init__(*args, **kwargs)
        self._synth_var_ix = 0
        self.name_mapping = NameMapping()

    def make_var(self, *args, **kwargs):
        return Variable(*args, **kwargs, problem=self)

    def _synth_var(self):
        name = str(self._synth_var_ix)
        self._synth_var_ix += 1
        return name

    def dicts(self, indices: Iterable[Hashable], **kwargs):
        d = {}
        for k in indices:
            d[k] = self.make_var(long_name=k, **kwargs)
        return d

    def addConstraint(self, constraint, name=None):
        if name is None:
            name = self._synth_var()
        short = self.name_mapping.create_short_name(name, prefix='constraint_')
        super().addConstraint(constraint, name=short)

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
    def __init__(self, long_name, **kwargs):
        self._problem = kwargs.pop('problem')
        self.long_name = long_name
        short = self._problem.name_mapping.create_short_name(long_name)
        super().__init__(**kwargs, name=short)

    def __or__(self, other):
        return logical_or([self, other])

    def __and__(self, other):
        return logical_and([self, other])

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
    return AffineExpression(total, xs[0]._problem)
