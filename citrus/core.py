import pulp

def logical_or(x, y):
    pass

def logical_and(x, y):
    pass

def minimum(*xs, name=None):
    pass

class Problem(pulp.LpProblem):
    def __init__(self, *args ,**kwargs):
        super().__init__(*args, **kwargs)
        self._synth_var_ix = 0

    def make_var(self, *args, **kwargs):
        return Variable(*args, **kwargs, problem=self)

    def synthetic_var_name(self):
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

def negate(x: Variable):
    assert x.isBinary(), 'Only binary variables can be negated'
    problem = x._problem
    y = pulp.LpVariable('(NOT {})_{}'.format(x.name, problem.synthetic_var_name()), cat=pulp.LpBinary)
    problem.addConstraint(y == 1 - x, 'constraint_{}'.format(problem.synthetic_var_name()))
    return y
