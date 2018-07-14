# Citrus

Intended to work like [PuLP](https://github.com/coin-or/pulp), but with a few convenience functions thrown in.

```
pip install citrus
```

# Comparisons

## ANDing two variables

```
# without citrus
import pulp
p = pulp.LpProblem('and example', pulp.LpMinimize)

x = pulp.LpVariable('x', cat=pulp.LpBinary)
y = pulp.LpVariable('y', cat=pulp.LpBinary)
x_and_y = pulp.LpVariable('x_and_y', cat=pulp.LpBinary)
model.addConstraint(x_and_y >= x + y - 1)
model.addConstraint(x_and_y <= x)
model.addConstraint(x_and_y <= y)
```

```
# with citrus
import citrus
p = citrus.Problem('and example', pulp.LpMinimize)

x = p.make_var('x', cat=pulp.LpBinary)
y = p.make_var('y', cat=pulp.LpBinary)
x_and_y = x & y
# alternatively, x_and_y = citrus.logical_and(x, y)
```

## ORing two variables

```
# without citrus
import pulp
p = pulp.LpProblem('or example', pulp.LpMinimize)

x = pulp.LpVariable('x', cat=pulp.LpBinary)
y = pulp.LpVariable('y', cat=pulp.LpBinary)
x_or_y = pulp.LpVariable('x_or_y', cat=pulp.LpBinary)
model.addConstraint(x_or_y <= x + y)
model.addConstraint(x_or_y >= x)
model.addConstraint(x_or_y >= y)
```

```
# with citrus
import citrus
p = citrus.Problem('or example', pulp.LpMinimize)

x = p.make_var('x', cat=pulp.LpBinary)
y = p.make_var('y', cat=pulp.LpBinary)
x_or_y = x | y
# alternatively, x_or_y = citrus.logical_or(x, y)
```

## Negating a variable

```
# without citrus
p = pulp.LpProblem('negation test', pulp.LpMinimize)

x = pulp.LpVariable('x', cat=pulp.LpBinary)
not_x = pulp.LpVariable('not_x', cat=pulp.LpBinary)
p.addConstraint(not_x == 1 - x)
```

```
# With citrus
import citrus
p = citrus.Problem('negation test', pulp.LpMinimize)

x = p.make_var('x', cat=pulp.LpBinary)
not_x = citrus.negate(x)
```

# Tips & Tricks

Sometimes, you'll have many variables that you want to AND or OR together:

```
p = citrus.Problem('vacation at some point', pulp.Maximize)

vacation_in_x_month = [
  p.make_var('vacation in ' + month, cat=pulp.LpBinary)
  for month in MONTHS
]

take_a_vacation = reduce(citrus.logical_or, vacation_in_x_month)
p.addConstraint(take_a_vacation)
```

# API

## Classes

- `Variable` is a subclass of `pulp.LpVariable`. It adds the following methods:
  - (classmethod) `from_lp_var`. Upgrade a `pulp.LpVariable` to a Variable.
  - `__or__(self, other)` Compute the `logical_or` of two binary `Variable`s
  - `__and__(self, other)` Compute the `logical_and` of two binary `Variable`s
  - `__and__(self, other)` Compute the `logical_and` of two binary `Variable`s

- `Problem` A subclass of `pulp.LpProblem`. It adds the following method
  - `make_var()` accepts same arguments as `pulp.LpVariable`, but produces a `Variable`

## Functions

- `negate(x: Variable)` Produce a new `Variable` with the opposite value of `x`.
- `logical_and(x: Variable, y: Variable)` Produce a new `Variable` constrained to take on the AND of `x` and `y`.
- `logical_or(x: Variable, y: Variable)` Produce a new `Variable` constrained to take on the OR of `x` and `y`.
- `logical_xor(x: Variable, y: Variable)` Produce a new `Variable` constrained to take on the XOR of `x` and `y`.
- `implies(x: Variable, y: Variable)` Produce a new variable constrained to take on the value of `x => y`
- `minimum(*xs)` Produce a new `Variable` that can be no larger than the smallest in `xs`
- `maximum(*xy)` Produce a new `Variable` that can be no smaller than the largest in `xs`
