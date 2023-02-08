import pulp

class CitrusError(pulp.PulpError):
    pass

class NonBinaryVariableError(CitrusError):
    pass

class MissingProblemReference(CitrusError):
    pass

def assert_binary(var):
    if var.isBinary(): return
    if var.isConstant() and (var.value() == 1 or var.value() == 0): return
    raise NonBinaryVariableError(var.name)

def assert_same_problem(x, y):
    if x._problem is not y._problem:
        raise CitrusError('Variables must be associated with the same problem.')
    return x # return for use in reduce(assert_same_problem, (x, y, z))
