from sympy import Mul

from sympy.printing.lambdarepr import NumPyPrinter
from sympy.printing.precedence import precedence

class MatrixNumPyPrinter(NumPyPrinter):
    """
    Print an expression for numpy assuming the variables are matrices

    Prints inversions as solve() and multiplication of nonconstants as @.

    """
    def _print_Mul(self, expr):
        prec = precedence(expr)

        pows = [i for i in expr.args if i.is_Pow and i.exp < 0]
        if len(pows) > 1:
            raise NotImplementedError("Need exactly one inverted Pow, not %s" % len(pows))

        if not pows:
            consts = [self._print(self.parenthesize(i, prec)) for i in expr.args if i.is_Number]
            rest = [self._print(self.parenthesize(i, prec)) for i in expr.args if not i.is_Number]
            if consts and rest:
                return '*'.join(consts) + '*' + '@'.join(rest)
            else:
                return '*'.join(consts) + '@'.join(rest)

        [pow] = pows

        rest = Mul(*[i for i in expr.args if i != pow])

        return 'solve(%s, %s)' % (self._print(1/pow), self._print(rest))

    def _print_Float(self, expr):
        return 'autoeye(%s)' % super()._print_Float(expr)

    def _print_Pow(self, expr):
        if expr.exp.is_Integer and expr.exp > 1:
            return 'matrix_power(%s, %s)' % (self._print(expr.base), expr.exp)
        return super()._print_Pow(expr)

    def _print_ImaginaryUnit(self, expr):
        return 'autoeye(1j)'

class autoeye:
    __array_priority__ = 11

    def __init__(self, coeff=1):
        self.coeff = coeff

    def eval(self, shape, dtype=None):
        import numpy
        return self.coeff*numpy.eye(shape, dtype=dtype)

    def __eq__(self, other):
        if isinstance(other, autoeye):
            return self.coeff == other.coeff
        return False

    def __add__(self, other):
        if isinstance(other, autoeye):
            return autoeye(self.coeff + other.coeff)

        if isinstance(other, (int, float, complex)):
            return autoeye(self.coeff + other)

        import numpy
        if not isinstance(other, numpy.ndarray):
            raise TypeError("autoeye can only be added to numpy.array, not %s" % type(other))

        if len(other.shape) != 2:
            raise ValueError("autoeye can only be added to 2-dim numpy arrays")

        if other.shape[0] != other.shape[1]:
            raise ValueError("autoeye can only be added to square numpy arrays")

        return self.eval(other.shape[0], dtype=other.dtype) + other

    __radd__ = __add__

    def __mul__(self, other):
        if isinstance(other, autoeye):
            return autoeye(self.coeff * other.coeff)

        if isinstance(other, (int, float, complex)):
            return autoeye(self.coeff * other)

        return NotImplemented

    __rmul__ = __mul__

    def __matmul__(self, other):
        import numpy
        if not isinstance(other, numpy.ndarray):
            raise TypeError("autoeye can only be added to numpy.array, not %s" % type(other))

        if len(other.shape) != 2:
            raise ValueError("autoeye can only be added to 2-dim numpy arrays")

        if other.shape[0] != other.shape[1]:
            raise ValueError("autoeye can only be added to square numpy arrays")

        return self.eval(other.shape[0], dtype=other.dtype) @ other

    __rmatmul__ = __matmul__

    def __str__(self):
        return 'autoeye(%s)' % self.coeff

    __repr__ = __str__

def solve_with_autoeye(a, b, **kwargs):
    import numpy.linalg

    if isinstance(a, autoeye):
        a = a.eval(b.shape[0])
    if isinstance(b, autoeye):
        b = b.eval(a.shape[0])

    return numpy.linalg.solve(a, b, **kwargs)
