from sympy import Mul

from sympy.printing.lambdarepr import NumPyPrinter
from sympy.printing.precedence import precedence

import numpy as np
import scipy.sparse

class MatrixNumPyPrinter(NumPyPrinter):
    """
    Print an expression for numpy assuming the variables are matrices

    Prints inversions as solve() and multiplication of nonconstants as @.

    """
    _default_settings = {
        **NumPyPrinter._default_settings,
        # TODO: Make this automatic
        'use_autoeye': True,
        }

    def _print_Mul(self, expr):
        prec = precedence(expr)

        pows = [i for i in expr.args if i.is_Pow and i.exp < 0]
        if len(pows) > 1:
            raise NotImplementedError("Need exactly one inverted Pow, not %s" % len(pows))

        if not pows:
            if self._settings['use_autoeye']:
                terms = [self._print(self.parenthesize(i, prec)) for i in expr.args]
                return '@'.join(terms)
            else:
                num_terms = [self._print(self.parenthesize(i, prec)) for i in
                    expr.args if i.is_Number]
                mat_terms = [self._print(self.parenthesize(i, prec)) for i in
                    expr.args if not i.is_Number]
                if num_terms:
                    return '*'.join(num_terms) + '*' + '@'.join(mat_terms)
                return '@'.join(mat_terms)

        [pow] = pows

        rest = Mul(*[i for i in expr.args if i != pow])

        if self._settings['use_autoeye']:
            return 'solve_with_autoeye(%s, %s)' % (self._print(1/pow), self._print(rest))
        return 'solve(%s, %s)' % (self._print(1/pow), self._print(rest))

    def _print_Integer(self, expr):
        if self._settings['use_autoeye']:
            return 'autoeye(%s)' % super()._print_Integer(expr)
        return super()._print_Integer(expr)

    def _print_Float(self, expr):
        if self._settings['use_autoeye']:
            return 'autoeye(%s)' % super()._print_Float(expr)
        return super()._print_Float(expr)

    def _print_Pow(self, expr):
        if expr.exp.is_Integer and expr.exp > 1:
            return 'matrix_power(%s, %s)' % (self._print(expr.base), expr.exp)
        return super()._print_Pow(expr)

    def _print_ImaginaryUnit(self, expr):
        if self._settings['use_autoeye']:
            return 'autoeye(1j)'
        return '1j'

    def _print_customre(self, expr):
        return 'real(%s)' % self._print(expr.args[0])

class autoeye:
    __array_priority__ = 11

    def __init__(self, coeff=1):
        self.coeff = coeff

    def eval(self, shape, eye_type, dtype=None):
        return self.coeff*eye_type(shape, dtype=dtype)

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
        import scipy.sparse
        if isinstance(other, numpy.ndarray):
            eye_type = numpy.eye
        elif isinstance(other, scipy.sparse.spmatrix):
            eye_type = scipy.sparse.eye
        else:
            return NotImplemented

        if len(other.shape) != 2:
            raise ValueError("autoeye can only be added to 2-dim numpy arrays")

        if other.shape[0] != other.shape[1]:
            raise ValueError("autoeye can only be added to square numpy arrays, other.shape is %s" % (other.shape,))

        return self.eval(other.shape[0], eye_type, dtype=other.dtype) + other

    __radd__ = __add__

    def __mul__(self, other):
        if isinstance(other, autoeye):
            return autoeye(self.coeff * other.coeff)

        if isinstance(other, (int, float, complex)):
            return autoeye(self.coeff * other)

        return NotImplemented

    __rmul__ = __mul__

    def __matmul__(self, other):
        if isinstance(other, autoeye):
            return autoeye(self.coeff * other.coeff)

        import numpy
        import scipy.sparse
        if isinstance(other, numpy.ndarray):
            eye_type = numpy.eye
        elif isinstance(other, scipy.sparse.spmatrix):
            eye_type = scipy.sparse.eye
        else:
            return NotImplemented

        if len(other.shape) != 2:
            raise ValueError("autoeye can only be matmuled by 2-dim numpy arrays")

        if other.shape[0] != other.shape[1]:
            raise ValueError("autoeye can only be matmuled by square numpy arrays")

        return self.eval(other.shape[0], eye_type=eye_type, dtype=other.dtype) @ other

    __rmatmul__ = __matmul__

    def __str__(self):
        return 'autoeye(%s)' % self.coeff

    __repr__ = __str__

def numpy_solve_with_autoeye(a, b, **kwargs):
    import numpy

    if isinstance(a, autoeye):
        a = a.eval(b.shape[0], numpy.eye)
    if isinstance(b, autoeye):
        b = b.eval(a.shape[0], numpy.eye)

    return numpy.linalg.solve(a, b, **kwargs)


def scipy_sparse_solve_with_autoeye(a, b, **kwargs):
    import scipy.sparse.linalg

    if isinstance(a, autoeye):
        a = a.eval(b.shape[0], scipy.sparse.eye)
    if isinstance(b, autoeye):
        b = b.eval(a.shape[0], scipy.sparse.eye)

    return scipy.sparse.linalg.spsolve(a, b, **kwargs)

scipy_translations = {
    'solve_with_autoeye': scipy_sparse_solve_with_autoeye,
    'autoeye': autoeye,
    'matrix_power': lambda a, b: a**b,
    'real': lambda m: scipy.sparse.csr_matrix((np.real(m.data), m.indices,
        m.indptr), shape=m.shape),
    }
