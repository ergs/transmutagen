from sympy import Mul

from sympy.printing.lambdarepr import NumPyPrinter
from sympy.printing.precedence import precedence

import numpy as np
import scipy.sparse.linalg

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
                    expr.args if i.is_number]
                mat_terms = [self._print(self.parenthesize(i, prec)) for i in
                    expr.args if not i.is_number]
                if num_terms and mat_terms:
                    return '*'.join(num_terms) + '*' + '@'.join(mat_terms)
                else:
                    return '*'.join(num_terms) + '@'.join(mat_terms)

        [pow] = pows

        rest = Mul(*[i for i in expr.args if i != pow])

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

        if isinstance(other, np.ndarray):
            eye_type = np.eye
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

        if isinstance(other, np.ndarray):
            eye_type = np.eye
        elif isinstance(other, scipy.sparse.spmatrix):
            eye_type = scipy.sparse.eye
        else:
            return NotImplemented

        if len(other.shape) != 2:
            raise ValueError("autoeye can only be matmuled by 2-dim numpy arrays")

        return self.eval(other.shape[0], eye_type=eye_type, dtype=other.dtype) @ other

    def __rmatmul__(self, other):
        if isinstance(other, autoeye):
            return autoeye(self.coeff * other.coeff)

        if isinstance(other, np.ndarray):
            eye_type = np.eye
        elif isinstance(other, scipy.sparse.spmatrix):
            eye_type = scipy.sparse.eye
        else:
            return NotImplemented

        if len(other.shape) != 2:
            # Matmul works weird on 1d arrays. It treats them as column
            # vectors from the left and row vectors from the right.
            raise ValueError("autoeye can only be matmuled by 2-dim numpy arrays")

        ret_shape = other.shape[1]
        return other @ self.eval(ret_shape, eye_type=eye_type, dtype=other.dtype)

    def __str__(self):
        return 'autoeye(%s)' % self.coeff

    __repr__ = __str__

def numpy_solve_with_autoeye(a, b, **kwargs):
    if isinstance(a, autoeye):
        a = a.eval(b.shape[0], np.eye)
    if isinstance(b, autoeye):
        b = b.eval(a.shape[0], np.eye)

    return np.linalg.solve(a, b, **kwargs)


def scipy_sparse_solve_with_autoeye(a, b, **kwargs):
    if isinstance(a, autoeye):
        a = a.eval(b.shape[0], scipy.sparse.eye)
    if isinstance(b, autoeye):
        b = b.eval(a.shape[0], scipy.sparse.eye)

    ret = scipy.sparse.linalg.spsolve(a, b, **kwargs)
    if isinstance(ret, np.ndarray):
        ret = ret[:,np.newaxis]

    return ret

scipy_translations = {
    'solve': scipy.sparse.linalg.spsolve,
    'autoeye': autoeye,
    'matrix_power': lambda a, b: a**b,
    'real': lambda m: np.real(m) if isinstance(m, np.ndarray) else scipy.sparse.csr_matrix((np.real(m.data), m.indices,
        m.indptr), shape=m.shape),
    }

scipy_translations_autoeye = {
    **scipy_translations,
    'solve': scipy_sparse_solve_with_autoeye,
    }
