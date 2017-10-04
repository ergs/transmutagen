from functools import wraps

from sympy import Mul, symbols, lambdify, Add, LC, fraction

from sympy.printing.lambdarepr import NumPyPrinter
from sympy.printing.precedence import precedence

import numpy as np
import scipy.sparse.linalg

from .cram import CRAM_exp, get_CRAM_from_cache
from .partialfrac import (thetas_alphas, thetas_alphas_to_expr_complex,
    thetas_alphas_to_expr_real, t, multiply_vector, allroots)
from .util import memoize

class MatrixNumPyPrinter(NumPyPrinter):
    """
    Print an expression for numpy assuming the variables are matrices

    Prints inversions as solve() and multiplication of nonconstants as @.

    """
    _default_settings = {
        **NumPyPrinter._default_settings,
        # TODO: Make this automatic
        'use_autoeye': True,
        'py_solve': False,
        'float128': False,
        }

    def __init__(self, settings=None):
        if settings is not None and 'py_solve' in settings:
            if settings.get('use_autoeye', False):
                raise ValueError("use_autoeye cannot be used with py_solve")
            settings['use_autoeye'] = False
        super().__init__(settings)

    def _print_Add(self, expr):
        if not (self._settings['use_autoeye'] or self._settings['py_solve']):
            return super()._print_Add(expr)

        prec = precedence(expr)

        num_terms = [i for i in expr.args if i.is_number]
        rest_terms = [i for i in expr.args if i not in num_terms]

        if len(rest_terms) > 1:
            rest = super()._print_Add(Add(*rest_terms))
        elif len(rest_terms) == 1:
            rest = self._print(rest_terms[0])
        else:
            if self._settings['py_solve']:
                return super()._print_Add(expr)
            rest = ''

        if len(num_terms) > 1:
            num = self.__class__({**self._settings, 'use_autoeye': False})._print_Add(Add(*num_terms))
        elif len(num_terms) == 1:
            num = self.__class__({**self._settings, 'use_autoeye': False})._print(num_terms[0])
        else:
            num = ''

        if rest and num:
            if self._settings['py_solve']:
                return "diag_add(%s, %s)" % (self._print(rest_terms[0]), self._print(Add(*num_terms)))
            return self.parenthesize(rest + ' + autoeye(%s)' % num, prec)
        elif rest:
            return self.parenthesize(rest, prec)
        else:
            if self._settings['use_autoeye']:
                # No need to parenthesize
                return 'autoeye(%s)' % num
            else:
                return self.parenthesize(num, prec)

    def _print_Mul(self, expr):
        prec = precedence(expr)

        pows = [i for i in expr.args if i.is_Pow and i.exp < 0]
        if len(pows) > 1:
            raise NotImplementedError("Need exactly one inverted Pow, not %s" % len(pows))

        if not pows:
            no_autoeye = self.__class__({**self._settings, 'use_autoeye': False})
            num_terms = [no_autoeye._print(no_autoeye.parenthesize(i, prec)) for i in
                expr.args if i.is_number]
            mat_terms = [self._print(self.parenthesize(i, prec)) for i in
                expr.args if not i.is_number]
            if len(mat_terms) >= 2 and self._settings['py_solve']:
                raise NotImplementedError("matrix multiplication is not yet supported with py_solve")
            if num_terms and mat_terms:
                return '*'.join(num_terms) + '*' + '@'.join(mat_terms)
            else:
                if self._settings['use_autoeye']:
                    if num_terms:
                        return ('autoeye(%s)' % '*'.join(num_terms)) + '@'.join(mat_terms)
                    return '@'.join(mat_terms)

                return '*'.join(num_terms) + '@'.join(mat_terms)

        [pow] = pows

        rest = Mul(*[i for i in expr.args if i != pow])

        return 'solve(%s, %s)' % (self._print(1/pow), self._print(rest))

    def _print_Integer(self, expr):
        if self._settings['use_autoeye']:
            return 'autoeye(%s)' % super()._print_Integer(expr)
        return super()._print_Integer(expr)

    def _print_Float(self, expr):
        super_float = super()._print_Float(expr)
        if self._settings['float128']:
            super_float = 'float128(%r)' % super_float
        if self._settings['use_autoeye']:
            return 'autoeye(%s)' % super_float
        return super_float

    def _print_Pow(self, expr):
        if self._settings['py_solve']:
            raise NotImplementedError("Matrix powers are not yet supported with py_solve")
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
    'float128': np.float128,
    }

scipy_translations_autoeye = {
    **scipy_translations,
    'solve': scipy_sparse_solve_with_autoeye,
    }

@memoize
def CRAM_matrix_exp_lambdify(degree=14, prec=200, *, use_cache=True,
    form='complex partial fraction', py_solve=False):
    """
    Return a lambdified function for the CRAM approximation to exp(-x)

    form can be one of

    'complex partial fraction' (the default)
    'real partial fraction'
    'rational function'
    'rational function horner'
    'factored'

    When py_solve = True, the py_solve module will be used (scipy is used
    otherwise). In this case, it is much faster to pre-flatten the input
    matrix:

    >>> mat, time, b = ...
    >>> mat = py_solve.asflat(mat)
    >>> f = CRAM_matrix_exp_lambdify(py_solve=True)
    >>> f(-mat*time, b)

    """
    # TODO: This function should give exp(x), not exp(-x)

    if use_cache:
        rat_func = get_CRAM_from_cache(degree, prec)
    else:
        rat_func = CRAM_exp(degree, prec, plot=False)
    thetas, alphas, alpha0 = thetas_alphas(rat_func, prec)
    if form == 'complex partial fraction':
        expr = thetas_alphas_to_expr_complex(thetas, alphas, alpha0)
    elif form == 'real partial fraction':
        expr = thetas_alphas_to_expr_real(thetas, alphas, alpha0)
    elif form in ['rational function', 'rational function horner']:
        expr = rat_func
    elif form == 'factored':
        num, den = fraction(rat_func)
        # XXX: complex conjugate roots have the same absolute value
        numroots = sorted(allroots(num, degree, prec), key=lambda i: abs(i))
        denroots = sorted(allroots(den, degree, prec), key=lambda i: abs(i))
        p1q1 = LC(num)/LC(den)
    else:
        raise ValueError("Invalid argument for 'form': %s" % (form,))
    n0 = symbols("n0", commutative=False)

    if py_solve:
        from py_solve import py_solve
        module = [py_solve, 'numpy']
        printer = MatrixNumPyPrinter({'py_solve': True})
        def wrapper(f):
            @wraps(f)
            def _f(t, n0):
                t = py_solve.asflat(t)
                return f(t, n0)
            return _f
    else:
        module = scipy_translations_autoeye
        printer = MatrixNumPyPrinter({'use_autoeye': True})
        wrapper = lambda f: f

    if form != 'factored':
        return wrapper(lambdify((t, n0), multiply_vector(expr, n0,
            horner=(form == 'rational function horner')),
            module, printer=printer, dummify=False))
    else:
        if py_solve:
            raise NotImplementedError("py_solve is not supported with factor yet")

        # TODO: Code generate this as a single expression
        def e_factored(mat, b, reverse=False):
            if reverse:
                r = reversed
            else:
                r = lambda i: i

            for num_root, den_root in zip(r(numroots), r(denroots)):
                f = lambdify((t, n0), multiply_vector((t - num_root)/(t -
                    den_root), n0), scipy_translations_autoeye,
                    printer=MatrixNumPyPrinter())
                b = f(mat, b)
            return float(p1q1)*b

        return e_factored
