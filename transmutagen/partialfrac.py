import random

from sympy import (symbols, fraction, nsimplify, intervals, div, LC, Add,
    degree, re, together, expand_complex, Mul, I, nsolve)

from sympy.printing.lambdarepr import NumPyPrinter
from sympy.printing.precedence import precedence

from sympy.utilities.decorator import conserve_mpmath_dps

t = symbols('t', real=True)

@conserve_mpmath_dps
def thetas_alphas(rat_func, prec, eps=None):
    """
    Do a partial fraction decomposition of rat_func

    Returns (thetas, alphas, alpha0), where thetas and alphas are lists of
    values such that

        rat_func = alpha0 + sum([alpha/(t - theta) for theta,
            alpha in zip(thetas, alphas)])

    The thetas and alphas are in general complex numbers.

    Assumes that rat_func has the same degree numerator as denominator.

    This uses the intevals() algorithm to do root finding, since apart() loses
    precision.

    eps is the length of the intervals for root finding. By default it is set
    to 10**-prec but it may need to be set smaller if there are roots smaller
    than ~1/10 to get full precision.
    """
    import mpmath
    mpmath.mp.dps = prec

    rational_rat_func = nsimplify(rat_func)
    num, den = fraction(rational_rat_func)

    if degree(den) % 1:
        raise NotImplementedError("Odd degrees are not yet supported")

    # Note, eps is NOT the precision. It's the length of the interval.
    # If a root is small (say, on the order of 10**-N), then eps will need to be 10**(-N - d)
    # to get d digits of precision. For our exp(-t) approximations, the roots
    # (thetas) are all
    # within order 10**-1...10**1, so eps is *roughly* the precision.
    eps = eps or 10**-prec

    roots = intervals(den, all=True, eps=eps)[1]
    # eps ought to be small enough that either side of the interval is the
    # precision we want, but take the average (center of the rectangle)
    # anyway.
    # XXX: Make sure to change the evalf precision if eps is lowered.
    thetas = [((i + j)/2).evalf(prec) for ((i, j), _) in roots]
    # error = [(j - i).evalf(prec) for ((i, j), _) in roots]
    alphas = []
    for theta in thetas:
        q, r = div(den, t - theta)
        alpha = (num/q).evalf(prec, subs={t: theta})
        alphas.append(alpha)
    alpha0 = (LC(num)/LC(den)).evalf(prec)
    return thetas, alphas, alpha0

def thetas_alphas_to_expr(thetas, alphas, alpha0):
    theta, alpha = symbols('theta, alpha')

    re_form = together(expand_complex(re(alpha/(t - theta))))

    return alpha0 + Add(*[re_form.subs({alpha: al, theta: th}) for th,
        al in zip(thetas, alphas)])


def thetas_alphas_to_expr_complex(thetas, alphas, alpha0):
    return alpha0 + Add(*[alpha/(t - theta) for theta,
        alpha in zip(thetas, alphas)])

def allroots(expr, degree, prec):
    roots = set()
    start = random.random() + random.random()*I
    while len(roots) < degree:
        try:
            r = nsolve(expr/Mul(*[t - r for r in roots]), start,
                maxsteps=1.7*prec, prec=prec)
        except ValueError:
            start *= random.random() + random.random()*I
        else:
            roots.add(r)
    return roots

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

        return self.coeff*numpy.eye(other.shape[0], dtype=other.dtype) + other

    __radd__ = __add__

    def __mul__(self, other):
        if isinstance(other, autoeye):
            return autoeye(self.coeff * other.coeff)

        if isinstance(other, (int, float, complex)):
            return autoeye(self.coeff * other)

    __rmul__ = __mul__

    def __str__(self):
        return 'autoeye(%s)' % self.coeff

    __repr__ = __str__
