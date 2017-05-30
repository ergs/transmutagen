import warnings
import random

from sympy import (symbols, fraction, nsimplify, intervals, div, LC, Add,
    degree, re, im, together, expand_complex, Mul, I, nsolve, Function,
    Symbol, Poly, pi, count_roots)

from sympy.utilities.decorator import conserve_mpmath_dps

from .util import memoize

t = symbols('t', real=True)

@memoize
@conserve_mpmath_dps
def thetas_alphas(rat_func, prec, *, use_intervals=False, eps=None):
    """
    Do a partial fraction decomposition of rat_func

    Returns (thetas, alphas, alpha0), where thetas and alphas are lists of
    values such that

        rat_func = alpha0 + sum([alpha/(t - theta) for theta,
            alpha in zip(thetas, alphas)])

    The thetas and alphas are in general complex numbers.

    Assumes that rat_func has the same degree numerator as denominator.

    If use_intervals=True, this uses the intevals() algorithm to do root
    finding. This algorithm is very slow, but has guaranteed precision, and is
    guaranteed to find all the roots. If it is False (the default), nsolve is
    used.

    eps is the length of the intervals for root finding. By default it is set
    to 10**-prec but it may need to be set smaller if there are roots smaller
    than ~1/10 to get full precision. If use_intervals=False, eps is ignored.
    """
    import mpmath
    mpmath.mp.dps = prec

    num, den = fraction(rat_func)
    d = degree(den)

    if use_intervals:
        rational_rat_func = nsimplify(rat_func)
        num, den = fraction(rational_rat_func)

        if d % 1:
            raise NotImplementedError("Odd degrees are not yet supported with use_intervals=True")

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
    else:
        thetas = list(allroots(den, d, prec))
    alphas = []
    for theta in thetas:
        q, r = div(den, t - theta)
        alpha = (num/q).evalf(prec, subs={t: theta})
        alphas.append(alpha)
    alpha0 = (LC(num)/LC(den)).evalf(prec)
    return thetas, alphas, alpha0

def thetas_alphas_to_expr_real(thetas, alphas, alpha0):
    theta, alpha = symbols('theta, alpha')

    re_form = together(expand_complex(re(alpha/(t - theta))))

    return alpha0 + Add(*[re_form.subs({alpha: al, theta: th}) for th,
        al in zip(thetas, alphas)])

# sympy.re evaluates by default, which we don't want
class customre(Function):
    nargs = 1

    def _eval_evalf(self, prec):
        return re(self.args[0]._eval_evalf(prec))

def thetas_alphas_to_expr_complex(thetas, alphas, alpha0):
    return alpha0 + 2*customre(Add(*[alpha/(t - theta) for theta,
        alpha in zip(thetas, alphas) if im(theta) >= 0]))

def thetas_alphas_to_expr_complex2(thetas, alphas, alpha0):
    """
    Same as thetas_alphas_to_expr_complex except without the 2*re()
    optimization
    """
    return alpha0 + Add(*[alpha/(t - theta) for theta,
        alpha in zip(thetas, alphas)])

def thetas_alphas_to_expr_expanded(thetas, alphas, alpha0):
    expr = thetas_alphas_to_expr_complex2(thetas, alphas, alpha0)
    p, q = expr.as_numer_denom()
    p = Poly(p, t)
    q = Poly(q, t)
    p = Poly(p/q.TC(), t, expand=True)
    q = Poly(q/q.TC(), t, expand=True)

    return p, q

def allroots(expr, degree, prec, chop=True):
    roots = set()
    start = random.random() + random.random()*I
    MAX_ITERATIONS = 5*degree
    i = 0
    while len(roots) < degree:
        i += 1
        if i > MAX_ITERATIONS:
            raise RuntimeError("MAX_ITERATIONS exceeded. Could only find %s roots" % len(roots))
        try:
            r = nsolve(expr/Mul(*[t - r for r in roots]), start,
                maxsteps=1.7*prec, prec=prec)
        except ValueError:
            start = 10*(random.random() + random.random()*I)
        else:
            roots.add(r)

    # Because we started with a complex number, real roots will have a small
    # complex part. Assume that the roots with the smallest complex parts are
    # real. We could also check which roots are conjugate pairs.
    if chop:
        from sympy.core.compatibility import GROUND_TYPES
        if GROUND_TYPES == 'python':
            warnings.simplefilter('default')
            warnings.warn("It is recommended to install gmpy2 to speed up transmutagen")
        n_real_roots = count_roots(nsimplify(expr))
        roots = sorted(roots, key=lambda i:abs(im(i)))
        real_roots, complex_roots = roots[:n_real_roots], roots[n_real_roots:]
        real_roots = [re(i) for i in real_roots]
        roots = {*real_roots, *complex_roots}
    return roots

@conserve_mpmath_dps
def multiply_vector(expr, n0, horner=False):
    """
    Multiply the vector n0 into the expression expr

    It is recommended to make the vector noncommutative (Symbol("n0",
    commutative=False).

    This makes assumptions that the form of expr will be one of the ones
    generated by this file.

    To apply the horner scheme to the numerator and denominator, use horner=True.
    """
    # Make sure we are using a version of SymPy that has
    # https://github.com/sympy/sympy/pull/12088.
    x, y = symbols('x y')
    if Poly(pi.evalf(100)*x*y, x).as_expr() != pi.evalf(100)*x*y:
        raise RuntimeError("multiply_vector requires https://github.com/sympy/sympy/pull/12088")

    from sympy import horner as _horner
    if horner:
        # horner doesn't work on noncommutatives
        n1 = Symbol(n0.name)
        num, den = fraction(expr)
        return _horner(num*n1).subs(n1, n0)/_horner(den)

    # TODO: Don't distribute across complex numbers
    if expr.is_Add:
        if expr.is_number:
            return expr*n0
        # expand_mul(deep=False) does too much (it breaks horner)
        return Add(*[multiply_vector(i, n0) for i in expr.args])

    coeff, rest = expr.as_coeff_Mul()
    if isinstance(rest, customre):
        return coeff*customre(multiply_vector(rest.args[0], n0))

    num, den = fraction(expr)
    if expr != num:
        return multiply_vector(num, n0)/den
    return expr*n0
