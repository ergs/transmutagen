#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

import argparse
import os
import logging
import datetime
import inspect
from functools import wraps

import mpmath
from sympy import (nsolve, symbols, Mul, Add, chebyshevt, exp, simplify,
    chebyshevt_root, Tuple, diff, N, solve, Poly, lambdify, sign, fraction)

from sympy.utilities.decorator import conserve_mpmath_dps

# Give a better error message if not using SymPy master
try:
    @conserve_mpmath_dps
    def test(a):
        return a

    test(1)
except TypeError:
    raise ImportError("transmutagen requires the git master version of SymPy")

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
# Change INFO to DEBUG for more output
logger.setLevel(logging.INFO)

t = symbols('t')

def general_rat_func(d, x, chebyshev=False):
    """
    Return a general rational function with numerator and denominator degree d

    Returns a tuple, (rational function, numerator coefficients, denominator coefficients)

    The constant coefficient in the denominator is always 1.
    """
    num_coeffs = symbols('p:%s' % (d+1))
    den_coeffs = symbols('q:%s' % (d+1))[1:]
    if chebyshev:
        num = Add(*(Mul(c, chebyshevt(i, x)) for i, c in enumerate(num_coeffs)))
        den = Add(*(Mul(c, chebyshevt(i, x)) for i, c in enumerate([1, *den_coeffs])))
        rat_func = num/den
    else:
        rat_func = Poly(reversed(num_coeffs), x)/Poly([*reversed(den_coeffs), 1], x)
    return rat_func, num_coeffs, den_coeffs


def nsolve_intervals(expr, bounds, division=200, solver='bisect', scale=True, prec=None, **kwargs):
    """
    Divide bounds into division intervals and nsolve in each one
    """
    roots = []
    L = bounds[1] - bounds[0]
    # These are only needed for scaling and sign checks, so don't bother with
    # full precision
    points = [bounds[0] + i*L/division for i in range(division+1)]
    low_prec_values = [expr.evalf(subs={t: point}) for point in points]
    for i in range(division):
        interval = [bounds[0] + i*L/division, bounds[0] + (i + 1)*L/division]
        try:
            logger.debug("Solving in interval %s", interval)
            s1 = low_prec_values[i]
            s2 = low_prec_values[i+1]
            if sign(s1) == sign(s2):
                logger.debug("Expr doesn't change signs on %s, skipping", interval)
                # logger.debug("Expr values, %s, %s", s1, s2)
                continue

            if scale:
                val = low_prec_values[i]
                logger.debug("Scaling by %s", val)
                scaled_expr = expr/val
            else:
                scaled_expr = expr

            root = nsolve(scaled_expr, interval, solver=solver, prec=prec, **kwargs)
        except ValueError as e:
            logger.debug("No solution found: %s", e)
            continue
        else:
            if interval[0] < root < interval[1]:
                logger.debug("Solution found: %s", root)
                roots.append(root)
                if sign(s1) == sign(s2):
                    logger.debug("Root found even though signs did not change")
            else:
                logger.warn("%s is not in %s, discarding", root, interval)

    return roots

def nsolve_points(expr, bounds, division=200, scale=True, **kwargs):
    """
    Divide bounds into division points and nsolve near each one
    """
    roots = []
    L = bounds[1] - bounds[0]
    for i in range(division):
        point = bounds[0] + i*L/division
        try:
            logger.debug("Solving near point %s", point)
            if scale:
                val = expr.evalf(kwargs['prec'], subs={t:point})
                logger.debug("Scaling by %s", val)
                expr /= val
            root = nsolve(expr, point, **kwargs)
        except ValueError as e:
            logger.debug("No solution found: %s", e)
            continue
        else:
            if root not in roots:
                logger.debug("Solution found: %s", root)
                roots.append(root)
            else:
                logger.debug("%s already found, discarding", root)

    return sorted(roots)

@conserve_mpmath_dps
def plot_in_terminal(expr, *args, prec=None, logname=None, **kwargs):
    """
    Run plot() but show in terminal if possible
    """
    from mpmath import plot
    if prec:
        mpmath.mp.dps = prec
    f = lambdify(t, expr, mpmath)
    try:
        from iterm2_tools.images import display_image_bytes
    except ImportError:
        if logname:
            os.makedirs('plots', exist_ok=True)
            file = 'plots/%s.png' % logname
        else:
            file = None
        plot(f, *args, file=file, **kwargs)
    else:
        from io import BytesIO
        b = BytesIO()
        plot(f, *args, **kwargs, file=b)
        if logname:
            os.makedirs('plots', exist_ok=True)
            with open('plots/%s.png' % logname, 'wb') as f:
                f.write(b.getvalue())
        print(display_image_bytes(b.getvalue()))

def _get_log_file_name(locals_dict):
    d = locals_dict.copy()
    kwargs = d.pop('kwargs')
    d.update(kwargs)
    d.setdefault('maxsteps')
    d.setdefault('division')
    degree = d.pop('degree')
    prec = d.pop('prec')
    info = 'degree=%s prec=%s ' % (degree, prec)
    info += ' '.join('%s=%s' % (i, d[i]) for i in sorted(d))
    return info


def log_function_args(func):
    """
    Decorator to log the arguments to func, and other info
    """
    @wraps(func)
    def _func(*args, **kwargs):
        func_name = func.__name__
        logger.info("%s with arguments %s", func_name, args)
        logger.info("%s with keyword arguments %s", func_name, kwargs)

        os.makedirs('logs', exist_ok=True)
        binding = inspect.signature(func).bind(*args, **kwargs)
        binding.apply_defaults()
        logname = _get_log_file_name(binding.arguments)
        logger.addHandler(logging.FileHandler('logs/%s.log' % logname))
        logger.info("Logging to file 'logs/%s.log'", logname)

        kwargs['logname'] = logname

        starttime = datetime.datetime.now()
        logger.info("Start time: %s", starttime)
        try:
            return func(*args, **kwargs)
        except BaseException as e:
            logger.error("Exception raised", exc_info=True)
            raise
        finally:
            endtime = datetime.datetime.now()
            logger.info("End time: %s", endtime)
            logger.info("Total time: %s", endtime - starttime)

    return _func

@conserve_mpmath_dps
@log_function_args
def CRAM_exp(degree, prec=128, *, max_loops=10, c=None, maxsteps=None,
    tol=None, nsolve_type='intervals', D_scale=1, **kwargs):
    """
    Compute the CRAM approximation of exp(-t) from t in [0, oo) of the given degree

    The Remez algorithm is used.

    degree is the degree of the numerator and denominator of the
    approximation.

    prec is the precision of the floats used in the calculation. Note that, as
    of now, the result may not be accurate to prec digits.

    max_loops is the number of loops to run the Remez algorithm before giving
    up (default 10).

    c is the coefficient of the transformation c*(t + 1)/(t - 1) that is
    applied to exp(-t) to translate it to the interval [-1, 1]. It should be
    chosen so that the maximal errors of the approximation are equally spaced
    in that interval. The default is 0.6*degree.

    maxsteps is the argument passed to the mpmath bisection solver. The
    default is 1.7*prec. See also
    https://github.com/fredrik-johansson/mpmath/issues/339.

    tol is the tolerance passed to nsolve. The default is 10**-(prec - 7).

    nsolve_type can be 'points' or 'intervals'.

    D_scale is a factor used to scale the derivative before root finding.

    Additional keyword arguments are passed to nsolve_intervals, such as
    division and scale.

    The SymPy master branch is required for this to work.

    """
    # From the log_function_args decorator
    logname = kwargs['logname']

    epsilon, t, i, y = symbols("epsilon t i y")

    c = c or 0.6*degree
    maxsteps = int(maxsteps or 1.7*prec)
    tol = tol or 10**-(prec - 7)

    if nsolve_type == 'points':
        nsolve_func = nsolve_points
    elif nsolve_type == 'intervals':
        nsolve_func = nsolve_intervals
    else:
        raise ValueError("nsolve_type must be 'points' or 'intervals'")

    r, num_coeffs, den_coeffs = general_rat_func(degree, t, chebyshev=True)
    E = exp(c*(t + 1)/(t - 1)) - r
    expr = E + (-1)**i*epsilon
    expr = expr*r.as_numer_denom()[1]
    expr = simplify(expr)

    points = [chebyshevt_root(2*(degree + 1) + 1, 2*(degree + 1) - j) for j in range(1, 2*(degree + 1) + 1)]
    points = [i.evalf(prec) for i in points]
    for iteration in range(max_loops):
        logger.info('-'*80)
        logger.info("Iteration %s:", iteration)
        system = Tuple(*[expr.subs({i: j, t: points[j]}) for j in range(2*degree + 1)])
        system = system + Tuple(expr.replace(exp, lambda i: 0).subs({i: 2*degree + 1, t: 1}))
        logger.debug('system: %s', system)
        logger.debug('[*num_coeffs, *den_coeffs, epsilon]: %s', [*num_coeffs, *den_coeffs, epsilon])
        sol = dict(zip([*num_coeffs, *den_coeffs, epsilon], nsolve(system,
            [*num_coeffs, *den_coeffs, epsilon], [*[1]*(2*(degree + 1) - 1), 0],
            prec=prec, maxsteps=maxsteps)))
        logger.info('sol: %s', sol)
        logger.info('system.subs(sol): %s', [i.evalf() for i in system.subs(sol)])
        D = diff(E.subs(sol), t)
        plot_in_terminal(E.subs(sol), (-1, 0.999), prec=prec, points=1000,
            logname=logname + ' iteration=%s' % iteration)
        logger.info('E.subs(sol): %s', E.subs(sol))

        D *= D_scale
        # we can't use 1 because of the singularity
        points = [-1, *nsolve_func(D, [-1, 0.999999], prec=prec, tol=tol, maxsteps=maxsteps, **kwargs), 1]
        logger.debug('points: %s', points)
        logger.info('D: %s', D)
        logger.info('[(i, D.subs(t, i)) for i in points]: %s', [(i, D.subs(t, i)) for i in points])
        if not len(points) == 2*(degree + 1):
            logger.error("ERROR: len(points) is (%s), not 2*(degree + 1) (%s)",
                len(points), 2*(degree + 1))
            raise RuntimeError
        Evals = [E.evalf(prec, subs={**sol, t: point}) for point in points[:-1]] + [-r.evalf(prec, subs={**sol, t: 1})]
        logger.info('Evals: %s', Evals)
        maxmin = N(max(map(abs, Evals)) - min(map(abs, Evals)))
        logger.info('max - min: %s', maxmin)
        logger.info('epsilon: %s', N(sol[epsilon]))
        if maxmin < 10**-prec:
            logger.info("Converged in %d iterations.", iteration + 1)
            break
    else:
        logger.warn("!!!WARNING: DID NOT CONVERGE AFTER %d ITERATIONS!!!", max_loops)


    # Workaround an issue. Poly loses precision unless the mpmath precision is
    # set.
    mpmath.mp.dps = prec

    # We need to be very careful about doing the inverse Mobius
    # transformation. See SymPy issue
    # https://github.com/sympy/sympy/issues/12003.
    #
    # TODO: generate this programmatically
    #
    # C = Symbol("C")
    # inv = solve(-C*(t + 1)/(t - 1) - y, t, rational=True)[0].subs(y, t)
    # inv == -2*c/(c + t) + 1
    #
    # this means:
    #
    # Shift by 1
    # Compose (multiply) with -2*c*t
    # Invert (replace t with 1/t)
    # Shift by c

    # The inversion can be done by reversing the terms, since P(1/t) ==
    # t**d*P'(t), where d = deg(P) and P' is P with the terms reversed. The
    # t**d will cancel in the numerator and denominator.

    # fraction is better than as_numer_denom(). It doesn't try to do anything
    # smart, which is what we want ("smart" things can lose precision)
    frac = list(map(Poly, fraction(r.subs(sol))))
    for i in range(len(frac)):
        # Shift by 1
        frac[i] = frac[i].shift(1)
        # Compose with -2*c*t
        frac[i] = frac[i].compose(Poly(-2*c*t))
        # Invert
        # XXX: Is there a better way than rep.rep
        frac[i] = Poly(reversed(frac[i].rep.rep), t)
        # Shift by c
        frac[i] = frac[i].shift(c)

    n, d = frac
    rat_func = n/d.TC()/(d/d.TC())
    ret = rat_func.evalf(prec)

    logger.info('rat_func: %s', rat_func)
    plot_in_terminal(rat_func - exp(-t), (0, 100), prec=prec, points=1000,
        logname=logname + ' final')

    return ret

def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('degree', type=int)
    parser.add_argument('prec', type=int)
    parser.add_argument('--division', type=int)
    parser.add_argument('--c', type=float)
    parser.add_argument('--maxsteps', type=int)
    parser.add_argument('--max-loops', type=int)
    parser.add_argument('--tol', type=mpmath.mpf)
    parser.add_argument('--nsolve-type', default=None, choices=['points',
        'intervals'])
    parser.add_argument('--solver', default=None)
    parser.add_argument('--D-scale', default=None, type=float)
    parser.add_argument('--scale', default=None, type=bool)
    parser.add_argument('--log-level', default=None, choices=['debug', 'info',
        'warning', 'error', 'critical'])
    try:
        import argcomplete
        argcomplete.autocomplete(parser)
    except ImportError:
        pass
    args = parser.parse_args()

    arguments = args.__dict__.copy()
    for i in arguments.copy():
        if not arguments[i]:
           del arguments[i]
    if args.log_level:
        logger.setLevel(getattr(logging, args.log_level.upper()))
        del arguments['log_level']

    CRAM_exp(**arguments)

if __name__ == '__main__':
    main()
