#!/usr/bin/env python
import argparse
import os
import logging

# PYTHON_ARGCOMPLETE_OK

from sympy import (nsolve, symbols, Mul, Add, chebyshevt, exp, simplify,
    chebyshevt_root, Tuple, diff, plot, N, solve, together, Poly)

from sympy.utilities.decorator import conserve_mpmath_dps

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
# Change INFO to DEBUG for more output
logger.setLevel(logging.INFO)

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


def nsolve_intervals(expr, bounds, division=30, solver='bisect', **kwargs):
    """
    Divide bounds into division intervals and nsolve in each one
    """
    roots = []
    L = bounds[1] - bounds[0]
    for i in range(division):
        interval = [bounds[0] + i*L/division, bounds[0] + (i + 1)*L/division]
        try:
            logger.debug("Solving in interval %s", interval)
            root = nsolve(expr, interval, solver=solver, **kwargs)
        except ValueError:
            logger.debug("No solution found")
            continue
        else:
            if interval[0] < root < interval[1]:
                logger.debug("Solution found: %s", root)
                roots.append(root)
            else:
                logger.warn("%s is not in %s, discarding", root, interval)

    return roots

def plot_in_terminal(*args, logname=None, **kwargs):
    """
    Run plot() but show in terminal if possible
    """
    try:
        from iterm2_tools.images import display_image_bytes
    except ImportError:
        p = plot(*args, **kwargs)
        if logname:
            os.makedirs('plots', exist_ok=True)
            p.save('plots/%s.png' % logname)
    else:
        from sympy.plotting.plot import unset_show
        from io import BytesIO
        unset_show()
        p = plot(*args, **kwargs, show=False)
        b = BytesIO()
        p.save(b)
        if logname:
            os.makedirs('plots', exist_ok=True)
            p.save('plots/%s.png' % logname)
        print(display_image_bytes(b.getvalue()))
        p._backend.close()

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

# This decorator is actually not needed any more, but we leave it in as it
# will fail early if we are not running in SymPy master.
@conserve_mpmath_dps
def CRAM_exp(degree, prec=128, *, max_loops=10, c=None, maxsteps=None, **kwargs):
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

    Additional keyword arguments are passed to nsolve_intervals, such as
    division.

    The SymPy master branch is required for this to work.

    """
    logger.info("CRAM_exp with arguments %s", locals())

    if kwargs.get('log', True):
        os.makedirs('logs', exist_ok=True)
        logname = _get_log_file_name(locals())
        logger.addHandler(logging.FileHandler('logs/%s.log' % logname))
        logger.info("Logging to file 'logs/%s.log'", logname)
    else:
        logname = None

    epsilon, t, i, y = symbols("epsilon t i y")

    c = c or 0.6*degree
    maxsteps = int(maxsteps or 1.7*prec)

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
        sol = dict(zip([*num_coeffs, *den_coeffs, epsilon], nsolve(system, [*num_coeffs, *den_coeffs, epsilon], [*[1]*(2*(degree + 1) - 1), 0], prec=prec)))
        logger.info('sol: %s', sol)
        logger.info('system.subs(sol): %s', [i.evalf() for i in system.subs(sol)])
        D = diff(E.subs(sol), t)
        plot_in_terminal(E.subs(sol), (t, -1, 0.999), adaptive=False,
            nb_of_points=1000, logname=logname + 'iteration=%s' % iteration)
        logger.info('E.subs(sol): %s', E.subs(sol))

        # we can't use 1 because of the singularity
        points = [-1, *nsolve_intervals(D, [-1, 0.999999], prec=prec, tol=10**-(2*prec), **kwargs), 1]
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

    inv = solve(-c*(t + 1)/(t - 1) - y, t)[0].subs(y, t)
    n, d = together(r.subs(sol).subs(t, inv)).as_numer_denom() # simplify/cancel here will add degree to the numerator and denominator
    rat_func = (Poly(n)/Poly(d).TC())/(Poly(d)/Poly(d).TC())
    return rat_func.evalf(prec)


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('degree', type=int)
    parser.add_argument('prec', type=int)
    parser.add_argument('--division', type=int)
    parser.add_argument('--c', type=float)
    parser.add_argument('--maxsteps', type=int)
    parser.add_argument('--max-loops', type=int)
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

    rat_func = CRAM_exp(**arguments)
    logger.info('rat_func: %s', rat_func)
    # TODO: log this plot
    t = symbols('t')
    plot_in_terminal(rat_func - exp(-t), (t, 0, 100), adaptive=False, nb_of_points=1000)

if __name__ == '__main__':
    main()
