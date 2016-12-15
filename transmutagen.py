import sys

from sympy import (nsolve, symbols, Mul, Add, chebyshevt, exp, simplify,
    chebyshevt_root, Tuple, diff, plot, N, solve, together, Poly)

from sympy.utilities.decorator import conserve_mpmath_dps

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


def nsolve_intervals(expr, bounds, division=30, warn=False, verbose=False, solver='bisect', **kwargs):
    """
    Divide bounds into division intervals and nsolve in each one
    """
    if verbose:
        warn = True
    roots = []
    L = bounds[1] - bounds[0]
    for i in range(division):
        interval = [bounds[0] + i*L/division, bounds[0] + (i + 1)*L/division]
        try:
            if verbose:
                print("Solving in interval", interval)
            root = nsolve(expr, interval, solver=solver, **kwargs)
        except ValueError:
            if verbose:
                print("No solution found")
            continue
        else:
            if interval[0] < root < interval[1]:
                if verbose:
                    print("Solution found:", root)
                roots.append(root)
            else:
                if warn:
                    print(root, "is not in", interval, 'discarding', file=sys.stderr)

    return roots

def plot_in_terminal(*args, **kwargs):
    """
    Run plot() but show in terminal if possible
    """
    try:
        from iterm2_tools.images import display_image_bytes
    except ImportError:
        plot(*args, **kwargs)
    else:
        from sympy.plotting.plot import unset_show
        from io import BytesIO
        unset_show()
        p = plot(*args, **kwargs, show=False)
        b = BytesIO()
        p.save(b)
        print(display_image_bytes(b.getvalue()))
        p._backend.close()

@conserve_mpmath_dps
def CRAM_exp(degree, prec=128, *, max_loops=10, c=None, maxsteps=10000,
    division=200):

    epsilon, t, i, y = symbols("epsilon t i y")

    c = c or degree*0.6

    r, num_coeffs, den_coeffs = general_rat_func(degree, t, chebyshev=True)
    E = exp(c*(t + 1)/(t - 1)) - r
    expr = E + (-1)**i*epsilon
    expr = expr*r.as_numer_denom()[1]
    expr = simplify(expr)

    points = [chebyshevt_root(2*(degree + 1) + 1, 2*(degree + 1) - j) for j in range(1, 2*(degree + 1) + 1)]
    points = [i.evalf(prec) for i in points]
    for iteration in range(max_loops):
        print('-'*80)
        print("Iteration %s:" % iteration)
        system = Tuple(*[expr.subs({i: j, t: points[j]}) for j in range(2*degree + 1)])
        system = system + Tuple(expr.replace(exp, lambda i: 0).subs({i: 2*degree + 1, t: 1}))
        #print(system)
        #print([*num_coeffs, *den_coeffs, epsilon])
        sol = dict(zip([*num_coeffs, *den_coeffs, epsilon], nsolve(system, [*num_coeffs, *den_coeffs, epsilon], [*[1]*(2*(degree + 1) - 1), 0], prec=prec)))
        print('sol', sol)
        print('system.subs(sol)', [i.evalf() for i in system.subs(sol)])
        D = diff(E.subs(sol), t)
        plot_in_terminal(E.subs(sol), (t, -1, 0.999), adaptive=False, nb_of_points=1000)
        plot_in_terminal(E.subs(sol), (t, -0.5, 0.5), adaptive=False, nb_of_points=1000)
        #plot(E.subs(sol), (t, 0.9, 1))
        # we can't use 1 because of the singularity
        print(E.subs(sol))
        points = [-1, *nsolve_intervals(D, [-1, 0.999999], maxsteps=maxsteps, prec=prec, division=division, warn=True, tol=10**-(2*prec)), 1]
        #print('points', points)
        print('D', D)
        print('[(i, D.subs(t, i)) for i in points]', [(i, D.subs(t, i)) for i in points])
        assert len(points) == 2*(degree + 1), len(points)
        Evals = [E.evalf(prec, subs={**sol, t: point}) for point in points[:-1]] + [-r.evalf(prec, subs={**sol, t: 1})]
        print('Evals', Evals)
        maxmin = N(max(map(abs, Evals)) - min(map(abs, Evals)))
        print('max - min', maxmin)
        print('epsilon', N(sol[epsilon]))
        if maxmin < 10**-prec:
            print("Converged in", iteration + 1, "iterations.")
            break
    else:
        print("!!!WARNING: DID NOT CONVERGE AFTER", max_loops, "ITERATIONS!!!", file=sys.stderr)

    inv = solve(-c*(t + 1)/(t - 1) - y, t)[0].subs(y, t)
    n, d = together(r.subs(sol).subs(t, inv)).as_numer_denom() # simplify/cancel here will add degree to the numerator and denominator
    rat_func = (Poly(n)/Poly(d).TC())/(Poly(d)/Poly(d).TC())
    return rat_func.evalf(prec)

if __name__ == '__main__':
    t = symbols('t')
    rat_func = CRAM_exp(15, 1000)
    print(rat_func)
    plot_in_terminal(rat_func - exp(-t), (t, 0, 100))
