from sympy import symbols, fraction, degree as poly_degree, nsimplify, intervals

t = symbols('t')

def theta_alphas(rat_func, eps=None):
    rational_rat_func = nsimplify(rat_func)
    num, den = fraction(rational_rat_func)
    degree = poly_degree(num)
    # Note, eps is NOT the precision. It's the length of the interval.
    # If a root is small (say, on the order of 10**-N), then eps will need to be 10**(-N - d)
    # to get d digits of precision. For our exp(-t) approximations, the roots (thetas) are all
    # within order 10**-1...10**1, so eps is *roughly* the precision.
    eps = eps or 10**-degree


    roots = intervals(den, all=True, eps=eps)[1]
    # eps ought to be small enough that either side of the interval is the precision we want,
    # but for the sake of with smaller eps, take the average (center of the rectangle)
    # XXX: Make sure to change the evalf precision if eps is lowered.
    thetas = [((i + j)/2).evalf(17) for ((i, j), _) in roots]
    error = [(j - i).evalf(17) for ((i, j), _) in roots]
    alphas = []
    for theta in thetas:
        q, r = div(den, t - theta)
        alpha = (num/q).evalf(17, subs={t: theta})
        alphas.append(alpha)
    alpha0 = LC(num)/LC(den)
    return thetas, alphas, alpha0
