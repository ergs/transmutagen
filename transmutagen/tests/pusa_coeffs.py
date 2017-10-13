"""
Extracted from M. Pusa, "Correction to Partial Fraction Decomposition
Coefficients for Chebyshev Rational Approximation on the Negative Real Axis"
"""

part_frac_coeffs = {
    14: {
        'thetas': {
            'real': [
                "-8.8977731864688888199e+0",
                "-3.7032750494234480603e+0",
                # published as "-0.2087586382501301251e+0",
                "-2.087586382501301251e-1",
                "+3.9933697105785685194e+0",
                "+5.0893450605806245066e+0",
                "+5.6231425727459771248e+0",
                "+2.2697838292311127097e+0",
                ],
            'imaginary': [
                "+1.6630982619902085304e+1",
                "+1.3656371871483268171e+1",
                "+1.0991260561901260913e+1",
                "+6.0048316422350373178e+0",
                "+3.5888240290270065102e+0",
                "+1.1940690463439669766e+0",
                "+8.4617379730402214019e+0",
                ],
            },

        'alphas': {
            'real': [
                "-7.1542880635890672853e-5",
                "+9.4390253107361688779e-3",
                "-3.7636003878226968717e-1",
                "-2.3498232091082701191e+1",
                "+4.6933274488831293047e+1",
                "-2.7875161940145646468e+1",
                "+4.8071120988325088907e+0",
                ],
            'imaginary': [
                "+1.4361043349541300111e-4",
                "-1.7184791958483017511e-2",
                "+3.3518347029450104214e-1",
                "-5.8083591297142074004e+0",
                "+4.5643649768827760791e+1",
                "-1.0214733999056451434e+2",
                "-1.3209793837428723881e+0",
                ],
            },

        'alpha0': {
            'real': [
                "+1.8321743782540412751e-14",
                ],
            'imaginary': [
                "+0.0000000000000000000e+0",
                ],
            },
        },

    16: {
        'thetas': {
            'real': [
                "-1.0843917078696988026e+1",
                "-5.2649713434426468895e+0",
                "+5.9481522689511774808e+0",
                "+3.5091036084149180974e+0",
                "+6.4161776990994341923e+0",
                "+1.4193758971856659786e+0",
                "+4.9931747377179963991e+0",
                "-1.4139284624888862114e+0",
                ],
            'imaginary': [
                "+1.9277446167181652284e+1",
                "+1.6220221473167927305e+1",
                "+3.5874573620183222829e+0",
                "+8.4361989858843750826e+0",
                "+1.1941223933701386874e+0",
                "+1.0925363484496722585e+1",
                "+5.9968817136039422260e+0",
                "+1.3497725698892745389e+1",
                ],
            },

        'alphas': {
            'real': [
                "-5.0901521865224915650e-7",
                "+2.1151742182466030907e-4",
                "+1.1339775178483930527e+2",
                "+1.5059585270023467528e+1",
                "-6.4500878025539646595e+1",
                "-1.4793007113557999718e+0",
                "-6.2518392463207918892e+1",
                "+4.1023136835410021273e-2",
                ],
            'imaginary': [
                "-2.4220017652852287970e-5",
                "+4.3892969647380673918e-3",
                "+1.0194721704215856450e+2",
                "-5.7514052776421819979e+0",
                "-2.2459440762652096056e+2",
                "+1.7686588323782937906e+0",
                "-1.1190391094283228480e+1",
                "-1.5743466173455468191e-1",
                ],
            },

        'alpha0': {
            'real': [
                "+2.1248537104952237488e-16",
                ],
            'imaginary': [
                "+0.0000000000000000000e+0",
                ],
            },
        },
}

def get_paper_part_frac(degree):
    from ..partialfrac import thetas_alphas_to_expr_complex
    from sympy import Float, I
    # Values above are negative what we expect. It also only includes one of
    # each complex conjugate, which is fine so long as we pass in the positive
    # imaginary parts for the thetas.
    thetas = [-Float(r) + Float(i)*I for r, i in
        zip(part_frac_coeffs[degree]['thetas']['real'],
            part_frac_coeffs[degree]['thetas']['imaginary'])]

    alphas = [-Float(r) + Float(i)*I for r, i in
        zip(part_frac_coeffs[degree]['alphas']['real'],
            part_frac_coeffs[degree]['alphas']['imaginary'])]

    [alpha0] = [Float(r) + Float(i)*I for r, i in
        zip(part_frac_coeffs[degree]['alpha0']['real'],
            part_frac_coeffs[degree]['alpha0']['imaginary'])]

    return thetas_alphas_to_expr_complex(thetas, alphas, alpha0)


def get_paper_expr(degree):
    from ..partialfrac import thetas_alphas_to_expr_expanded
    from sympy import Float, I

    thetas = [-Float(r) + Float(i)*I for r, i in
        zip(part_frac_coeffs[degree]['thetas']['real'],
            part_frac_coeffs[degree]['thetas']['imaginary'])]
    thetas = thetas + [i.conjugate() for i in thetas]

    alphas = [-Float(r) + Float(i)*I for r, i in
        zip(part_frac_coeffs[degree]['alphas']['real'],
            part_frac_coeffs[degree]['alphas']['imaginary'])]
    alphas = alphas + [i.conjugate() for i in alphas]

    [alpha0] = [Float(r) + Float(i)*I for r, i in
        zip(part_frac_coeffs[degree]['alpha0']['real'],
            part_frac_coeffs[degree]['alpha0']['imaginary'])]

    return thetas_alphas_to_expr_expanded(thetas, alphas, alpha0)

def plot_difference(*, file=None, all_plots=False):
    """
    Plot the difference between our coefficients and the Pusa ones

    The file argument is the file to save the plot to. If all_plots is True,
    more plots are shown. Note that file only controls the first plot when
    all_plots=True.

    """
    # TODO: Avoid using SymPy's re, which evaluate to re form.
    from ..partialfrac import (thetas_alphas, thetas_alphas_to_expr_complex,
        customre, t)
    from ..cram import get_CRAM_from_cache
    from ..util import plot_in_terminal, cplot_in_terminal
    from sympy import re, exp

    import matplotlib.pyplot as plt

    part_fracs = {}
    paper_part_fracs = {}
    for degree in [14, 16]:
        expr = get_CRAM_from_cache(degree, 200)
        thetas, alphas, alpha0 = thetas_alphas(expr, 200)
        part_frac = thetas_alphas_to_expr_complex(thetas, alphas, alpha0)
        part_frac = part_frac.replace(customre, re)

        paper_part_frac = get_paper_part_frac(degree).replace(customre, re)

        part_fracs[degree] = part_frac
        paper_part_fracs[degree] = paper_part_frac

    # print('part_frac', part_frac)
    # print('paper_part_frac', paper_part_frac)

    if not all_plots:
        # mpmath.plot doesn't allow us to customize the way the plot looks, so
        # we have it plot it, and then change it directly from pylab, then use
        # plt.savefig to actually save the file
        plot_in_terminal([part_fracs[14] - paper_part_fracs[14],
            part_fracs[16] - paper_part_fracs[16]], (0, 100), prec=200,
            points=1000, file=None)
        axes = plt.gca()
        axes.set_xlabel('t')
        axes.set_ylabel(r'Difference in $\hat{r}_{k,k}(t)$')
        plt.legend(['Degree 14', 'Degree 16'])

        # Put "x 10^-19" on every y-axis tick
        locs, labels = plt.yticks()
        plt.yticks(locs, [r"$%d\times 10^{%d}$" % (int(i/1e-19), -19) for i
            in locs])

        plt.tight_layout()
        plt.savefig(file)
    else:
        print("Difference between our partial fraction and Pusa paper partial fraction, degree", degree)
        plot_in_terminal(part_frac - paper_part_frac, (0, 100), prec=200,
            points=1000, file=file)
        cplot_in_terminal(part_frac - paper_part_frac, re=(0, 100), im=[-30, 30],
            prec=200, points=100000, verbose=False)

        print("Difference between our partial fraction and exp(-t), degree", degree)
        plot_in_terminal(part_frac - exp(-t), (0, 100), prec=200, points=1000)
        cplot_in_terminal(part_frac - exp(-t), re=(0, 100), im=[-30, 30],
            prec=200, points=100000, verbose=False)

        print("Difference between Pusa paper partial fraction and exp(-t), degree", degree)
        plot_in_terminal(paper_part_frac - exp(-t), (0, 100), prec=200, points=1000)
        cplot_in_terminal(part_frac - exp(-t), re=(0, 100), im=[-30, 30],
            prec=200, points=100000, verbose=False)

def transmutagen_cram_error(degree, t0, prec=200):
    from ..partialfrac import (thetas_alphas, thetas_alphas_to_expr_complex,
        customre, t)
    from ..cram import get_CRAM_from_cache

    from sympy import re, exp, nsolve, diff

    expr = get_CRAM_from_cache(degree, prec)
    thetas, alphas, alpha0 = thetas_alphas(expr, prec)
    part_frac = thetas_alphas_to_expr_complex(thetas, alphas, alpha0)
    part_frac = part_frac.replace(customre, re)


    E = part_frac - exp(-t)
    E = E.evalf(20)

    return E.subs(t, nsolve(diff(E, t), t0, prec=prec))

def paper_cram_error(degree, t0, prec=200):
    from ..partialfrac import t, customre

    from sympy import re, exp, nsolve, diff

    paper_part_frac = get_paper_part_frac(degree).replace(customre, re)

    E = paper_part_frac - exp(-t)

    return E.subs(t, nsolve(diff(E, t), t0, prec=prec))

if __name__ == '__main__':
    plot_difference(all_plots=True)
