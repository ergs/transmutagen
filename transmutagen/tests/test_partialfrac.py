from sympy import together, expand_complex, re, im, symbols

from ..partialfrac import t

def test_re_form():
    theta, alpha = symbols('theta, alpha')

    # Check that this doesn't change
    re_form = together(expand_complex(re(alpha/(t - theta))))
    assert re_form == (t*re(alpha) - re(alpha)*re(theta) -
        im(alpha)*im(theta))/((t - re(theta))**2 + im(theta)**2)
