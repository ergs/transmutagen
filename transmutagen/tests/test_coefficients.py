import decimal

import pytest
from sympy import im

slow = pytest.mark.skipif(
    not pytest.config.getoption("--runslow"),
    reason="need --runslow option to run"
)

TOTAL_DEGREES = 30

from .crv_coeffs import coeffs as correct_coeffs
from .partfrac_coeffs import part_frac_coeffs
from ..cram import get_CRAM_from_cache, CRAM_coeffs
from ..partialfrac import thetas_alphas
from ..util import memoize

# @slow
@pytest.mark.parametrize('degree', range(1, TOTAL_DEGREES+1))
def test_coefficients(degree):
    generated_coeffs = {}
    expr = get_CRAM_from_cache(degree, 200)
    generated_coeffs[degree] = CRAM_coeffs(expr, 20,
        decimal_rounding=True)
    # pytest won't show the full expr from the assert, so we print it too
    print(expr)
    assert generated_coeffs[degree] == correct_coeffs[degree], expr

@memoize
def _thetas_alphas(expr, prec):
    return thetas_alphas(expr, prec)

# @pytest.mark.xfail
@pytest.mark.parametrize('degree', [14, 16])
@pytest.mark.parametrize('idx', range(16//2))
@pytest.mark.parametrize('real_imag', ['real', 'imag'])
def test_partial_fraction_coefficients(degree, idx, real_imag):
    if idx >= degree//2:
        # How do I do this in the decorators?
        return
    expr = get_CRAM_from_cache(degree, 200)
    thetas, alphas, alpha0 = _thetas_alphas(expr, 200)
    format_str = '{:+.19e}'
    correct_coeffs = part_frac_coeffs[degree]
    # Thetas in the paper are negative what we have, and are only counted once
    # per conjugate.
    thetas = [-i for i in thetas if im(-i) >= 0]
    theta = sorted(thetas, key=im)[idx]
    real_theta, imag_theta = sorted(zip(correct_coeffs['thetas']['real'],
        correct_coeffs['thetas']['imaginary']), key=lambda i: float(i[1]))[idx]

    real, imag = theta.as_real_imag()
    if real_imag == 'real':
        assert format_str.format(decimal.Decimal(repr(real))) == real_theta
    else:
        assert format_str.format(decimal.Decimal(repr(imag))) == imag_theta
