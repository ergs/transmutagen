import decimal

import pytest
from sympy import im

slow = pytest.mark.skipif(
    not pytest.config.getoption("--runslow"),
    reason="need --runslow option to run"
)

TOTAL_DEGREES = 30

from .crv_coeffs import coeffs as correct_coeffs
from .pusa_coeffs import part_frac_coeffs
from ..cram import get_CRAM_from_cache, CRAM_coeffs
from ..partialfrac import thetas_alphas

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

@pytest.mark.xfail
@pytest.mark.parametrize('real_imag', ['real', 'imag'])
@pytest.mark.parametrize('idx', range(16//2))
@pytest.mark.parametrize('typ', ['thetas', 'alphas', 'alpha0'])
@pytest.mark.parametrize('degree', [14, 16])
def test_partial_fraction_coefficients(degree, typ, idx, real_imag):
    if idx >= degree//2:
        # How do I do this in the decorators?
        return
    expr = get_CRAM_from_cache(degree, 200)
    thetas, alphas, alpha0 = thetas_alphas(expr, 200)
    format_str = '{:+.19e}'
    paper_coeffs = part_frac_coeffs[degree]
    # Thetas and alphas in the paper are negative what we have, and are only
    # counted once per conjugate.
    if typ == 'thetas':
        vals = [-i for i in thetas if im(-i) >= 0]
    elif typ == 'alphas':
        vals = [-j for i, j in zip(thetas, alphas) if im(-i) >= 0]
    elif typ == 'alpha0':
        if idx != 0:
            return
        vals = [alpha0]
    val = sorted(vals, key=im)[idx]
    real_val_paper, imag_val_paper = sorted(zip(paper_coeffs[typ]['real'],
        paper_coeffs[typ]['imaginary']), key=lambda i: float(i[1]))[idx]

    real_val, imag_val = val.as_real_imag()
    if real_imag == 'real':
        assert format_str.format(decimal.Decimal(repr(real_val))) == real_val_paper
    else:
        assert format_str.format(decimal.Decimal(repr(imag_val))) == imag_val_paper
