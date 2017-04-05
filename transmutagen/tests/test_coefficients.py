import pytest

slow = pytest.mark.skipif(
    not pytest.config.getoption("--runslow"),
    reason="need --runslow option to run"
)

TOTAL_DEGREES = 27

from .crv_coeffs import coeffs as correct_coeffs
# TODO: Should we use the CRAM cache here?
from ..cram import CRAM_exp, CRAM_coeffs

@slow
@pytest.mark.parametrize('degree', range(1, TOTAL_DEGREES+1))
def test_coefficients(degree):
    generated_coeffs = {}
    expr = CRAM_exp(degree, 30)
    generated_coeffs[degree] = CRAM_coeffs(expr, 20,
        decimal_rounding=True)
    assert generated_coeffs[degree] == correct_coeffs[degree]
