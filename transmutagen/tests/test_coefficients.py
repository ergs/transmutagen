import pytest

slow = pytest.mark.skipif(
    not pytest.config.getoption("--runslow"),
    reason="need --runslow option to run"
)

TOTAL_DEGREES = 30

from .crv_coeffs import coeffs as correct_coeffs
from ..cram import get_CRAM_from_cache, CRAM_coeffs

@slow
@pytest.mark.parametrize('degree', range(1, TOTAL_DEGREES+1))
def test_coefficients(degree):
    generated_coeffs = {}
    expr = get_CRAM_from_cache(degree, 200)
    generated_coeffs[degree] = CRAM_coeffs(expr, 20,
        decimal_rounding=True)
    # pytest won't show the full expr from the assert, so we print it too
    print(expr)
    assert generated_coeffs[degree] == correct_coeffs[degree], expr
