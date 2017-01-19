from sympy import together, expand_complex, re, im, symbols

import numpy as np
from numpy.testing import assert_array_equal

from pytest import raises

from ..partialfrac import t, autoeye

def test_re_form():
    theta, alpha = symbols('theta, alpha')

    # Check that this doesn't change
    re_form = together(expand_complex(re(alpha/(t - theta))))
    assert re_form == (t*re(alpha) - re(alpha)*re(theta) -
        im(alpha)*im(theta))/((t - re(theta))**2 + im(theta)**2)

def test_autoeye():
    e = autoeye(2)

    assert e == autoeye(2)
    assert e != autoeye(3)

    raises(ValueError, lambda: e + np.array([1]))
    raises(ValueError, lambda: e + np.array([[1, 2]]))

    res = e + np.array([[1, 2], [3, 4]])
    assert_array_equal(res, np.array([[3, 2], [3, 6]]))
    assert res.dtype == int

    res = np.array([[1, 2], [3, 4]]) + e
    assert_array_equal(res, np.array([[3, 2], [3, 6]]))
    assert res.dtype == int

    res = e + np.array([[1., 2.], [3., 4.]])
    assert_array_equal(res, np.array([[3., 2.], [3., 6.]]))
    assert res.dtype == float

    res = e + np.matrix([[1, 2], [3, 4]])
    assert_array_equal(res, np.matrix([[3, 2], [3, 6]]))
    assert res.dtype == int
    assert isinstance(res, np.matrix)

    res = np.matrix([[1, 2], [3, 4]]) + e
    assert_array_equal(res, np.matrix([[3, 2], [3, 6]]))
    assert res.dtype == int
    assert isinstance(res, np.matrix)

    assert e + 1 == 1 + e == autoeye(3)
    assert 2*e == e*2 == autoeye(4)
    assert e + e == autoeye(4)
    assert e*e == autoeye(4)
