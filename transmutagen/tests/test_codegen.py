import numpy as np
from numpy.testing import assert_array_equal

from pytest import raises

from ..codegen import autoeye

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

    assert_array_equal(e @ np.array([1, 2]), np.array([2, 4]))
    assert_array_equal(e @ np.array([[1, 2]]), np.array([[2, 4]]))
    assert_array_equal(e @ np.array([[1], [2]]), np.array([[2], [4]]))

    assert_array_equal(np.array([[1, 2]]) @ e, np.array([[2, 4]]))
    assert_array_equal(np.array([[1], [2]]) @ e, np.array([[2], [4]]))

    raises(ValueError, lambda: np.array([1, 2]) @ e)
    raises(ValueError, lambda: e @ np.array([[[1, 2]]]))

    res = e @ np.array([[1, 2], [3, 4]])
    assert_array_equal(res, np.array([[2, 4], [6, 8]]))
    assert res.dtype == int

    res = np.array([[1, 2], [3, 4]]) @ e
    assert_array_equal(res, np.array([[2, 4], [6, 8]]))
    assert res.dtype == int

    res = e @ np.array([[1., 2.], [3., 4.]])
    assert_array_equal(res, np.array([[2., 4.], [6., 8.]]))
    assert res.dtype == float

    res = e @ np.matrix([[1, 2], [3, 4]])
    assert_array_equal(res, np.matrix([[2., 4.], [6., 8.]]))
    assert res.dtype == int
    assert isinstance(res, np.matrix)

    res = np.matrix([[1, 2], [3, 4]]) @ e
    assert_array_equal(res, np.matrix([[2, 4], [6, 8]]))
    assert res.dtype == int
    assert isinstance(res, np.matrix)

    res = e @ e
    assert res == autoeye(4)
