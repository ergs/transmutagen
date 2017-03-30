import numpy as np
from numpy.testing import assert_array_equal

from sympy import symbols, I, S

from pytest import raises

from ..codegen import autoeye, MatrixNumPyPrinter
from ..partialfrac import customre

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

    assert_array_equal(e @ np.array([[1, 2]]), np.array([[2, 4]]))
    assert_array_equal(e @ np.array([[1], [2]]), np.array([[2], [4]]))

    assert_array_equal(np.array([[1, 2]]) @ e, np.array([[2, 4]]))
    assert_array_equal(np.array([[1], [2]]) @ e, np.array([[2], [4]]))

    raises(ValueError, lambda: e @ np.array([1, 2]))
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

def test_MatrixNumPyPrinter():
    t = symbols('t', positive=True)

    Mautoeye = MatrixNumPyPrinter({'use_autoeye': True}).doprint
    Mnoautoeye = MatrixNumPyPrinter({'use_autoeye': False}).doprint
    Mpy_solve = MatrixNumPyPrinter({'py_solve': True}).doprint
    raises(ValueError, lambda: MatrixNumPyPrinter({'py_solve': True,
        'use_autoeye': True}))

    assert Mautoeye(t**2) == Mnoautoeye(t**2) == 'matrix_power(t, 2)'
    raises(NotImplementedError, lambda: Mpy_solve(t**2))

    assert Mautoeye(S(1)) == 'autoeye(1)'
    assert Mnoautoeye(S(1)) == Mpy_solve(S(1)) == '1'

    assert Mautoeye(I) == 'autoeye(1j)'
    assert Mnoautoeye(I) == Mpy_solve(I) == '1j'

    assert Mautoeye(S(2.0)) == 'autoeye(2.00000000000000)'
    assert Mnoautoeye(S(2.0)) == Mpy_solve(S(2.0)) == '2.00000000000000'

    assert Mautoeye(customre(t)) == Mautoeye(customre(t)) == \
        Mpy_solve(customre(t)) == 'real(t)'

    assert Mautoeye(2*I) == 'autoeye(2*1j)'
    # assert Mautoeye(2*I) == 'autoeye(2j)'
    assert Mnoautoeye(2*I) == Mpy_solve(2*I) == '2*1j'
    # assert Mnoautoeye(2*I) == Mpy_solve(2*I) == '2j'

    assert Mautoeye(1 + I) == 'autoeye(1 + 1j)'
    assert Mnoautoeye(1 + I) == Mpy_solve(1 + I) == '1 + 1j'

    assert Mautoeye(t + 1 + I) == 't + autoeye(1 + 1j)'
    assert Mnoautoeye(t + 1 + I) == '1 + 1j + t'
    assert Mpy_solve(t + 1 + I) == 'diag_add(t, 1 + 1j)'

    assert Mautoeye(t + t**2 + 1 + I) == 't + matrix_power(t, 2) + autoeye(1 + 1j)'
    assert Mnoautoeye(t + t**2 + 1 + I) == '1 + 1j + t + matrix_power(t, 2)'
    raises(NotImplementedError, lambda: Mpy_solve(t + t**2 + 1 + I))

    assert Mautoeye(t + t**2 + I) == 't + matrix_power(t, 2) + autoeye(1j)'
    assert Mnoautoeye(t + t**2 + I) == '1j + t + matrix_power(t, 2)'
    raises(NotImplementedError, lambda: Mpy_solve(t + t**2 + I))

    assert Mautoeye(t + t**2) == Mnoautoeye(t + t**2) == 't + matrix_power(t, 2)'
    raises(NotImplementedError, lambda: Mpy_solve(t + t**2))

    assert Mautoeye(t*(1 + I)) == Mnoautoeye(t*(1 + I)) == \
        Mpy_solve(t*(1 + I)) == '(1 + 1j)*t'

    assert Mautoeye(t*(t + 1 + I)) == 't@(t + autoeye(1 + 1j))'
    assert Mnoautoeye(t*(t + 1 + I)) == 't@(1 + 1j + t)'
    raises(NotImplementedError, lambda: Mpy_solve(t*(t + 1 + I)))

    assert Mautoeye(t*(t + t**2 + 1 + I)) == 't@(t + matrix_power(t, 2) + autoeye(1 + 1j))'
    assert Mnoautoeye(t*(t + t**2 + 1 + I)) == 't@(1 + 1j + t + matrix_power(t, 2))'
    raises(NotImplementedError, lambda: Mpy_solve(t*(t + t**2 + 1 + I)))

    assert Mautoeye(t*(t + t**2 + I)) == 't@(t + matrix_power(t, 2) + autoeye(1j))'
    assert Mnoautoeye(t*(t + t**2 + I)) == 't@(1j + t + matrix_power(t, 2))'
    raises(NotImplementedError, lambda: Mpy_solve(t*(t + t**2 + I)))

    assert Mautoeye(t*(t + t**2)) == Mnoautoeye(t*(t + t**2)) == 't@(t + matrix_power(t, 2))'
    raises(NotImplementedError, lambda: Mpy_solve(t*(t + t**2)))
