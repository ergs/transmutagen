import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import pytest

try:
    import transmutagen.py_solve as solver
except ImportError:
    pytestmark = pytest.mark.skipif(True, reason='transmutagen.py_solve not found')

DTYPES = ['f8', np.complex128]


@pytest.mark.parametrize('dtype', DTYPES)
def test_identity_ones(dtype):
    b = np.ones(solver.N, dtype=dtype)
    mat = sp.eye(solver.N, format='csr', dtype=dtype)
    obs = solver.solve(mat, b)
    exp = spla.spsolve(mat, b)
    assert np.allclose(exp, obs)


@pytest.mark.parametrize('dtype', DTYPES)
def test_identity_range(dtype):
    b = np.arange(solver.N, dtype=dtype)
    mat = sp.eye(solver.N, format='csr', dtype=dtype)
    obs = solver.solve(mat, b)
    exp = spla.spsolve(mat, b)
    assert np.allclose(exp, obs)


@pytest.mark.parametrize('dtype', DTYPES)
def test_ones_ones(dtype):
    b = np.ones(solver.N, dtype=dtype)
    mat = solver.ones(dtype=dtype) + 9*sp.eye(solver.N, format='csr', dtype=dtype)
    obs = solver.solve(mat, b)
    exp = spla.spsolve(mat, b)
    assert np.allclose(exp, obs)


@pytest.mark.parametrize('dtype', DTYPES)
def test_ones_range(dtype):
    b = np.arange(solver.N, dtype=dtype)
    mat = solver.ones(dtype=dtype) + 9*sp.eye(solver.N, format='csr', dtype=dtype)
    obs = solver.solve(mat, b)
    exp = spla.spsolve(mat, b)
    assert np.allclose(exp, obs)


@pytest.mark.parametrize('dtype', DTYPES)
def test_range_range(dtype):
    b = np.arange(solver.N, dtype=dtype)
    mat = solver.ones(dtype=dtype) + sp.diags([b], offsets=[0], shape=(solver.N, solver.N),
                                              format='csr', dtype=dtype)
    obs = solver.solve(mat, b)
    exp = spla.spsolve(mat, b)
    assert np.allclose(exp, obs)



