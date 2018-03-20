import sys
import os

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .. import N, solve, ones, flatten_sparse_matrix, diag_add, dot

import pytest

this_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(this_dir, os.pardir))

DTYPES = ['f8', np.complex128]


@pytest.mark.parametrize('dtype', DTYPES)
def test_solve_identity_ones(dtype):
    b = np.ones(N, dtype=dtype)
    mat = sp.eye(N, format='csr', dtype=dtype)
    obs = solve(mat, b)
    exp = spla.spsolve(mat, b)
    assert np.allclose(exp, obs)


@pytest.mark.parametrize('dtype', DTYPES)
def test_solve_identity_range(dtype):
    b = np.arange(N, dtype=dtype)
    mat = sp.eye(N, format='csr', dtype=dtype)
    obs = solve(mat, b)
    exp = spla.spsolve(mat, b)
    assert np.allclose(exp, obs)


@pytest.mark.parametrize('dtype', DTYPES)
def test_solve_ones_ones(dtype):
    b = np.ones(N, dtype=dtype)
    mat = ones(dtype=dtype) + 9*sp.eye(N, format='csr', dtype=dtype)
    obs = solve(mat, b)
    exp = spla.spsolve(mat, b)
    assert np.allclose(exp, obs)


@pytest.mark.parametrize('dtype', DTYPES)
def test_solve_ones_range(dtype):
    b = np.arange(N, dtype=dtype)
    mat = ones(dtype=dtype) + 9*sp.eye(N, format='csr', dtype=dtype)
    obs = solve(mat, b)
    exp = spla.spsolve(mat, b)
    assert np.allclose(exp, obs)


@pytest.mark.parametrize('dtype', DTYPES)
def test_solve_range_range(dtype):
    b = np.arange(N, dtype=dtype)
    mat = ones(dtype=dtype) + sp.diags([b], offsets=[0], shape=(N, N),
                                              format='csr', dtype=dtype)
    obs = solve(mat, b)
    exp = spla.spsolve(mat, b)
    assert np.allclose(exp, obs)


@pytest.mark.parametrize('dtype', DTYPES)
def test_diag_add(dtype):
    mat = ones(dtype=dtype)
    res = mat + 9*sp.eye(N, format='csr', dtype=dtype)
    exp = flatten_sparse_matrix(res)
    obs = diag_add(mat, 9.0)
    assert np.allclose(exp, obs)


@pytest.mark.parametrize('dtype', DTYPES)
def test_dot(dtype):
    x = np.arange(N, dtype=dtype)
    mat = ones(dtype=dtype) + 9*sp.eye(N, format='csr', dtype=dtype)
    exp = mat.dot(x)
    obs = dot(mat, x)
    assert np.allclose(exp, obs)
