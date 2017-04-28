import sys
import os

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import py_solve as solver

import pytest

this_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(this_dir, os.pardir))

DTYPES = ['f8', np.complex128]


@pytest.mark.parametrize('dtype', DTYPES)
def test_solve_identity_ones(dtype):
    b = np.ones(solver.N, dtype=dtype)
    mat = sp.eye(solver.N, format='csr', dtype=dtype)
    obs = solver.solve(mat, b)
    exp = spla.spsolve(mat, b)
    assert np.allclose(exp, obs)


@pytest.mark.parametrize('dtype', DTYPES)
def test_solve_identity_range(dtype):
    b = np.arange(solver.N, dtype=dtype)
    mat = sp.eye(solver.N, format='csr', dtype=dtype)
    obs = solver.solve(mat, b)
    exp = spla.spsolve(mat, b)
    assert np.allclose(exp, obs)


@pytest.mark.parametrize('dtype', DTYPES)
def test_solve_ones_ones(dtype):
    b = np.ones(solver.N, dtype=dtype)
    mat = solver.ones(dtype=dtype) + 9*sp.eye(solver.N, format='csr', dtype=dtype)
    obs = solver.solve(mat, b)
    exp = spla.spsolve(mat, b)
    assert np.allclose(exp, obs)


@pytest.mark.parametrize('dtype', DTYPES)
def test_solve_ones_range(dtype):
    b = np.arange(solver.N, dtype=dtype)
    mat = solver.ones(dtype=dtype) + 9*sp.eye(solver.N, format='csr', dtype=dtype)
    obs = solver.solve(mat, b)
    exp = spla.spsolve(mat, b)
    assert np.allclose(exp, obs)


@pytest.mark.parametrize('dtype', DTYPES)
def test_solve_range_range(dtype):
    b = np.arange(solver.N, dtype=dtype)
    mat = solver.ones(dtype=dtype) + sp.diags([b], offsets=[0], shape=(solver.N, solver.N),
                                              format='csr', dtype=dtype)
    obs = solver.solve(mat, b)
    exp = spla.spsolve(mat, b)
    assert np.allclose(exp, obs)


@pytest.mark.parametrize('dtype', DTYPES)
def test_diag_add(dtype):
    mat = solver.ones(dtype=dtype)
    res = mat + 9*sp.eye(solver.N, format='csr', dtype=dtype)
    exp = solver.flatten_sparse_matrix(res)
    obs = solver.diag_add(mat, 9.0)
    assert np.allclose(exp, obs)


@pytest.mark.parametrize('dtype', DTYPES)
def test_dot(dtype):
    x = np.arange(solver.N, dtype=dtype)
    mat = solver.ones(dtype=dtype) + 9*sp.eye(solver.N, format='csr', dtype=dtype)
    exp = mat.dot(x)
    obs = solver.dot(mat, x)
    assert np.allclose(exp, obs)
