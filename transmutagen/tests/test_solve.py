import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import pytest

solver = pytest.importorskip('transmutagen.py_solve')


def sparse_ones():
    data = np.ones(solver.NNZ, 'f8')
    rows = np.empty(solver.NNZ, 'i4')
    cols = np.empty(solver.NNZ, 'i4')
    for n, (i, j) in enumerate(solver.IJ):
        rows[n] = i
        cols[n] = j
    mat = sp.csr_matrix((data, (rows, cols)))
    return mat


def test_identity_ones():
    b = np.ones(solver.N, 'f8')
    mat = sp.eye(solver.N, format='csr')
    obs = solver.solve(mat, b)
    exp = spla.spsolve(mat, b)
    assert np.allclose(exp, obs)


def test_identity_range():
    b = np.arange(solver.N, dtype='f8')
    mat = sp.eye(solver.N, format='csr')
    obs = solver.solve(mat, b)
    exp = spla.spsolve(mat, b)
    assert np.allclose(exp, obs)


def test_ones_ones():
    b = np.ones(solver.N, 'f8')
    mat = sparse_ones() + 9*sp.eye(solver.N, format='csr')
    obs = solver.solve(mat, b)
    exp = spla.spsolve(mat, b)
    print()
    print("exp: ", exp[:100])
    print("obs: ", obs[:100])
    assert np.allclose(exp, obs)
