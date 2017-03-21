import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import transmutagen.py_solve as solver


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
    #mat[0, 1] = 0.0
    #mat[2, 1] = 0.0
    obs = solver.solve(mat, b)
    exp = spla.spsolve(mat, b)
    print("exp: ", exp[:100])
    print("obs: ", obs[:100])
    assert np.allclose(exp, obs)

