cimport numpy as np
import numpy as np

import scipy.sparse as sp

cimport c_solve

# startup numpy
np.import_array()
np.import_ufunc()


# some info translations
cdef int i, j, idx
N = c_solve.transmutagen_info.n
NNZ = c_solve.transmutagen_info.nnz
cpdef dict C_IJ = {}
for idx in range(c_solve.transmutagen_info.nnz):
    C_IJ[c_solve.transmutagen_info.i[idx], c_solve.transmutagen_info.j[idx]] = idx
IJ = C_IJ
cpdef list C_NUCS = []
for idx in range(c_solve.transmutagen_info.n):
    b = c_solve.transmutagen_info.nucs[idx]
    s = b.decode()
    C_NUCS.append(s)
NUCS= C_NUCS
NUCS_IDX = {nuc: idx for idx, nuc in enumerate(NUCS)}


def solve(A, b):
    """Solves Ax = b for x."""
    if not sp.isspmatrix_csr(A):
        A = A.tocsr(copy=True)
    cdef np.ndarray[np.float64_t] x = np.empty(c_solve.transmutagen_info.n, dtype=np.float64)
    c_solve.transmutagen_solve_double(<double*> np.PyArray_DATA(A.data),
                                      <double*> np.PyArray_DATA(b),
                                      <double*> np.PyArray_DATA(x))
    return x
