cimport numpy as np
import numpy as np

import scipy.sparse as sp

cimport c_solve

# startup numpy
np.import_array()
np.import_ufunc()


# some info translations
cdef int i, j, idx
cpdef int N = c_solve.transmutagen_info.n
cpdef int NNZ = c_solve.transmutagen_info.nnz
cpdef dict IJ = {}
for idx in range(c_solve.transmutagen_info.nnz):
    IJ[c_solve.transmutagen_info.i[idx], c_solve.transmutagen_info.j[idx]] = idx
cpdef list NUCS = []
for idx in range(c_solve.transmutagen_info.n):
    b = c_solve.transmutagen_info.nucs[idx]
    s = b.decode()
    NUCS.append(s)
cpdef dict NUCS_IDX = {nuc: idx for idx, nuc in enumerate(NUCS)}


def solve(A, b):
    """Solves Ax = b for x."""
    if not sp.isspmatrix_csr(A):
        A = A.tocsr(copy=True)
    cdef np.ndarray[np.float64_t] x = np.empty(A.shape[0], dtype=np.float64)
    c_solve.transmutagen_solve_double(<double*> np.PyArray_DATA(A.data),
                                      <double*> np.PyArray_DATA(b),
                                      <double*> np.PyArray_DATA(x))
    return x
