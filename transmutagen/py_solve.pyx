cimport numpy as np
import numpy as np

import scipy.sparse as sp

cimport c_solve

# startup numpy
np.import_array()
np.import_ufunc()


def solve(A, b):
    """Solves Ax = b for x."""
    if not sp.isspmatrix_csr(A):
        A = A.tocsr(copy=False)
    #ctypedef np.float64
    cdef np.ndarray[np.float64_t] x = np.empty(A.shape[0], dtype=np.float64)
    c_solve.transmutagen_solve_double(<double*> np.PyArray_DATA(A.data),
                                      <double*> np.PyArray_DATA(b),
                                      <double*> np.PyArray_DATA(x))
    return x
