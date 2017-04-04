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

cdef np.npy_intp npy_nnz = c_solve.transmutagen_info.nnz
ROWS = np.PyArray_SimpleNewFromData(1, &npy_nnz, np.NPY_INT, c_solve.transmutagen_info.i)
COLS = np.PyArray_SimpleNewFromData(1, &npy_nnz, np.NPY_INT, c_solve.transmutagen_info.j)


def ones(dtype='f8'):
    """Returns a CSR matrix of ones with the given sparsity pattern."""
    data = np.ones(c_solve.transmutagen_info.nnz, dtype=dtype)
    mat = sp.csr_matrix((data, (ROWS, COLS)))
    return mat


def flatten_sparse_matrix(mat):
    """Flattens a sparse matrix to a solvable form."""
    rows, cols, vals = sp.find(mat)
    cdef int nmat = len(rows)
    cdef np.ndarray A = np.zeros(c_solve.transmutagen_info.nnz, dtype=mat.dtype)
    cdef int n
    for n in range(nmat):
        idx = C_IJ.get((rows[n], cols[n]), None)
        if idx is not None:
            A[idx] = vals[n]
    return A


def csr_from_flat(A):
    """Converts a flatten matrix into a CSR sparse matrix."""
    return sp.csr_matrix((A, (ROWS, COLS)))


def asflat(A):
    """Returns a flat version of the matrix. Does nothing if the matrix is already flat."""
    if not sp.issparse(A):
        pass
    elif A.nnz != c_solve.transmutagen_info.nnz or not sp.isspmatrix_csr(A):
        A = flatten_sparse_matrix(A)
    else:
        # is CSR with right shape
        A = A.data
    return A


def solve(A, b):
    """Solves Ax = b for x."""
    A = asflat(A)
    b_flat = b.flatten()
    # solve for type
    if A.dtype == np.complex128:
        x = np.empty(c_solve.transmutagen_info.n, dtype=np.complex128)
        c_solve.transmutagen_solve_complex(<double complex*> np.PyArray_DATA(A),
                                           <double complex*> np.PyArray_DATA(b),
                                           <double complex*> np.PyArray_DATA(x))
    elif A.dtype == np.float64:
        x = np.empty(c_solve.transmutagen_info.n, dtype=np.float64)
        c_solve.transmutagen_solve_double(<double*> np.PyArray_DATA(A),
                                          <double*> np.PyArray_DATA(b),
                                          <double*> np.PyArray_DATA(x))
    else:
        raise ValueError("dtype not recognized.")
    x.shape = b.shape
    return x


def diag_add(A, alpha):
    """Returns a flat matrix which represents A + αI."""
    dtype = np.common_type(A, np.array(alpha))
    r = np.array(asflat(A), dtype=dtype)
    if dtype == np.complex128:
        c_solve.transmutagen_diag_add_complex(<double complex*> np.PyArray_DATA(r), alpha)
    elif dtype == np.float64:
        c_solve.transmutagen_diag_add_double(<double*> np.PyArray_DATA(r), alpha)
    else:
        raise ValueError("dtype not recognized.")
    return r


def dot(A, x):
    """Takes the dot product of Ax and returns y."""
    A = asflat(A)
    # solve for type
    if A.dtype == np.complex128:
        y = np.empty(c_solve.transmutagen_info.n, dtype=np.complex128)
        c_solve.transmutagen_dot_complex(<double complex*> np.PyArray_DATA(A),
                                         <double complex*> np.PyArray_DATA(x),
                                         <double complex*> np.PyArray_DATA(y))
    elif A.dtype == np.float64:
        y = np.empty(c_solve.transmutagen_info.n, dtype=np.float64)
        c_solve.transmutagen_dot_double(<double*> np.PyArray_DATA(A),
                                        <double*> np.PyArray_DATA(x),
                                        <double*> np.PyArray_DATA(y))
    else:
        raise ValueError("dtype not recognized.")
    return y

def add7(x0, x1, x2, x3, x4, x5, x6):
    """Takes the dot product of Ax and returns y."""
    if x0.dtype == np.complex128:
        y = np.empty(c_solve.transmutagen_info.n, dtype=np.complex128)
        c_solve.transmutagen_vector_add_7_complex(
            <double complex*> np.PyArray_DATA(x0),
            <double complex*> np.PyArray_DATA(x1),
            <double complex*> np.PyArray_DATA(x2),
            <double complex*> np.PyArray_DATA(x3),
            <double complex*> np.PyArray_DATA(x4),
            <double complex*> np.PyArray_DATA(x5),
            <double complex*> np.PyArray_DATA(x6),
            <double complex*> np.PyArray_DATA(y))
    else:
        raise ValueError("dtype not recognized.")
    return y
