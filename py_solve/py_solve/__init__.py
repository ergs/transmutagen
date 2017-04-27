try:
    from . import py_solve
    del py_solve
except ImportError:
    raise ImportError("Run 'python -m transmutagen.gensolve --py-solve' and 'setup.py build_ext --inplace' to generate the module")

from .py_solve import (N, NNZ, IJ, NUCS, NUCS_IDX, ROWS, COLS, ones,
    flatten_sparse_matrix, csr_from_flat, asflat, solve, diag_add, dot, scalar_times_vector)

__all__ = ['N', 'NNZ', 'IJ', 'NUCS', 'NUCS_IDX', 'ROWS', 'COLS', 'ones',
    'flatten_sparse_matrix', 'csr_from_flat', 'asflat', 'solve', 'diag_add',
    'dot', 'scalar_times_vector']
