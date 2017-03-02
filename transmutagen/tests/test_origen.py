import os
from itertools import combinations

import numpy as np
from scipy.sparse import csr_matrix
from sympy import symbols, lambdify

from ..partialfrac import (thetas_alphas, thetas_alphas_to_expr_complex, t,
    multiply_vector)
from ..codegen import MatrixNumPyPrinter, scipy_translations_autoeye, get_CRAM_from_cache
from ..util import load_sparse_csr
from ..origen import (load_data, NUCLIDE_KEYS, origen_data_to_array_weighted,
    origen_data_to_array_materials, DATA_DIR)

def test_data_sanity():
    for datafile in os.listdir(DATA_DIR):
        data = load_data(os.path.join(DATA_DIR, datafile))
        tape9, time, nuc, phi = os.path.splitext(datafile)[0].split()

        for table in ['table_4', 'table_5']:
            assert table in data
            assert 'nuclide' in data[table]

            nuclide = data['table_4']['nuclide']

            # Sanity check
            for comb in combinations(NUCLIDE_KEYS, 2):
                a, b = comb
                for common in set.intersection(set(nuclide[a]), set(nuclide[b])):
                    array_a, array_b = nuclide[a][common], nuclide[b][common]
                    assert np.allclose(array_a, 0) \
                        or np.allclose(array_b, 0)
                        # or np.allclose(array_a, array_b)
