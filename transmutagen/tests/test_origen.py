import os
from itertools import combinations

import numpy as np
from scipy.sparse import csr_matrix
from sympy import symbols, lambdify

from ..partialfrac import (thetas_alphas, thetas_alphas_to_expr_complex, t,
    multiply_vector)
from ..codegen import MatrixNumPyPrinter, scipy_translations_autoeye
from ..util import load_sparse_csr
from .test_transmute import get_CRAM_from_cache
from ..origen import (load_data, NUCLIDE_KEYS, origen_data_to_array_weighted, origen_data_to_array_materials)

DATA_DIR = os.path.abspath(os.path.join(__file__, os.path.pardir,
    os.path.pardir, os.path.pardir, 'docker', 'data'))

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

def test_origen_against_CRAM():
    rat_func = get_CRAM_from_cache(14, 30)
    thetas, alphas, alpha0 = thetas_alphas(rat_func, 30)
    part_frac_complex = thetas_alphas_to_expr_complex(thetas, alphas, alpha0)
    n0 = symbols("n0", commutative=False)

    e_complex = lambdify((t, n0), multiply_vector(part_frac_complex, n0),
        scipy_translations_autoeye, printer=MatrixNumPyPrinter({'use_autoeye': True
            }))

    # ORIGEN returns 3 digits
    rtol=1e-3
    # ORIGEN stops the taylor expansion with the error term
    # exp(ASUM)*ASUM**n/n! (using Sterling's approximation), where n =
    # 3.5*ASUM + 6 and ASUM is the max of the column sums. The max of the
    # column sums is ~2 because of fission, giving ~1e-5 (see ORIGEN lines
    # 5075-5100)
    atol=1e-5

    for datafile in os.listdir(DATA_DIR):
        origen_data = load_data(os.path.join(DATA_DIR, datafile))
        tape9, time, nuc, phi = os.path.splitext(datafile)[0].split()
        print("Analyzing", tape9, "at time=", time, "nuc=", nuc, "phi=", phi)
        print('-'*80)

        npzfilename = os.path.join('data', os.path.splitext(tape9)[0] + '_' + phi + '.npz')
        nucs, mat = load_sparse_csr(npzfilename)

        nuc_to_idx = {v: i for i, v in enumerate(nucs)}
        b = csr_matrix(([1], [[nuc_to_idx[nuc]], [0]]), shape=[mat.shape[1],
        1])

        CRAM_res = np.asarray(e_complex(-mat.T*float(time), b))
        CRAM_res_normalized = CRAM_res/np.sum(CRAM_res)

        ORIGEN_res_weighted = origen_data_to_array_weighted(origen_data, nucs,)
        ORIGEN_res_materials = origen_data_to_array_materials(origen_data, nucs)
        # ORIGEN_res_atom_fraction = origen_data_to_array_atom_fraction(origen_data, nucs)

        for C, O, units in [
            (CRAM_res, ORIGEN_res_weighted, 'atom fractions'),
            (CRAM_res_normalized, ORIGEN_res_materials, 'mass fractions'),
            # (CRAM_res_normalized, ORIGEN_res_atom_fraction, 'atom fraction'),
            ]:

            print("Units:", units)
            try:
                np.testing.assert_allclose(C, O, rtol=rtol, atol=atol)
            except AssertionError as e:
                print(e)
                print("Mismatching elements sorted by error (CRAM, ORIGEN, symmetric relative error)")
                A = np.isclose(C, O, rtol=rtol, atol=atol)
                rel_error = abs(C - O)/(C + O)
                for i, in np.argsort(rel_error, axis=0)[::-1]:
                    if A[i]:
                        continue
                    print(nucs[i], C[i], O[i], rel_error[i])
            else:
                print("Arrays match with rtol=", rtol, "atol=", atol)

            print()

if __name__ == '__main__':
    test_origen_against_CRAM()
