import os
from itertools import combinations

import numpy as np
from scipy.sparse import csr_matrix
from sympy import symbols, lambdify
import pyne.material
import pyne.nucname

from ..partialfrac import (thetas_alphas, thetas_alphas_to_expr_complex, t,
    multiply_vector)
from ..codegen import MatrixNumPyPrinter, scipy_translations_autoeye
from ..util import load_sparse_csr
from .test_transmute import get_CRAM_from_cache

DATA_DIR = os.path.abspath(os.path.join(__file__, os.path.pardir,
    os.path.pardir, os.path.pardir, 'docker', 'data'))

NUCLIDE_KEYS = ['activation_products', 'actinides', 'fission_products']
# NUCLIDE_KEYS = ['fission_products']

def load_data(datafile):
    with open(datafile) as f:
        return eval(f.read(), {'array': np.array, 'pyne': pyne})

def origen_data_to_array(data, nucs):
    material = data['materials'][1]
    new_data = np.zeros((len(nucs), 1))
    nuc_to_idx = {v: i for i, v in enumerate(nucs)}

    for nuc, atom_frac in material.to_atom_frac().items():
        new_data[nuc_to_idx[pyne.nucname.name(nuc)]] = atom_frac

    return new_data

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

    for datafile in os.listdir(DATA_DIR):
        origen_data = load_data(os.path.join(DATA_DIR, datafile))
        tape9, time, nuc, phi = os.path.splitext(datafile)[0].split()


        npzfilename = os.path.join('data', os.path.splitext(tape9)[0] + '_' + phi + '.npz')
        nucs, mat = load_sparse_csr(npzfilename)

        nuc_to_idx = {v: i for i, v in enumerate(nucs)}
        b = csr_matrix(([1], [[nuc_to_idx[nuc]], [0]]), shape=[mat.shape[1],
        1])

        CRAM_res = np.asarray(e_complex(-mat.T*float(time), b))

        ORIGEN_res = origen_data_to_array(origen_data, nucs)

        import pudb;pudb.set_trace()
        np.testing.assert_allclose(CRAM_res, ORIGEN_res)
