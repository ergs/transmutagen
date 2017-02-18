import os
from itertools import combinations

import numpy as np
from scipy.sparse import csr_matrix
from sympy import symbols, lambdify
import pyne.material

from ..tape9utils import origen_to_name
from ..partialfrac import (thetas_alphas, thetas_alphas_to_expr_complex, t,
    multiply_vector)
from ..codegen import MatrixNumPyPrinter, scipy_translations_autoeye
from ..util import load_sparse_csr
from .test_transmute import get_CRAM_from_cache

DATA_DIR = os.path.abspath(os.path.join(__file__, os.path.pardir,
    os.path.pardir, os.path.pardir, 'docker', 'data'))

NUCLIDE_KEYS = ['activation_products', 'actinides', 'fission_products']

def load_data(datafile):
    import pyne.data
    # Make pyne use naive atomic mass numbers to match ORIGEN
    for i in pyne.data.atomic_mass_map:
        pyne.data.atomic_mass_map[i] = float(pyne.nucname.anum(i))

    with open(datafile) as f:
        return eval(f.read(), {'array': np.array, 'pyne': pyne})

def origen_to_array(origen_dict, nucs):
    new_data = np.zeros((len(nucs), 1))
    nuc_to_idx = {v: i for i, v in enumerate(nucs)}
    for i in origen_dict:
        new_data[nuc_to_idx[origen_to_name(i)]] += origen_dict[i][1]

    return new_data

def origen_data_to_array_weighted(data, nucs, n_fission_fragments=2.004):
    # Table 5 is grams
    table_5_weights = {}
    table_5_nuclide = data['table_5']['nuclide']
    for key in NUCLIDE_KEYS:
        table_5_weights[key] = np.sum(origen_to_array(table_5_nuclide[key], nucs), axis=0)
    table_5_weights['fission_products'] *= n_fission_fragments

    # Table 4 is atom fraction
    table_4_nuclide = data['table_4']['nuclide']
    new_data = np.zeros((len(nucs), 1))
    for key in NUCLIDE_KEYS:
        new_data += table_5_weights[key]*origen_to_array(table_4_nuclide[key], nucs)

    return new_data

def origen_data_to_array_atom_fraction(data, nucs, n_fission_fragments=2):
    # Table 4 is atom fraction
    table_4_nuclide = data['table_4']['nuclide']
    new_data = np.zeros((len(nucs), 1))
    for key in NUCLIDE_KEYS:
        new_data += origen_to_array(table_4_nuclide[key], nucs)

    return new_data

def origen_data_to_array_materials(data, nucs):
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
        CRAM_res_normalized = CRAM_res/np.sum(CRAM_res)

        ORIGEN_res_weighted = origen_data_to_array_weighted(origen_data, nucs,)
        ORIGEN_res_materials = origen_data_to_array_materials(origen_data, nucs)
        ORIGEN_res_atom_fraction = origen_data_to_array_atom_fraction(origen_data, nucs)

        np.testing.assert_allclose(CRAM_res, ORIGEN_res)
