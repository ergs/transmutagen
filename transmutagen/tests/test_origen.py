import os
from itertools import combinations

import numpy as np

from ..tape9utils import origen_to_name

DATA_DIR = os.path.abspath(os.path.join(__file__, os.path.pardir,
    os.path.pardir, os.path.pardir, 'docker', 'data'))

def load_data(datafile):
    with open(datafile) as f:
        return eval(f.read(), {'array': np.array})

def test_data_sanity():
    for datafile in os.listdir(DATA_DIR):
        data = load_data(os.path.join(DATA_DIR, datafile))

        tape9, time, nuc, phi = datafile.split()

        assert 'table_4' in data
        assert 'nuclide' in data['table_4']

        nuclide = data['table_4']['nuclide']

        keys = ['activation_products', 'actinides', 'fission_products']
        # Sanity check
        for comb in combinations(keys, 2):
            a, b = comb
            for common in set.intersection(set(nuclide[a]), set(nuclide[b])):
                array_a, array_b = nuclide[a][common], nuclide[b][common]
                assert np.allclose(array_a, 0) \
                    or np.allclose(array_b, 0)
                    # or np.allclose(array_a, array_b)
