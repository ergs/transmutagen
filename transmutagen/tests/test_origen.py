import os
from itertools import combinations

import numpy as np

from ..tape9utils import origen_to_name

DATA_DIR = os.path.abspath(os.path.join(__file__, os.path.pardir,
    os.path.pardir, os.path.pardir, 'docker', 'data'))

def load_data(datafile):
    with open(datafile) as f:
        return eval(f.read(), {'array': np.array})

def test_data():
    for datafile in os.listdir(DATA_DIR):
        data = load_data(os.path.join(DATA_DIR, datafile))

        tape9, time, nuc, phi = datafile.split()[0]

        assert 'table_4' in data
        assert 'nuclide' in data['table_4']

        nuclides = data['table_4']['nuclides']

        keys = ['activation_products', 'actinides', 'fission_products']
        # Sanity check
        for comb in combinations(keys, 2):
            assert set.intersection(*comb) == set()
