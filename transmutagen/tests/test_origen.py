import os

import numpy as np

DATA_DIR = os.path.abspath(os.path.join(__file__, os.path.pardir,
    os.path.pardir, os.path.pardir, 'docker', 'data'))

def load_data(datafile):
    with open(datafile) as f:
        return eval(f.read(), {'array': np.array})

def test_data():
    for datafile in os.listdir(DATA_DIR):
        data = load_data(os.path.join(DATA_DIR, datafile))

        assert 'table_4' in data
