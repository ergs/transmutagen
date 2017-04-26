"""
Regenerate the gensolve_json from ORIGEN with

    python -m transmutagen.generate_json /path/to/origen/libs/
"""

import os
import json
from argparse import ArgumentParser

from scipy.sparse import eye, csr_matrix
import numpy as np

from .tape9utils import normalize_tape9s, tape9_to_sparse

def csr_ij(mat):
    ij = {}
    for i, l, u in zip(range(mat.shape[0]), mat.indptr[:-1], mat.indptr[1:]):
        for p in range(l, u):
            ij[i, int(mat.indices[p])] = p
    return ij

def common_mat(mats):
    assert len({i.shape for i in mats}) == 1
    mats = [i.tocoo() for i in mats] + [eye(mats[0].shape[0], format='coo')]
    rows = np.hstack([i.row for i in mats])
    cols = np.hstack([i.col for i in mats])
    data = np.ones(len(rows))
    return csr_matrix((data, (rows, cols)))


def generate_json(tape9s, decaylib, file='data/gensolve.json'):
    mats, nucs = tape9_to_sparse(tape9s, phi=1.0, format='csr', decaylib=decaylib)
    mat = common_mat(mats)
    N = mat.shape[0]
    ij = csr_ij(mat)
    os.makedirs(os.path.dirname(file), exist_ok=True)

    with open(file, 'w') as f:
        print("Writing", file)
        json.dump({
            'nucs': list(nucs),
            'N': N,
            # JSON associative arrays can only have string keys
            'ij': sorted(ij.items()),
            }, f, sort_keys=True)

def main(args=None):
    p = ArgumentParser('generate_json', description="""Generate the JSON input
    file for gensolve from TAPE 9 files.""")
    p.add_argument('tape9s', nargs='+', help="""Paths to the TAPE9 files. If a
    path is a directory, a set of default libraries will be gathered from that
    directory (transmutagen.origen_all.ALL_LIBS)""")
    p.add_argument('-d', '--decay', help='path to the decay file, if needed',
                   default='decay.lib', dest='decaylib')
    p.add_argument('--file', default='data/gensolve.json')

    ns = p.parse_args(args=args)
    tape9s = normalize_tape9s(ns.tape9s)
    generate_json(tape9s, ns.decaylib, file=ns.file)


if __name__ == "__main__":
    main()
