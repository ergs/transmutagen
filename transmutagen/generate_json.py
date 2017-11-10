"""
Regenerate the gensolve_json from ORIGEN with

    python -m transmutagen.generate_json /path/to/origen/libs/
"""

import os
import json
from argparse import ArgumentParser
from operator import itemgetter

from scipy.sparse import eye, csr_matrix
import numpy as np

from .tape9utils import normalize_tape9s, tape9_to_sparse, THRESHOLD

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


def generate_json(tape9s, decaylib, outfile='transmutagen/data/gensolve.json',
    threshold=THRESHOLD):
    mats, nucs = tape9_to_sparse(tape9s, phi=1.0, format='csr',
        decaylib=decaylib, threshold=threshold)
    mat = common_mat(mats)
    ij = csr_ij(mat)
    fromto = [(nucs[j], nucs[i]) for i, j in sorted(ij, key=itemgetter(1))]
    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    with open(outfile, 'w') as f:
        print("Writing", outfile)
        json.dump({
            'nucs': list(nucs),
            # JSON associative arrays can only have string keys
            'fromto': fromto,
            }, f, sort_keys=True, indent=4)

def main(args=None):
    p = ArgumentParser('generate_json', description="""Generate the JSON input
        file for gensolve from TAPE 9 files.""")
    p.add_argument('tape9s', nargs='+', help="""Paths to the TAPE9 files. If a
        path is a directory, a set of default libraries will be gathered from
        that directory (transmutagen.origen_all.ALL_LIBS)""")
    p.add_argument('-d', '--decay', help='path to the decay file, if needed',
                   default='decay.lib', dest='decaylib')
    p.add_argument('--outfile', default='transmutagen/data/gensolve.json',
        help="""File to save the JSON file to. The default is %(default)r.""")
    p.add_argument('--threshold', default=THRESHOLD, help="""Cutoff for
        ignoring reactions.""", type=float)

    ns = p.parse_args(args=args)
    tape9s = normalize_tape9s(ns.tape9s)
    generate_json(tape9s, ns.decaylib, outfile=ns.outfile, threshold=ns.threshold)


if __name__ == "__main__":
    main()
