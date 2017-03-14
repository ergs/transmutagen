"""Converts a TAPE9 file to a sparse matrix on disk."""
import os
import argparse

from .util import save_sparse_csr
from .tape9utils import tape9_to_sparse, THRESHOLD


def make_parser():
    p = argparse.ArgumentParser('tape9sparse', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('tape9', help="path to the TAPE9 file.")
    p.add_argument('--phi', help='the neutron flux in [n/cm^2/sec]',
                   type=float, default=4e14)
    p.add_argument('-f', '--format', help='The sparse matrix format',
                   default='csr', dest='format')
    p.add_argument('-d', '--decay', help='path to the decay file, if needed',
                   default='decay.lib', dest='decaylib')
    p.add_argument('--include-fission', action='store_true', default=True,
                   dest='include_fission',
                   help='Include fission reactions in the matrix.')
    p.add_argument('--no-include-fission', action='store_false',
                   dest='include_fission',
                   help="Don't include fission reactions in the matrix.")
    p.add_argument('-t', '--threshold', default=THRESHOLD, dest='threshold',
                   help='cutoff for ignoring reactions', type=float)
    p.add_argument('-o', '--output', dest='output', default=None,
                   help='The filename to write the matrix to, in npz format.')
    return p

def save_sparse(tape9, phi=4e14, output=None, format='csr',
    decaylib='decay.lib', include_fission=True, threshold=THRESHOLD):
    if output is None:
        base = os.path.basename(tape9)
        base, _ = os.path.splitext(base)
        os.makedirs('data', exist_ok=True)
        fission_part = '' if include_fission else '_nofission'
        output = os.path.join('data', base + '_' + str(phi) + fission_part + '.npz')
    if format != 'csr':
        raise ValueError('Only the CSR format is currently available from the '
                         'command line interface.')
    mat, nucs = tape9_to_sparse(tape9, phi, format=format,
                                decaylib=decaylib,
                                include_fission=include_fission,
                                threshold=threshold)
    save_sparse_csr(output, mat, nucs, phi)

def main(args=None):
    p = make_parser()
    ns = p.parse_args(args=args)
    save_sparse(**vars(ns))

if __name__ == '__main__':
    main()
