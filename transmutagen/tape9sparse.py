"""Converts a TAPE9 file to a sparse matrix on disk."""
import os
from argparse import ArgumentParser

from .util import save_sparse_csr
from .tape9utils import tape9_to_sparse, THRESHOLD


def make_parser():
    p = ArgumentParser('tape9sparse')
    p.add_argument('tape9', help="path to the TAPE9 file.")
    p.add_argument('phi', help='the neutron flux in [n/cm^2/sec]',
                   type=float)
    p.add_argument('-f', '--format', help='The sparse matrix format',
                   default='csr', dest='format')
    p.add_argument('-d', '--decay', help='path to the decay file, if needed',
                   default='decay.lib', dest='decaylib')
    p.add_argument('--include-fission', action='store_true', default=True,
                   dest='include_fission',
                   help='Include fission reactions in the matrix.')
    p.add_argument('--dont-include-fission', action='store_false',
                   dest='include_fission',
                   help="Don't include fission reactions in the matrix.")
    p.add_argument('-t', '--threshold', default=THRESHOLD, dest='threshold',
                   help='cutoff for ignoring reactions', type=float)
    p.add_argument('-o', '--output', dest='output', default=None,
                   help='The filename to write the matrix to, in npz format.')
    return p


def main(args=None):
    p = make_parser()
    ns = p.parse_args(args=args)
    if ns.output is None:
        base = os.path.basename(ns.tape9)
        base, _ = os.path.splitext(base)
        ns.output = base + '.npz'
    if ns.format != 'csr':
        raise ValueError('Only the CSR format is currently available from the '
                         'command line interface.')
    mat, nucs = tape9_to_sparse(ns.tape9, ns.phi, format=ns.format,
                                decaylib=ns.decaylib,
                                include_fission=ns.include_fission,
                                threshold=ns.threshold)
    save_sparse_csr(ns.output, mat, nucs)


if __name__ == '__main__':
    main()
