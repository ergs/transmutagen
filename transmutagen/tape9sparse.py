"""Converts a TAPE9 file to a sparse matrix on disk."""
import os
import argparse

from .util import save_sparse_csr
from .tape9utils import tape9_to_sparse, normalize_tape9s

def make_parser():
    p = argparse.ArgumentParser('tape9sparse', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('tape9s', nargs='+', help="""Paths to the TAPE9 files. If a
    path is a directory, a set of default libraries will be gathered from that
    directory (transmutagen.origen_all.ALL_LIBS)""")
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
    p.add_argument('--alpha-as-He4', action='store_true', default=False,
                   help="Alpha reactions go to He4")
    p.add_argument('-o', '--output-dir', default=None,
                   help='The directory to write the output files to, in npz format.')
    return p

def save_sparse(tape9s, phi=4e14, output_dir=None, format='csr',
    decaylib='decay.lib', include_fission=True, alpha_as_He4=False):
    if output_dir is None:
        output_dir = 'data'
    os.makedirs(output_dir, exist_ok=True)

    if format != 'csr':
        raise ValueError('Only the CSR format is currently available from the '
                         'command line interface.')

    tape9s = normalize_tape9s(tape9s)
    mats, nucs = tape9_to_sparse(tape9s, phi, format=format,
                                decaylib=decaylib,
                                include_fission=include_fission,
                                alpha_as_He4=alpha_as_He4)
    for tape9, mat in zip(tape9s, mats):
        base = os.path.basename(tape9)
        base, _ = os.path.splitext(base)
        fission_part = '' if include_fission else '_nofission'
        alpha_part = '' if not alpha_as_He4 else '_alpha_as_He4'
        output = os.path.join(output_dir, base + '_' + str(phi) + fission_part + alpha_part + '.npz')

        print("Writing file to", output)
        save_sparse_csr(output, mat, nucs, phi)

def main(args=None):
    p = make_parser()
    ns = p.parse_args(args=args)
    try:
        save_sparse(**vars(ns))
    except Exception:
        import sys
        import pdb
        import traceback
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

if __name__ == '__main__':
    main()
