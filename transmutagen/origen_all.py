"""
Collect data for a matrix of ORIGEN and CRAM runs
"""
import argparse
import os

from .tape9sparse import save_sparse
from .origen import execute, ORIGEN, decay_TAPE9, LIBS_DIR

ALL_LIBS = ['amo0ttta.lib', 'amo0tttc.lib', 'amo0tttr.lib', 'amo1ttta.lib',
    'amo1tttc.lib', 'amo1tttr.lib', 'amo2ttta.lib', 'amo2tttc.lib',
    'amo2tttr.lib', 'amopttta.lib', 'amoptttc.lib', 'amoptttr.lib',
    'amopuuta.lib', 'amopuutc.lib', 'amopuutr.lib', 'amopuuua.lib',
    'amopuuuc.lib', 'amopuuur.lib', 'amoruuua.lib', 'amoruuuc.lib',
    'amoruuur.lib', 'bwrpupu.lib', 'bwrpuu.lib', 'bwru.lib', 'bwrue.lib',
    'bwrus.lib', 'bwrus0.lib', 'candunau.lib', 'canduseu.lib', 'crbra.lib',
    'crbrc.lib', 'crbri.lib', 'crbrr.lib', 'emopuuua.lib', 'emopuuuc.lib',
    'emopuuur.lib', 'fftfc.lib', 'pwrd5d33.lib', 'pwrd5d35.lib',
    'pwrdu3th.lib', 'pwrputh.lib', 'pwrpuu.lib', 'pwru.lib', 'pwru50.lib',
    'pwrue.lib', 'pwrus.lib']

INITIAL_NUCS = [
    'Th232',
    'U233',
    'U235',
    'U238',
    'Pu239',
    'Pu241',
    'Cm245',
    'Cf249',
    ]

DAY = 60*60*24
YEAR = 365.25*DAY
MONTH = YEAR/12
TIME_STEPS = [1, 1*DAY, 1*MONTH, 1*YEAR, 10*YEAR, 1000*YEAR, 1e6*YEAR]

PHI = 4e14

def main():
    p = argparse.ArgumentParser('origen-all',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--recompute-matrices', action='store_true',
        help="Recompute input matrices (npz files). Done automatically if they are not found.")
    p.add_argument('--decay-tape9', help="path to the decay TAPE9 file.",
        default=decay_TAPE9)
    p.add_argument('--origen', help="Path to the origen executable",
        default=ORIGEN)
    p.add_argument('--libs-dir', default=LIBS_DIR, help="Path to the libs/ directory")
    args = p.parse_args()

    for tape9 in ALL_LIBS:
        print("Computing library", tape9)
        xs_tape9 = os.path.join(args.libs_dir, tape9)

        base = os.path.basename(xs_tape9)
        base, _ = os.path.splitext(base)
        os.makedirs('data', exist_ok=True)
        npzfilename = os.path.join('data', base + '_' + str(PHI) + '.npz')

        if args.recompute_matrices or not os.path.exists(npzfilename):
            print("Saving matrix for", xs_tape9, "to", npzfilename)
            save_sparse(xs_tape9, phi=PHI, output=npzfilename,
                decaylib=args.decay_tape9)

        for initial_nuclide in INITIAL_NUCS:
            print("Using initial nuclide", initial_nuclide)
            for time in TIME_STEPS:
                print("Using time", time)
                execute(xs_tape9, time, PHI, initial_nuclide,
                    decay_tape9=args.decay_tape9, origen=args.origen)

if __name__ == '__main__':
    main()
