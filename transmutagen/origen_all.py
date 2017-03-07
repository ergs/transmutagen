"""
Collect data for a matrix of ORIGEN and CRAM runs
"""
import argparse
import os
import logging
import datetime

from .tape9sparse import save_sparse
from .origen import execute, ORIGEN, decay_TAPE9, LIBS_DIR, logger as origen_logger

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
# Set to WARN for less output
logger.setLevel(logging.INFO)

ALL_LIBS = [
    # 'amo0ttta.lib',
    # 'amo0tttc.lib',
    'amo0tttr.lib',
    'amo1ttta.lib',
    'amo1tttc.lib',
    # 'amo1tttr.lib',
    'amo2ttta.lib',
    # 'amo2tttc.lib',
    'amo2tttr.lib',
    'amopttta.lib',
    'amoptttc.lib',
    'amoptttr.lib',
    # 'amopuuta.lib',
    'amopuutc.lib',
    # 'amopuutr.lib',
    'amopuuua.lib',
    'amopuuuc.lib',
    'amopuuur.lib',
    'amoruuua.lib',
    # 'amoruuuc.lib',
    'amoruuur.lib',
    # 'bwrpupu.lib',
    # 'bwrpuu.lib',
    'bwru.lib',
    # 'bwrue.lib',
    'bwrus.lib',
    # 'bwrus0.lib',
    # 'candunau.lib',
    'canduseu.lib',
    'crbra.lib',
    'crbrc.lib',
    # 'crbri.lib',
    'crbrr.lib',
    # 'emopuuua.lib',
    'emopuuuc.lib',
    # 'emopuuur.lib',
    'fftfc.lib',
    # 'pwrd5d33.lib',
    # 'pwrd5d35.lib',
    'pwrdu3th.lib',
    'pwrputh.lib',
    # 'pwrpuu.lib',
    'pwru.lib',
    'pwru50.lib',
    'pwrue.lib',
    'pwrus.lib'
]

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
TIME_STEPS = [
    1,
    1*DAY,
    1*MONTH,
    1*YEAR,
    10*YEAR,
    1000*YEAR,
    1e6*YEAR,
]

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
    p.add_argument('--log-file', default='logs/origen_all.log', help='Path to file where output is logged')
    p.add_argument('--no-run-origen', action='store_false', dest='run_origen',
        help="Don't run origen")
    p.add_argument('--no-run-cram', action='store_false', dest='run_cram',
        help="Don't run cram")

    args = p.parse_args()


    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    logger.addHandler(logging.FileHandler(args.log_file, delay=True))
    origen_logger.addHandler(logging.FileHandler(args.log_file, delay=True))

    starttime = datetime.datetime.now()
    logger.info("Start time: %s", starttime)

    try:
        for tape9 in ALL_LIBS:
            logger.info("Computing library %s", tape9)
            xs_tape9 = os.path.join(args.libs_dir, tape9)

            base = os.path.basename(xs_tape9)
            base, _ = os.path.splitext(base)
            os.makedirs('data', exist_ok=True)
            npzfilename = os.path.join('data', base + '_' + str(PHI) + '.npz')

            if args.recompute_matrices or not os.path.exists(npzfilename):
                logger.info("Saving matrix for %s to %s", xs_tape9, npzfilename)
                save_sparse(xs_tape9, phi=PHI, output=npzfilename,
                    decaylib=args.decay_tape9)

            for initial_nuclide in INITIAL_NUCS:
                logger.info("Using initial nuclide %s", initial_nuclide)
                for time in TIME_STEPS:
                    logger.info("Using time %s", time)
                    logger.info("Run: %s %s %s", tape9, initial_nuclide, time)
                    try:
                        execute(xs_tape9, time, PHI, initial_nuclide,
                            decay_tape9=args.decay_tape9, origen=args.origen,
                            run_cram=args.run_cram, run_origen=args.run_origen)
                    except AssertionError as e:
                        logger.error("AssertionError with lib %s: %s", tape9, e)
    except BaseException as e:
        logger.error("Exception raised", exc_info=True)
        raise
    finally:
        endtime = datetime.datetime.now()
        logger.info("End time: %s", endtime)
        logger.info("Total time: %s", endtime - starttime)


if __name__ == '__main__':
    main()