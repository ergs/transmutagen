import argparse
import logging

import mpmath

from .transmutagen import CRAM_exp, logger

def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('degree', type=int)
    parser.add_argument('prec', type=int)
    parser.add_argument('--division', type=int)
    parser.add_argument('--c', type=float)
    parser.add_argument('--maxsteps', type=int)
    parser.add_argument('--max-loops', type=int)
    parser.add_argument('--tol', type=mpmath.mpf)
    parser.add_argument('--nsolve-type', default=None, choices=['points',
        'intervals'])
    parser.add_argument('--solver', default=None)
    parser.add_argument('--D-scale', default=None, type=float)
    parser.add_argument('--scale', default=None, type=bool)
    parser.add_argument('--log-level', default=None, choices=['debug', 'info',
        'warning', 'error', 'critical'])
    try:
        import argcomplete
        argcomplete.autocomplete(parser)
    except ImportError:
        pass
    args = parser.parse_args()

    arguments = args.__dict__.copy()
    for i in arguments.copy():
        if not arguments[i]:
           del arguments[i]
    if args.log_level:
        logger.setLevel(getattr(logging, args.log_level.upper()))
        del arguments['log_level']

    CRAM_exp(**arguments)

if __name__ == '__main__':
    main()
