import sys
import os
import argparse
import logging
from functools import wraps
import datetime

from sympy import sympify, lambdify, horner, fraction

import numpy as np
from scipy.sparse.linalg import expm

from ..cram import logger
from ..partialfrac import (thetas_alphas, thetas_alphas_to_expr,
    thetas_alphas_to_expr_complex, t)
from ..codegen import MatrixNumPyPrinter, scipy_translations_autoeye, get_CRAM_from_cache
from ..util import load_sparse_csr

def lambdify_expr(expr):
    return None_on_RuntimeError(lambdify(t, expr, scipy_translations_autoeye, printer=MatrixNumPyPrinter))

def None_on_RuntimeError(f):
    @wraps(f)
    def func(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except RuntimeError:
            return None

    return func

def time_and_run(f, *args, _print=False):
    start = datetime.datetime.now()
    res = f(*args)
    end = datetime.datetime.now()
    if _print:
        print("Total time", end - start)
    return res

def run_transmute_test(data, degree, prec, expr, time, plot=True,
    _print=False, run_all=True):
    """
    Run transmute test on the data

    If run_all=True, runs the test on all forms of the exponential. Otherwise,
    only use part_frac_complex (the fastest).
    """
    nucs, matrix = load_sparse_csr(data)

    expr = get_CRAM_from_cache(degree, prec, expr=expr, plot=plot)

    num, den = fraction(expr)

    thetas, alphas, alpha0 = thetas_alphas(expr, prec)
    part_frac = thetas_alphas_to_expr(thetas, alphas, alpha0)
    part_frac_complex = thetas_alphas_to_expr_complex(thetas, alphas, alpha0)
    # part_frac_complex2 = thetas_alphas_to_expr_complex2(thetas, alphas, alpha0)

    e = {}
    e['part_frac_complex'] = lambdify_expr(part_frac_complex)
    if run_all:
        e['rat_func'] = lambdify_expr(expr)
        e['rat_func_horner'] = lambdify_expr(horner(num)/horner(den))
        e['part_frac'] = lambdify_expr(part_frac)
        # e['part_frac_complex2'] = lambdify_expr(part_frac_complex2)
        e['expm'] = lambda m: expm(-m)

    res = {}
    for func in sorted(e):
        if _print:
            print(func)
        arg = -matrix*time
        res[func] = time_and_run(e[func], arg, _print=_print)

    return res

def test_transmute():
    data = os.path.join(os.path.dirname(__file__), 'data', 'transmute.npz')
    month = 2.6e6
    run_transmute_test(data, 6, 30, None, month, plot=False)

def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('degree', type=int)
    parser.add_argument('prec', type=int, default=50)
    parser.add_argument('data', help="""Data of matrix to compute exp of. Should
    be in scipy sparse csr format.""")
    parser.add_argument('time', type=float)
    parser.add_argument('--expr', type=lambda e: sympify(e, locals=globals()), help="""Precomputed CRAM
    expression. Should have the same prec as 'prec'. If not provided, will be
    computed from scratch.""")
    parser.add_argument('--log-level', default=None, choices=['debug', 'info',
        'warning', 'error', 'critical'])
    # TODO: Add options for arguments to pass to various functions as needed.
    try:
        import argcomplete
        argcomplete.autocomplete(parser)
    except ImportError:
        pass
    args = parser.parse_args()

    if args.log_level:
        logger.setLevel(getattr(logging, args.log_level.upper()))

    res = run_transmute_test(args.data, args.degree, args.prec, args.expr,
        args.time, _print=True)

    print("Column sums (min, max):")
    errors = {}
    colsums = {}
    for r in sorted(res):
        if res[r] is None:
            print('Could not compute', r)
            continue
        colsums[r] = np.sum(res[r], axis=1)
        errors[r] = np.max(colsums[r]) - np.min(colsums[r])
    for r in sorted(errors, key=lambda i:errors[i], reverse=True):
        print(r, np.min(colsums[r]), np.max(colsums[r]))

if __name__ == '__main__':
    sys.exit(main())
