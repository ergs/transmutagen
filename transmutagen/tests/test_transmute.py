import sys
import os
import argparse
import logging
from functools import wraps

from sympy import sympify, lambdify, horner, fraction

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm

from ..transmutagen import CRAM_exp, logger
from ..partialfrac import (thetas_alphas, thetas_alphas_to_expr,
    thetas_alphas_to_expr_complex, t)
from ..codegen import MatrixNumPyPrinter, scipy_translations

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                        shape=loader['shape'])

def lambdify_expr(expr):
    return nan_on_RuntimeError(lambdify(t, expr, scipy_translations, printer=MatrixNumPyPrinter))

def nan_on_RuntimeError(f):
    @wraps(f)
    def func(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except RuntimeError:
            return np.nan

    return func

def run_transmute_test(data, degree, prec, expr, time, plot=True):
    matrix = load_sparse_csr(data)

    expr = expr or CRAM_exp(degree, prec, plot=plot)
    num, den = fraction(expr)

    thetas, alphas, alpha0 = thetas_alphas(expr, prec)
    part_frac = thetas_alphas_to_expr(thetas, alphas, alpha0)
    part_frac_complex = thetas_alphas_to_expr_complex(thetas, alphas, alpha0)

    e_rat_func = lambdify_expr(expr)
    e_rat_func_horner = lambdify_expr(horner(num)/horner(den))
    e_part_frac = lambdify_expr(part_frac)
    e_part_frac_complex = lambdify_expr(part_frac_complex)

    res = {}

    res['rat_func'] = e_rat_func(-matrix*time)
    res['rat_func_horner'] = e_rat_func_horner(-matrix*time)
    res['part_frac'] = e_part_frac(-matrix*time)
    res['part_frac_complex'] = e_part_frac_complex(-matrix*time)
    res['expm'] = expm(matrix*time)

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
    parser.add_argument('--expr', type=sympify, help="""Precomputed CRAM
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

    res = run_transmute_test(args.data, args.degree, args.prec, args.expr, args.time)

    print("Column sums (min, max)")
    for r in sorted(res):
        col_sum = np.sum(res[r], axis=1)
        print(r, np.min(col_sum), np.max(col_sum))

if __name__ == '__main__':
    sys.exit(main())
