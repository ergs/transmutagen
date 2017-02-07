import sys
import os
import argparse
import logging

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
    return lambdify(t, expr, scipy_translations, printer=MatrixNumPyPrinter)

def run_transmute_test(data, degree, prec, expr, time):
    matrix = load_sparse_csr(data)

    expr = expr or CRAM_exp(degree, prec)
    num, den = fraction(expr)

    thetas, alphas, alpha0 = thetas_alphas(expr, prec)
    part_frac = thetas_alphas_to_expr(thetas, alphas, alpha0)
    part_frac_complex = thetas_alphas_to_expr_complex(thetas, alphas, alpha0)

    e_rat_func = lambdify_expr(expr)
    e_rat_func_horner = lambdify_expr(horner(num)/horner(den))
    e_part_frac = lambdify_expr(part_frac)
    e_part_frac_complex = lambdify_expr(part_frac_complex)

    rat_func_res = e_rat_func(-matrix*time)
    rat_func_horner_res = e_rat_func_horner(-matrix*time)
    part_frac_res = e_part_frac(-matrix*time)
    part_frac_complex_res = e_part_frac_complex(-matrix*time)
    expm_res = expm(matrix*time)

def test_transmute():
    data = os.path.join(os.path.dirname(__file__), 'data', 'transmute.npz')
    month = 2.6e6
    run_transmute_test(data, 6, 30, None, month)

def main():
    parser = argparse.ArgumentError(description=__doc__)

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

    run_transmute_test(args.data, args.degree, args.prec, args.expr, args.time)

if __name__ == '__main__':
    sys.exit(main())
