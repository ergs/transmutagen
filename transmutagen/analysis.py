from collections import defaultdict
import os
import argparse
import decimal
from ast import literal_eval

import tables
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from sympy import re, im, Float

from .tests.test_transmute import run_transmute_test
from .origen_all import TIME_STEPS
from .util import plt_show_in_terminal, load_sparse_csr, diff_strs
from .cram import get_CRAM_from_cache, CRAM_coeffs
from .partialfrac import thetas_alphas

def analyze_origen(origen_results, *, file=None, title=True):
    plt.clf()
    fig, ax = plt.subplots()

    times = {
        'ORIGEN': defaultdict(list),
        'CRAM lambdify UMFPACK': defaultdict(list),
        'CRAM lambdify SuperLU': defaultdict(list),
        'CRAM py_solve': defaultdict(list),
    }
    label = {
        'ORIGEN': 'ORIGEN',
        'CRAM lambdify UMFPACK': 'CRAM SciPy solver (UMFPACK)',
        'CRAM lambdify SuperLU': 'CRAM SciPy solver (SuperLU)',
        'CRAM py_solve': 'CRAM C generated solver',
    }
    formats = {
        'ORIGEN': '+',
        'CRAM lambdify UMFPACK': 'x',
        'CRAM lambdify SuperLU': '<',
        'CRAM py_solve': '.',
    }
    offsets = {
        'ORIGEN': 0,
        'CRAM lambdify UMFPACK': -0.25,
        'CRAM lambdify SuperLU': 0.25,
        'CRAM py_solve': 0,
    }
    with tables.open_file(origen_results, mode='r') as h5file:
        for run in 'ORIGEN', 'CRAM lambdify UMFPACK', 'CRAM lambdify SuperLU', 'CRAM py_solve':
            for lib in h5file.root:
                table = h5file.get_node(lib, run.lower().replace(' ', '-'))
                for row in table:
                    exec_time = 'execution time CRAM lambdify' if run.startswith("CRAM lambdify") else 'execution time ' + run
                    times[run][row['time']].append(row[exec_time])

            xvals = sorted(TIME_STEPS)

            x = []
            y = []
            for i, t in enumerate(xvals):
                itimes = times[run][sorted(times[run])[i]]
                x += [10**offsets[run]*t]*len(itimes)
                y += itimes

            print("Longest", run, "runtime", max(y), "seconds")
            print("Shortest", run, "runtime", min(y), "seconds")

            ax.plot(x, y, formats[run], label=label[run])

    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.15)
    if title:
        plt.title("""Runtimes for different solvers computing transmutation over
several starting libraries, nuclides, and timesteps.""")

    ax.set_xscale('log')
    ax.set_xticks(sorted(TIME_STEPS))
    ax.xaxis.set_ticklabels([TIME_STEPS[i].replace(' ', '\n') for i in
        sorted(TIME_STEPS)], size='small')
    ax.set_yscale('log')
    ax.legend()
    plt.ylabel('Runtime (seconds)')
    plt.xlabel('Time step t')

    if file:
        plt.savefig(file)

    plt_show_in_terminal()

def analyze_nofission():
    plt.clf()
    for time, time_name in sorted(TIME_STEPS.items()):
        nofission_transmutes = {}
        for f in os.listdir('data'):
            if f.endswith('_nofission.npz'):
                lib = f.split('_', 1)[0]
                data = os.path.join('data', f)
                print("analyzing", data, 'on', time_name)
                nofission_transmutes[lib] = run_transmute_test(data, 14, 200,
                    time, run_all=False, _print=True)

        for lib in nofission_transmutes:
            for r in nofission_transmutes[lib]:
                m = nofission_transmutes[lib][r]
                if not isinstance(m, np.ndarray):
                    m = m.toarray()
                if m is None or np.isnan(m).any() or np.isinf(m).any():
                    print("Could not compute", r, "for", lib)
                    continue
                title = lib + ' ' + r + ' ' + time_name
                plot_matrix_sum_histogram(m, title)

def plot_matrix_sum_histogram(m, title='', axis=0):
    plt.clf()
    plt.hist(np.asarray(np.sum(m, axis=axis)).flatten())
    plt.yscale('log', nonposy='clip')
    plt.title(title)
    plt_show_in_terminal()
    plt.close()


def analyze_eigenvals(*, pwru50_data='data/pwru50_400000000000000.0.npz',
    file=None, title=True):
    from py_solve.py_solve import DECAY_MATRIX, csr_from_flat
    nucs, matpwru50 = load_sparse_csr(pwru50_data)
    matdecay = csr_from_flat(DECAY_MATRIX)
    for desc, mat in {'pwru50': matpwru50, 'decay': matdecay}.items():
        plt.clf()
        print("analyzing eigenvalues of", desc)
        eigvals, eigvects = scipy.sparse.linalg.eigen.eigs(mat, 3507)
        plt.scatter(np.real(eigvals), np.imag(eigvals))
        plt.yscale('symlog', linthreshy=1e-20)
        plt.xscale('symlog')
        plt.xlim([np.min(np.real(eigvals))*2, 1])
        plt.ylim([np.min(np.imag(eigvals))*10, np.max(np.imag(eigvals))*10])
        if title:
            plt.title("Eigenvalues of transmutation matrix for " + desc)
        plt_show_in_terminal()
        if file:
            path, ext = os.path.splitext(file)
            plt.savefig(path + '_' + desc + ext)

def analyze_cram_digits(max_degree=20):
    print("Computing coefficients (or getting from cache)")
    exprs = defaultdict(dict)
    cram_coeffs = defaultdict(dict)
    part_frac_coeffs = defaultdict(lambda: defaultdict(dict))
    # {degree: {prec: {'p': [coeffs], 'q': [coeffs]}}}
    correct_expr_digits = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # {degree: {prec: {'thetas': [[real coeffs, ..., im coeffs]], 'alphas':
    #     [[real coeffs, ..., im coeffs]], 'alpha0', [[real coeff, im coeff]]}}}
    correct_part_frac_digits = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for degree in range(1, max_degree+1):
        print("Degree", degree)
        for prec in range(100, 1100, 100):
            print("Precision", prec)
            exprs[degree][prec] = get_CRAM_from_cache(degree, prec, log=True, plot=True)
            cram_coeffs[degree][prec] = CRAM_coeffs(exprs[degree][prec], prec)
            thetas, alphas, alpha0 = thetas_alphas(exprs[degree][prec], prec)
            t = sorted(thetas, key=im)
            part_frac_coeffs[degree][prec]['thetas'] = [[re(i) for i in t], [im(i) for i in t]]
            t = sorted(alphas, key=im)
            part_frac_coeffs[degree][prec]['alphas'] = [[re(i) for i in t], [im(i) for i in t]]
            part_frac_coeffs[degree][prec]['alpha0'] = [[re(alpha0)], [im(alpha0)]]

        # Assume that 1000 has the most correct digits
        coeffs1000 = cram_coeffs[degree][1000]
        part_frac_coeffs1000 = part_frac_coeffs[degree][1000]
        for prec in range(100, 1000, 100):
            coeffs = cram_coeffs[degree][prec]
            for l in 'pq':
                for coeff, coeff1000 in zip(coeffs[l], coeffs1000[l]):
                    correct_expr_digits[degree][prec][l].append(len(os.path.commonprefix([coeff,
                        coeff1000])) - 1)

            these_part_frac_coeffs = part_frac_coeffs[degree][prec]
            for l in ['thetas', 'alphas', 'alpha0']:
                for i in range(2):
                    for coeff, coeff1000 in zip(these_part_frac_coeffs[l][i], part_frac_coeffs1000[l][i]):
                        format_str = '{:.%se}' % (prec - 1)
                        coeff = format_str.format(Float(coeff, prec))
                        coeff1000 = format_str.format(Float(coeff1000, prec))
                        correct_part_frac_digits[degree][prec][l].append(len(os.path.commonprefix([coeff,
                            coeff1000])) - 1)

    for typ, L, correct_digits in [('CRAM expression', 'pq', correct_expr_digits),
        ('Partial fraction', ['thetas', 'alphas', 'alpha0'], correct_part_frac_digits),]:

        print("Correct digits for", typ)

        # Plot minimum number of correct digits as a function of precision
        plt.clf()
        fig, ax = plt.subplots()

        minvals = defaultdict(list)
        for degree in range(1, max_degree+1):
            print("Degree", degree)
            for prec in range(100, 1000, 100):
                print("  Precision", prec)
                for l in L:
                    print('    ', end='')
                    print(l, end=' ')
                    for i in correct_digits[degree][prec][l]:
                        print(i, end=' ')
                    print()
                minvals[degree].append(min(sum([correct_digits[degree][prec][i] for i in L], [])))
            ax.plot(range(100, 1000, 100), minvals[degree], label=degree)

        # TODO: Make window wider so the legend isn't chopped off
        ax.legend(title=typ + " coefficients by degree", loc="upper left", bbox_to_anchor=(1,1))
        plt.ylabel('Number of correct digits')
        plt.xlabel('Precision')

        plt_show_in_terminal()

        # Plot minimum number of correct digits as a function of degree
        plt.clf()
        fig, ax = plt.subplots()

        minvals = defaultdict(list)
        for prec in range(100, 1000, 100):
            for degree in range(1, max_degree+1):
                minvals[prec].append(min(sum([correct_digits[degree][prec][i] for i in L], [])))
            ax.plot(range(1, max_degree+1), minvals[prec], label=prec)

        # TODO: Make window wider so the legend isn't chopped off
        ax.legend(title=typ + " coefficients by precision", loc="upper left", bbox_to_anchor=(1,1))
        plt.ylabel('Number of correct digits')
        plt.xlabel('Degree')
        ax.set_xticks(range(1, max_degree+1))

        plt_show_in_terminal()

def analyze_pusa_coeffs(*, file=None, title=True):
    from .tests.pusa_coeffs import part_frac_coeffs, plot_difference

    try:
        import colorama
    except ImportError:
        raise ImportError("colorama is required to use diff_strs")

    print("Differing coefficients:")
    for degree in [14, 16]:
        print("Degree:", degree)
        for typ in ['thetas', 'alphas', 'alpha0']:
            for idx in range(degree//2) if typ != 'alpha0' else range(1):
                print(typ, '-', idx, sep='', end=': ')
                for real_imag in ['real', 'imag']:
                    expr = get_CRAM_from_cache(degree, 200)
                    thetas, alphas, alpha0 = thetas_alphas(expr, 200)
                    format_str = '{:+.19e}'
                    paper_coeffs = part_frac_coeffs[degree]
                    # Thetas and alphas in the paper are negative what we have, and are only
                    # counted once per conjugate.
                    if typ == 'thetas':
                        vals = [-i for i in thetas if im(-i) >= 0]
                    elif typ == 'alphas':
                        vals = [-j for i, j in zip(thetas, alphas) if im(-i) >= 0]
                    elif typ == 'alpha0':
                        vals = [alpha0]
                    val = sorted(vals, key=im)[idx]
                    real_val_paper, imag_val_paper = sorted(zip(paper_coeffs[typ]['real'],
                        paper_coeffs[typ]['imaginary']), key=lambda i: float(i[1]))[idx]

                    real_val, imag_val = val.as_real_imag()
                    if real_imag == 'real':
                        our_str, pusa_str = format_str.format(decimal.Decimal(repr(real_val))), real_val_paper
                    else:
                        our_str, pusa_str = format_str.format(decimal.Decimal(repr(imag_val))), imag_val_paper
                    diff_strs(pusa_str, our_str, end=' ')
                    if not literal_eval(pusa_str) == literal_eval(our_str):
                        print(colorama.Back.RED, colorama.Fore.WHITE,
                            "<- Machine floats differ",
                            colorama.Style.RESET_ALL, sep='', end=' ')
                print()

        if file:
            path, ext = os.path.splitext(file)
            save_file = path + '-' + str(degree) + ext

        plot_difference(degree, file=save_file, all_plots=False)

def analyze():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--file', help="""File name to save the plot(s) to.
        For --eigenvals, a filename like "eigenvals.pdf" will be saved as
        "eigenvals_pwru50.pdf" and "eigenvals_decay.pdf". If not provided the
        plot is not saved. For --pusa-coeffs, a filename like
        "pusa-difference.pdf" will be saved as "pusa-difference-14.pdf" and
        "pusa-difference-16.pdf".""")
    parser.add_argument('--no-title', action='store_false', dest='title',
        help="""Don't add a title to plots""")

    origen = parser.add_argument_group('origen')
    origen.add_argument('--origen', action='store_true', dest='origen',
        help="""Run the origen analysis.""")
    origen.add_argument('--origen-results', default='data/results.hdf5',
        help="""HDF5 file for the results of the ORIGEN runs.""")
    nofission = parser.add_argument_group('nofission')
    nofission.add_argument('--nofission', action='store_true',
        dest='nofission', help="""Run the nofission analysis.""")
    eigenvals = parser.add_argument_group('eigenvals')
    eigenvals.add_argument('--eigenvals', action='store_true',
        dest='eigenvals', help="""Run the eigenvalue analysis.""")
    eigenvals.add_argument('--pwru50-data', help="""Path to pwru50 data file
        to analyze.""", default='data/pwru50_400000000000000.0.npz')
    cram_digits = parser.add_argument_group('cram-digits')
    cram_digits.add_argument('--cram-digits', action='store_true', help="""Analyze
        accuracy of CRAM digits. WARNING: If cache values have not been
        precomputed, this will take a long time (> 1 day) to compute.""")
    cram_digits.add_argument('--max-degree', type=int, help="""Max degree for
        --cram-digits. Default is 20.""", default=20)
    pusa_coeffs = parser.add_argument_group('Pusa coefficients')
    pusa_coeffs.add_argument('--pusa-coeffs', action='store_true',
        help="""Analyze the coefficients from the Maria Pusa paper "Correction to
        Partial Fraction Decomposition Coefficients for Chebyshev Rational
        Approximation on the Negative Real Axis".""")

    try:
        import argcomplete
        argcomplete.autocomplete(parser)
    except ImportError:
        pass
    args = parser.parse_args()

    if args.origen:
        analyze_origen(args.origen_results, file=args.file,
            title=args.title)
    if args.nofission:
        analyze_nofission()
    if args.eigenvals:
        analyze_eigenvals(pwru50_data=args.pwru50_data,
            file=args.file, title=args.title)
    if args.cram_digits:
        analyze_cram_digits(args.max_degree)
    if args.pusa_coeffs:
        analyze_pusa_coeffs(file=args.file, title=args.title)

if __name__ == '__main__':
    analyze()
