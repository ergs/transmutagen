from collections import defaultdict
import os
import argparse
import decimal
from ast import literal_eval
import sys
import json
import subprocess
import copy
import re as re_module

import tables
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from sympy import re, im, Float, exp, diff

from .tests.test_transmute import run_transmute_test
from .origen_all import TIME_STEPS
from .util import (plt_show_in_terminal, load_sparse_csr, diff_strs,)
from .cram import get_CRAM_from_cache, CRAM_coeffs, nsolve_intervals
from .partialfrac import (thetas_alphas, thetas_alphas_to_expr_complex,
    customre)
from .gensolve import make_ijk

def setup_matplotlib_rc():
    from matplotlib import rcParams
    rcParams['pgf.texsystem'] = 'lualatex'
    rcParams["text.usetex"] = True
    rcParams["font.family"] = 'serif'
    rcParams["font.serif"] = "Computer Modern Roman"
    rcParams["font.sans-serif"] = "Computer Modern Sans serif"
    rcParams["font.monospace"] = "Computer Modern Typewriter"

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

def analyze_nofission(*, run_all=False, file=None, title=True, thetas=None,
    alphas=None, alpha0=None, nofission_data=os.path.join(os.path.dirname(__file__), 'tests',
        'data', 'pwru50_400000000000000.0_nofission.npz')):
    try:
        import scikits.umfpack
        del scikits
    except ImportError:
        import traceback
        traceback.print_exc()
        sys.exit("scikit-umfpack is required to run the nofission analysis")

    valid_time_names = TIME_STEPS.values() if run_all else ['1 day', '1 year', '1000 years', '1 million years']
    nofission_transmutes = {t: {} for t in valid_time_names}

    for time, time_name in sorted(TIME_STEPS.items()):
        if time_name not in valid_time_names:
            continue
        if run_all:
            for f in os.listdir('data'):
                if f.endswith('_nofission.npz'):
                    lib = f.split('_', 1)[0]
                    data = os.path.join('data', f)
                    print("analyzing", data, 'on', time_name)
                    nofission_transmutes[time_name][lib] = run_transmute_test(data,
                        14, 200, time, run_all=False, _print=True,
                        thetas=thetas, alphas=alphas, alpha0=alpha0)
        else:
            print("analyzing", nofission_data, 'on', time_name)
            nofission_transmutes[time_name]['pwru50'] = run_transmute_test(nofission_data,
                14, 200, time, run_all=run_all, _print=True, thetas=thetas,
                alphas=alphas, alpha0=alpha0)

    plot_nofission_transmutes(nofission_transmutes, run_all=run_all,
        file=file, title=title)

    return nofission_transmutes

def plot_nofission_transmutes(nofission_transmutes, *, run_all=False,
    file=None, title=True):

    setup_matplotlib_rc()

    valid_time_names = TIME_STEPS.values() if run_all else ['1 day', '1 year', '1000 years', '1 million years']
    for time, time_name in sorted(TIME_STEPS.items()):
        if time_name not in valid_time_names:
            continue
        plt.switch_backend('pgf')
        plt.clf()
        fig, axes = plt.subplots(1, 4, sharey=True)
        fig.set_size_inches(1.5*6.4, 1.5/4*4.8)
        for lib in nofission_transmutes[time_name]:

            for (r, title), ax in zip([
                ('scipy.sparse.linalg.expm', r'\texttt{scipy.\allowbreak{}sparse.\allowbreak{}linalg.\allowbreak{}expm}'),
                ('part_frac_complex UMFPACK', '\\texttt{sympy.\\allowbreak{}lambdify}\nwith UMFPACK'),
                ('part_frac_complex SuperLU', '\\texttt{sympy.\\allowbreak{}lambdify}\nwith SuperLU'),
                ('transmutagen generated C solver', 'transmutagen generated\nC solver'),
                ],
                axes):

                m = nofission_transmutes[time_name][lib][r]
                if not isinstance(m, np.ndarray):
                    m = m.toarray()

                if m is None or np.isnan(m).any() or np.isinf(m).any():
                    print("Could not compute", r, "for", lib)
                    ax.text(0.5, 0.5, "Calculation resulted\nin NaNs",
                        va='center', ha='center', transform=ax.transAxes, size=12)
                    ax.set_xticks([])
                else:
                    ax.hist(np.asarray(np.sum(m, axis=0) - 1).flatten())

                ax.xaxis.offsetText.set_size(14)

                # if title:
                #     fig.suptitle(time_name, y=1.08)
                ax.set_yscale('log', nonposy='clip')

                # # Matplotlib's default ticks aren't great. We include the
                # # endpoints, and 0 if it isn't too close to the endpoints
                # # (within 5%).
                # xmin, xmax = ax.get_xlim()
                # if xmin < 0 < xmax and abs(xmin) > 0.05*(xmax - xmin) < abs(xmax):
                #     locs = [xmin, 0, xmax]
                # else:
                #     locs = [xmin, xmax]
                # ax.set_xticks(locs)
                #
                # # Put "x 10^-19" on every x-axis tick
                # ax.set_xticklabels([pretty_float(i) for i in locs])

                ax.ticklabel_format(scilimits=(-3, 3), axis='x')

                ax.minorticks_off()
                locs = ax.get_yticks()
                ax.set_ylim([0.5, 4000])
                ax.set_yticklabels(["$%d$" % int(i) for i in locs])


                if title:
                    ax.set_title(title)

            # Only the last axis
            fig.text(0.07, 0.5, "Count", ha='center', va='center', rotation='vertical')
            fig.text(0.5, -0.3, r'$\sum_i \left (e^{-At}\right )_{i,j} - 1$', ha='center', va='center')

            print(time_name)
            # plt_show_in_terminal()
            if file:
                path, ext = os.path.splitext(file)
                filename = '-'.join([path, lib, time_name.replace(' ', '-')]) + ext
                print("Saving to", filename)
            else:
                filename = file

            if filename:
                plt.savefig(filename, bbox_inches='tight')
            plt.close()

def pretty_float(i):
    """
    This function is specifically for the xticks in the nofission graphs. It
    might not give appropiate representations for other contexts.

    """
    if i == 0:
        return r'$0^{\vphantom{0}}$'
    float_exponent = np.floor(np.log10(abs(i)))

    if -3 <= float_exponent <= 3:
        return r"$%s^{\vphantom{0}}$" % str(i)[:6]
    lead_digit, exponent = ("%.0e" % i).split('e')
    return r"$%s\times 10^{%s}$" % (lead_digit, exponent)

def plot_matrix_sum_histogram(m, *, title='', axis=0, file=None):
    plt.clf()
    plt.hist(np.asarray(np.sum(m, axis=axis)).flatten())
    plt.yscale('log', nonposy='clip')
    if title:
        plt.title(title)
    plt_show_in_terminal()
    if file:
        plt.savefig(file)
    plt.close()


def analyze_eigenvals(*, pwru50_data=None, file=None, title=True):
    if not pwru50_data:
        pwru50_data = os.path.join(os.path.dirname(__file__), 'tests', 'data', 'pwru50_400000000000000.0.npz')
    from py_solve.py_solve import DECAY_MATRIX, csr_from_flat
    nucs, matpwru50 = load_sparse_csr(pwru50_data)
    matdecay = csr_from_flat(DECAY_MATRIX)
    for desc, mat in {'pwru50': matpwru50, 'decay': matdecay}.items():
        plt.clf()
        print("analyzing eigenvalues of", desc)
        eigvals, eigvects = scipy.sparse.linalg.eigen.eigs(mat, mat.shape[0]-2)
        plt.scatter(np.real(eigvals), np.imag(eigvals))
        plt.yscale('symlog', linthreshy=1e-20)
        plt.xscale('symlog')
        plt.xlim([np.min(np.real(eigvals))*2, 1])
        plt.ylim([np.min(np.imag(eigvals))*10, np.max(np.imag(eigvals))*10])
        plt.xticks([0] +
            [-10**i for i in range(1, 1+int(np.ceil(np.log10(-plt.xlim()[0]))), 2)])
        plt.yticks([-10**i for i in range(-19, int(np.log10(-plt.ylim()[0])), 2)] +
            [0] +
            [10**i for i in range(-19, int(np.log10(plt.ylim()[1])), 2)])
        plt.minorticks_off()
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

def _latex_typ(typ, idx):
    typ_mapping = {'thetas': r'\theta', 'alphas': r'\alpha', 'alpha0':
        r'\alpha_0'}
    if typ != 'alpha0':
        return '$' + typ_mapping[typ] + '_' + str(idx+1) + '$'
    else:
        return '$' + typ_mapping[typ] + '$'

def analyze_pusa_coeffs(*, file=None, title=True, latex=False):
    from .tests.pusa_coeffs import (part_frac_coeffs, plot_difference,
        transmutagen_cram_error, paper_cram_error, get_paper_part_frac,
        get_paper_thetas_alphas)
    from .partialfrac import t

    try:
        import colorama
    except ImportError:
        raise ImportError("colorama is required to use diff_strs")

    print("Differing coefficients:")
    for degree in [14, 16]:
        if latex:
            path, ext = os.path.splitext(latex)
            filename = '-'.join([path, str(degree)]) + ext
            print("Saving LaTeX table to", filename)
            f = open(filename, 'w')
        print("Degree:", degree)
        for typ in ['thetas', 'alphas', 'alpha0']:
            for idx in range(degree//2) if typ != 'alpha0' else range(1):
                if latex:
                    f.write(r'\multirow{2}{*}{%s} & ' % _latex_typ(typ, idx))
                else:
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
                        if imag_val == 0:
                            # Decimal formats the string with e+19, which is
                            # undesirable. See
                            # https://bugs.python.org/issue31684.
                            our_str = "+0.0000000000000000000e+0"

                    machine_differences = (literal_eval(pusa_str) != literal_eval(our_str))

                    if latex:
                        latex_pusa_str = pusa_str
                        latex_our_str = our_str
                        # if machine_differences:
                        #     latex_pusa_str = r'{\it %s}' % latex_pusa_str
                        #     latex_our_str = r'{\it %s}' % latex_our_str
                        if real_imag == 'imag':
                            f.write(' & ')
                            latex_pusa_str = latex_pusa_str + 'j'
                            latex_our_str = latex_our_str + 'j'

                        # For whatever reason, \setulcolor only applies to the
                        # next \ul
                        if machine_differences:
                            f.write(r'\setulcolor{red}')
                        sep = ' & ' if not machine_differences else r' & \setulcolor{red}'

                        diff_strs(latex_pusa_str, latex_our_str, end=r'\\',
                            style='latex separated', sep=sep,
                            stop_chars='e', file=f)
                        if real_imag == 'imag' and typ != 'alpha0':
                            f.write(r'\cline{2-3}')
                    diff_strs(pusa_str, our_str, end=' ')
                    if machine_differences:
                        print(colorama.Back.RED, colorama.Fore.WHITE,
                            "<- Machine floats differ",
                            colorama.Style.RESET_ALL, sep='', end=' ')
                print()

        if latex:
            f.close()

    plt.ion()
    plot_difference(file=file, all_plots=False)

    part_fracs = {}
    paper_part_fracs = {}
    interval = (0, 100)
    prec = 200

    # {degree: {'pusa_alpha_errors': [...], 'transmutagen_alpha_errors': [...], 't0s': [...]}}
    alpha_errors = defaultdict(lambda: defaultdict(list))

    for degree in [14, 16]:
        expr = get_CRAM_from_cache(degree, prec)
        thetas, alphas, alpha0 = thetas_alphas(expr, prec)
        part_frac = thetas_alphas_to_expr_complex(thetas, alphas, alpha0)
        part_frac = part_frac.replace(customre, re)

        paper_part_frac = get_paper_part_frac(degree).replace(customre, re)
        paper_thetas, paper_alphas, paper_alpha0 = get_paper_thetas_alphas(degree)

        part_fracs[degree] = part_frac
        paper_part_fracs[degree] = paper_part_frac

        print("Computing critical points for degree", degree)
        critical_points = nsolve_intervals(diff(part_fracs[degree] - exp(-t), t),
            interval, prec=prec)

        print('-'*80)
        print("Testing", len(critical_points), "points in", interval, "for degree", degree)
        for t0 in critical_points:
            print()

            transmutagen_error = transmutagen_cram_error(degree, t0, prec)
            pusa_error = paper_cram_error(degree, t0, prec)

            expr = get_CRAM_from_cache(degree, prec)
            thetas, alphas, alpha0 = thetas_alphas(expr, prec)
            print('degree', degree, 'alpha0:\t\t%.20g' % alpha0)

            transmutagen_alpha_error = abs(abs(transmutagen_error) - alpha0)
            pusa_alpha_error = abs(abs(pusa_error) - paper_alpha0)

            for name, error, alpha_error in [
                ("Our", transmutagen_error, transmutagen_alpha_error),
                ("Pusa", pusa_error, pusa_alpha_error)]:

                print(name, "error near t=%.4f:\t%.20g" % (t0,
                    error))
                alpha_error = abs(abs(error) - alpha0)
                color = colorama.Fore.RED if alpha_error > 1e-20 else colorama.Fore.GREEN

                print("Off by:", color, '\t\t\t%.5g' % alpha_error, colorama.Style.RESET_ALL)
                alpha_error = abs(abs(error) - alpha0)

            alpha_errors[degree]['t0s'].append(t0)
            alpha_errors[degree]['transmutagen_alpha_errors'].append(transmutagen_alpha_error)
            alpha_errors[degree]['pusa_alpha_errors'].append(pusa_alpha_error)

            if transmutagen_alpha_error >= pusa_alpha_error:
                print(colorama.Fore.RED, "Pusa error is better",
                    colorama.Style.RESET_ALL, sep='')
            else:
                print(colorama.Fore.GREEN, "Our error is better",
                    colorama.Style.RESET_ALL, sep='')

    plt.clf()
    for degree in [14, 16]:
        plt.figure()
        plt.plot(alpha_errors[degree]['t0s'],
            alpha_errors[degree]['transmutagen_alpha_errors'], '.',
            label=r'transmutagen')
        plt.plot(alpha_errors[degree]['t0s'],
            alpha_errors[degree]['pusa_alpha_errors'], '.',
            label=r'Pusa~\cite{pusa2012correction}')
        plt.title(r'degree %d' % degree)
        plt.legend()
        plt.xlabel('$t$')
        plt.ylabel(r'$||e^{-t_0} - \hat{r}_{%d,%d}(t_0)| - \alpha_0|$' %
            (degree, degree))
        if file:
            filename, ext = os.path.splitext(file)
            filename += '-errors-' + str(degree)
            plt.savefig(filename + ext)

    # analyze_nofission(thetas=paper_thetas, alphas=paper_alphas, alpha0=paper_alpha0)

def analyze_gensolve(*, origen_json_file=None, json_file=None,
    pairs_per_pass=1, runs=100, warm_up_runs=5):

    from pyne import nucname

    origen_json_data = json.load(origen_json_file or open(os.path.join(os.path.dirname(__file__), 'data', 'gensolve_origen.json')))
    json_data = json.load(json_file or open(os.path.join(os.path.dirname(__file__), 'data', 'gensolve.json')))

    # Make the nuclide list the same
    new_nuclides = set(json_data['nucs']) - set(origen_json_data['nucs'])
    removed_nuclides = set(origen_json_data['nucs']) - set(json_data['nucs'])
    [origen_json_data['fromto'].append([i, i]) for i in new_nuclides]
    [json_data['fromto'].append([i, i]) for i in removed_nuclides]
    full_nucs = sorted(set(json_data['nucs']) | set(origen_json_data['nucs']), key=nucname.cinder)
    json_data['nucs'] = origen_json_data['nucs'] = full_nucs

    new_json = copy.deepcopy(origen_json_data)

    new_fromtos = sorted(set(map(tuple, json_data['fromto'])) - set(map(tuple,
        origen_json_data['fromto'])))

    all_runtimes = []
    added = [0]

    print("Compiling 0/%d" % len(new_fromtos))
    outfile = generate_gensolve_test(new_json, 0)
    print("Running 0/%d" % len(new_fromtos))
    runtimes = run_gensolve_test(outfile)
    print("Run", 0, "took", np.mean(runtimes), "seconds on average")
    all_runtimes.append(runtimes)

    for i, fromto in enumerate(new_fromtos, 1):
        new_json['fromto'].append(list(fromto))

        if (len(new_fromtos) - i) % pairs_per_pass != 0:
            continue

        added.append(i)
        print("Compiling %d/%d" % (i, len(new_fromtos)))
        outfile = generate_gensolve_test(new_json, i)
        print("Running %d/%d" % (i, len(new_fromtos)))
        runtimes = run_gensolve_test(outfile, warm_up_runs=warm_up_runs, runs=runs)
        print("Run", i, "took", np.mean(runtimes), "seconds on average")
        all_runtimes.append(runtimes)

    plt.clf()
    plt.plot(added, list(map(np.mean, all_runtimes)))
    plt.fill_between(added, list(map(np.min, all_runtimes)), list(map(np.max,
        all_runtimes)), alpha=0.5)
    plt_show_in_terminal()

def generate_gensolve_test(json_data, tag, directory='gensolve-tests', recompile=False):
    from transmutagen.gensolve import generate, GCC_COMPILER_FLAGS

    outscript = os.path.join(directory, 'test_%s.o' % tag)

    if not recompile and os.path.exists(outscript):
        print("Already compiled")
        return outscript

    os.makedirs(directory, exist_ok=True)

    sourcefile = os.path.join(directory, 'test_%s.c' % tag)
    generate(json_data=json_data, py_solve=False, degrees=[14],
    outfile=sourcefile, timing_test=True)

    subprocess.run(['gcc'] + GCC_COMPILER_FLAGS + ['-o', outscript] +
        [sourcefile], check=True)

    return outscript

def run_gensolve_test(outscript, warm_up_runs=5, runs=100):
    TIMING_TEST_OUT = re_module.compile(r'Took (.*) seconds\nSum of resulting vector: (.*)\n')

    runtimes = []
    for i in range(warm_up_runs):
        p = subprocess.run([outscript], check=True, stdout=subprocess.PIPE)

    for i in range(runs):
        p = subprocess.run([outscript], check=True, stdout=subprocess.PIPE)
        m = TIMING_TEST_OUT.match(p.stdout.decode('utf-8'))
        if not m:
            raise ValueError("Gensolve command output not in the expected format: %s" % p.stdout)

        runtime, vector_sum = map(literal_eval, m.groups())
        runtimes.append(runtime)
        if not vector_sum == 1:
            raise ValueError("Gensolve vector sum not 1 (%s)" % m.group(2))

    return runtimes

# Based on https://matplotlib.org/users/event_handling.html
class InteractiveLUMatrix:
    def __init__(self, N, extra=()):
        from matplotlib.patches import Patch

        self.extra = list(extra)
        self.N = N
        self._make_matrix_data()
        self.press = None
        self.adding = None
        self._make_matrix_data()

        self.image = plt.imshow(self.data)
        patches = [
            Patch(color=self.image.cmap(self.image.norm(0)),
                label="zero value"),
            Patch(color=self.image.cmap(self.image.norm(1)),
                label="zero value that must be included for LU"),
            Patch(color=self.image.cmap(self.image.norm(2)),
                label="nonzero value"),
            ]
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )

        self._connect()

    def _make_matrix_data(self):
        extra = self.extra
        N = self.N
        ij = {i: j for j, i in enumerate([(i, i) for i in range(N)] + extra)}
        a = np.zeros((N, N), dtype=int)
        for i, j in ij:
            a[i, j] = 1
        ijk = make_ijk(ij, N)
        b = np.zeros((N, N), dtype=int)
        for i, j in ijk:
            b[i, j] = 1
        data = a + b
        self.data = data

    def _update_image(self):
        self.image.set_data(self.data)
        self.image.figure.canvas.draw()

    def _connect(self):
        'connect to all the events we need'
        self.cidpress = self.image.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.image.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.image.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)

    def _invert(self, event):
        # Inverted
        x, y = int(event.ydata+0.5), int(event.xdata+0.5)

        if (x, y) in self.extra:
            # When dragging, only add or only remove points
            if self.adding == False:
                return
            self.extra.remove((x, y))
            self.adding = True
        else:
            if self.adding == True:
                return
            self.extra.append((x, y))
            self.adding = False

    def on_press(self, event):
        self._invert(event)

        self._make_matrix_data()
        self._update_image()
        self.press = event.xdata, event.ydata

    def on_motion(self, event):
        'on motion we will move the image if the mouse is over us'
        if self.press is None:
            return

        self._invert(event)
        self._make_matrix_data()
        self._update_image()

    def on_release(self, event):
        'on release we reset the press data'
        self.press = None
        self.adding = None
        self._update_image()

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.image.figure.canvas.mpl_disconnect(self.cidpress)
        self.image.figure.canvas.mpl_disconnect(self.cidrelease)
        self.image.figure.canvas.mpl_disconnect(self.cidmotion)

def analyze_lusolve(N):
    plt.clf()
    plt.interactive(True)
    I = InteractiveLUMatrix(N)
    plt.show(block=True)
    I.disconnect()

def analyze():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--file', help="""File name to save the plot(s) to.

        For --eigenvals, a filename like "eigenvals.pdf" will be saved as
        "eigenvals_pwru50.pdf" and "eigenvals_decay.pdf".

        For --nofission, a filename like "nofission.pdf" will be saved as
        "nofission-pwru50-1-day.pdf", "nofission-pwru50-1-year.pdf",
        "nofission-pwru50-1000-years.pdf", and
        "nofission-pwru50-1-million-years.pdf".

        If not provided the plot is not saved.""")
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
    nofission.add_argument('--run-all', action='store_true', help="""Run the
        nofission analysis on all the nofission data in the data/ directory
        against all time steps and with all solvers. The default is to run the
        analysis on the pwru50 data in the transmutagen/tests directory and
        only against 1 day, 1 year, 1000 years, and 1 million years,
        against the generated C solver, part_frac_complex, and
        scipy.sparse.linalg.expm.""")
    nofission.add_argument('--nofission-data',
        default=os.path.join(os.path.dirname(__file__), 'tests',
            'data', 'pwru50_400000000000000.0_nofission.npz'), help="""Data file to use
        for nofission analysis (ignored when --run-all is passed). The default
        is %(default)s.""")

    eigenvals = parser.add_argument_group('eigenvals')
    eigenvals.add_argument('--eigenvals', action='store_true',
        dest='eigenvals', help="""Run the eigenvalue analysis.""")
    eigenvals.add_argument('--pwru50-data', help="""Path to pwru50 data file
        to analyze. The default is the data that is in the
    transmutagen/tests/data directory.""", default=None)

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
    pusa_coeffs.add_argument('--latex', help="""Save LaTeX table to the given
    file. A filename like pusa-table.tex will result in pusa-table-14.tex and
    pusa-table-16.tex.""")

    gensolve = parser.add_argument_group("Gensolve")
    gensolve.add_argument('--gensolve', action='store_true', help="""Run
        gensolve timing analysis.""")
    gensolve.add_argument('--pairs-per-pass', help="""Number of from-to pairs
        to add on each pass. The default is %(default)s.""", default=1, type=int)
    gensolve.add_argument('--warm-up-runs', help="""Number of times to run the
        command first without recording the output. The default is
        %(default)s.""", default=5, type=int)
    gensolve.add_argument('--runs', help="""Number of times to run the
        command. The default is %(default)s.""", default=100, type=int)

    lusolve = parser.add_argument_group("LUSolve")
    lusolve.add_argument('--lusolve', action='store_true', help="""Run
        LU solve interactive analysis.""")
    lusolve.add_argument('--N', help="""Size of the matrix. The default is
        %(default)s""", default=100, type=int)

    try:
        import argcomplete
        argcomplete.autocomplete(parser)
    except ImportError:
        pass
    args = parser.parse_args()

    setup_matplotlib_rc()

    if args.origen:
        analyze_origen(args.origen_results, file=args.file,
            title=args.title)
    if args.nofission:
        analyze_nofission(run_all=args.run_all, file=args.file,
            title=args.title, nofission_data=args.nofission_data)
    if args.eigenvals:
        analyze_eigenvals(pwru50_data=args.pwru50_data,
            file=args.file, title=args.title)
    if args.cram_digits:
        analyze_cram_digits(args.max_degree)
    if args.pusa_coeffs:
        analyze_pusa_coeffs(file=args.file, title=args.title, latex=args.latex)
    if args.gensolve:
        analyze_gensolve(pairs_per_pass=args.pairs_per_pass, runs=args.runs,
            warm_up_runs=args.warm_up_runs)
    if args.lusolve:
        analyze_lusolve(args.N)

if __name__ == '__main__':
    analyze()
