from collections import defaultdict
import os
import argparse

import tables
import numpy as np
import matplotlib.pyplot as plt

from .tests.test_transmute import run_transmute_test
from .origen_all import TIME_STEPS
from .util import plt_show_in_terminal
from .cram import get_CRAM_from_cache, CRAM_coeffs

def analyze_origen(file):
    plt.clf()
    fig, ax = plt.subplots()

    times = {'ORIGEN': defaultdict(list), 'CRAM': defaultdict(list)}
    with tables.open_file(file, mode='r') as h5file:
        for run in 'ORIGEN', 'CRAM':
            for lib in h5file.root:
                table = h5file.get_node(lib, run.lower())
                for row in table:
                    times[run][row['time']].append(row['execution time ' + run])

            xvals = sorted(TIME_STEPS)

            x = []
            y = []
            for i, t in enumerate(xvals):
                itimes = times[run][sorted(times[run])[i]]
                x += [t]*len(itimes)
                y += itimes

            print("Longest", run, "runtime", max(y), "seconds")
            print("Shortest", run, "runtime", min(y), "seconds")

            ax.plot(x, y, 'o', label=run)

    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.15)
    plt.title("""\
Runtimes for ORIGEN and CRAM computing transmutation
over several starting libraries, nuclides, and timesteps.""")

    ax.set_xscale('log')
    ax.set_xticks(sorted(TIME_STEPS))
    ax.xaxis.set_ticklabels([TIME_STEPS[i].replace(' ', '\n') for i in
        sorted(TIME_STEPS)], size='small')
    ax.set_yscale('log')
    ax.legend()
    plt.ylabel('Runtime (seconds)')
    plt.xlabel('Time step t')

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

def analyze_cram_digits():
    print("Computing coefficients (or getting from cache)")
    exprs = defaultdict(dict)
    # {degree: {prec: {'p': [coeffs], 'q': [coeffs]}}}
    correct_digits = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for degree in range(1, 21):
        print("Degree", degree)
        for prec in range(100, 1100, 100):
            print("Precision", prec)
            exprs[degree][prec] = CRAM_coeffs(get_CRAM_from_cache(degree,
                prec), prec)

        # Assume that 1000 has the most correct digits
        coeffs1000 = exprs[degree][1000]
        for prec in range(100, 1000, 100):
            coeffs = exprs[degree][prec]
            for l in 'pq':
                for coeff, coeff1000 in zip(coeffs[l], coeffs1000[l]):
                    correct_digits[degree][prec][l].append(len(os.path.commonprefix([coeff,
                        coeff1000])) - 1)


    # Plot minimum number of correct digits as a function of precision
    plt.clf()
    fig, ax = plt.subplots()

    minvals = defaultdict(list)
    for degree in range(1, 21):
        print("Degree", degree)
        for prec in range(100, 1000, 100):
            print("  Precision", prec)
            for l in 'pq':
                print('    ', end='')
                print(l, end=' ')
                for i in correct_digits[degree][prec][l]:
                    print(i, end=' ')
                print()
            minvals[degree].append(min(correct_digits[degree][prec]['p'] + correct_digits[degree][prec]['q']))
        ax.plot(range(100, 1000, 100), minvals[degree], label=degree)

    # TODO: Make window wider so the legend isn't chopped off
    ax.legend(title="Degree", loc="upper left", bbox_to_anchor=(1,1))
    plt.ylabel('Number of correct digits')
    plt.xlabel('Precision')

    plt_show_in_terminal()

    # Plot minimum number of correct digits as a function of degree
    plt.clf()
    fig, ax = plt.subplots()

    minvals = defaultdict(list)
    for prec in range(100, 1000, 100):
        for degree in range(1, 21):
            minvals[prec].append(min(correct_digits[degree][prec]['p'] + correct_digits[degree][prec]['q']))
        ax.plot(range(1, 21), minvals[prec], label=prec)

    # TODO: Make window wider so the legend isn't chopped off
    ax.legend(title="Precision", loc="upper left", bbox_to_anchor=(1,1))
    plt.ylabel('Number of correct digits')
    plt.xlabel('Degree')
    ax.set_xticks(range(1, 21))

    plt_show_in_terminal()


def analyze():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--origen-results', default='data/results.hdf5',
        help="""HDF5 file for the results of the ORIGEN runs.""")
    parser.add_argument('--no-origen', action='store_false', dest='origen',
        help="""Don't run the origen analysis.""")
    parser.add_argument('--no-nofission', action='store_false',
        dest='nofission', help="""Don't run the nofission analysis.""")
    parser.add_argument('--cram-digits', action='store_true', help="""Analyze
        accuracy of CRAM digits. WARNING: If cache values have not been
        precomputed, this will take a long time (> 1 day) to compute.""")
    try:
        import argcomplete
        argcomplete.autocomplete(parser)
    except ImportError:
        pass
    args = parser.parse_args()

    if args.origen:
        analyze_origen(args.origen_results)
    if args.nofission:
        analyze_nofission()
    if args.cram_digits:
        analyze_cram_digits()

if __name__ == '__main__':
    analyze()
