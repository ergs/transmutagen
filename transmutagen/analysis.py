from collections import defaultdict
import io
import os
import argparse

import tables
import numpy as np
import matplotlib.pyplot as plt

from .tests.test_transmute import run_transmute_test
from .origen_all import TIME_STEPS

def plt_show_in_terminal():
    try:
        from iterm2_tools.images import display_image_bytes
    except ImportError:
        plt.show()
    else:
        b = io.BytesIO()
        plt.savefig(b, format='png')
        print(display_image_bytes(b.getvalue()))

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
                nofission_transmutes[lib] = run_transmute_test(data, 14, 30,
                    time, run_all=True, _print=True)

        for lib in nofission_transmutes:
            for r in nofission_transmutes[lib]:
                m = nofission_transmutes[lib][r]
                if m is None or np.isnan(m.toarray()).any():
                    print("Could not compute", r, "for", lib)
                    continue
                title = lib + ' ' + r + ' ' + time_name
                plot_matrix_sum_histogram(m, title)

def plot_matrix_sum_histogram(m, title='', axis=1):
    plt.clf()
    plt.hist(np.sum(m, axis=axis))
    plt.yscale('log', nonposy='clip')
    plt.title(title)
    plt_show_in_terminal()
    plt.close()

def analyze():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--origen-results', default='data/results.hdf5',
        help="""HDF5 file for the results of the ORIGEN runs.""")
    parser.add_argument('--no-origen', action='store_false', dest='origen',
        help="""Don't run the origen analysis.""")
    parser.add_argument('--no-nofission', action='store_false',
        dest='nofission', help="""Don't run the nofission analysis.""")
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

if __name__ == '__main__':
    analyze()
