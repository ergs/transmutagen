from collections import defaultdict
import io
import os
import argparse

import tables
import numpy as np
import matplotlib.pyplot as plt

from .tests.test_transmute import run_transmute_test
from .origen_all import TIME_STEPS

TIME_LABELS = [
    '1 second',
    '1 day',
    '1 month',
    '1 year',
    '10 years',
    '1,000\nyears',
    '1,000,000\nyears',
]

def analyze_origen(file):
    times = {'origen': defaultdict(list), 'cram': defaultdict(list)}
    with tables.open_file(file, mode='r') as h5file:
        for t in 'origen', 'cram':
            for lib in h5file.root:
                table = h5file.get_node(lib, t)
                for row in table:
                    times[t][row['time']].append(row['execution time ' + t.upper()])

            xvals = range(len(times[t]))

            x = []
            y = []
            for i in xvals:
                itimes = times[t][sorted(times[t])[i]]
                x += [i]*len(itimes)
                y += itimes

            plt.plot(x, y, 'o')

        plt.xticks(xvals, TIME_LABELS, wrap=True)

    # # Pad margins so that markers don't get clipped by the axes
    # plt.margins(0.2)
    # # Tweak spacing to prevent clipping of tick-labels
    # plt.subplots_adjust(bottom=0.15)
    plt.title('runtimes')
    plt.semilogy()

    plt_show_in_terminal()

def plt_show_in_terminal():
    try:
        from iterm2_tools.images import display_image_bytes
    except ImportError:
        plt.show()
    else:
        b = io.BytesIO()
        plt.savefig(b, format='png')
        print(display_image_bytes(b.getvalue()))

def analyze_nofission():
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
                plt.hist(np.sum(m, axis=1))
                plt.yscale('log', nonposy='clip')
                plt.title(lib + ' ' + r + ' ' + time_name)
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
