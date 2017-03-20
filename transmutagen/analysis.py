from collections import defaultdict
import io
import os
import argparse

import tables
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import Formatter

from .tests.test_transmute import run_transmute_test
from .origen_all import TIME_STEPS

class TimeStepFormatter(Formatter):
    def __call__(self, i, pos=None):
        # TODO: Wrap the text
        return TIME_STEPS.get(i, str(i))

def analyze_origen(file):
    fig, ax = plt.subplots()

    times = {'origen': defaultdict(list), 'cram': defaultdict(list)}
    with tables.open_file(file, mode='r') as h5file:
        for run in 'origen', 'cram':
            for lib in h5file.root:
                table = h5file.get_node(lib, run)
                for row in table:
                    times[run][row['time']].append(row['execution time ' + run.upper()])

            xvals = sorted(TIME_STEPS)

            x = []
            y = []
            for i, t in enumerate(xvals):
                itimes = times[run][sorted(times[run])[i]]
                x += [t]*len(itimes)
                y += itimes

            ax.plot(x, y, 'o')

    # # Pad margins so that markers don't get clipped by the axes
    # plt.margins(0.2)
    # # Tweak spacing to prevent clipping of tick-labels
    # plt.subplots_adjust(bottom=0.15)
    plt.title('runtimes')

    ax.set_xscale('log')
    ax.set_xticks(sorted(TIME_STEPS))
    ax.get_xaxis().set_major_formatter(TimeStepFormatter())
    ax.set_yscale('log')

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
