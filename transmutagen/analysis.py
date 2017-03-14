from collections import defaultdict
import io

import tables

import matplotlib.pyplot as plt

TIME_LABELS = [
    '1 second',
    '1 day',
    '1 month',
    '1 year',
    '10 years',
    '1,000\nyears',
    '1,000,000\nyears',
]

def analyze(file):
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

    try:
        from iterm2_tools.images import display_image_bytes
    except ImportError:
        plt.show()
    else:
        b = io.BytesIO()
        plt.savefig(b, format='png')
        print(display_image_bytes(b.getvalue()))


if __name__ == '__main__':
    import sys
    analyze(sys.argv[-1])
