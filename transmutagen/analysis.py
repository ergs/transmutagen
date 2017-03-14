from collections import defaultdict
import io

import tables

import matplotlib.pyplot as plt

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

            plt.scatter(x, y)

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
