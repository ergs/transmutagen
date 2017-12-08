#!/usr/bin/env python
from transmutagen import get_CRAM_from_cache, t, plot_in_terminal
from transmutagen.analysis import setup_matplotlib_rc

from sympy import Poly, fraction, exp
import matplotlib.pyplot as plt

degree = 14
prec = 200
points = 10000

def main():
    setup_matplotlib_rc()

    expr = get_CRAM_from_cache(degree, prec)

    c = 0.6*degree

    # Get the translated approximation on [-1, 1]. This is similar logic from CRAM_exp().
    n, d = map(Poly, fraction(expr))
    inv = -c*(t + 1)/(t - 1)
    p, q = map(lambda i: Poly(i, t), fraction(inv))
    n, d = n.transform(p, q), d.transform(p, q)
    rat_func = n/d.TC()/(d/d.TC())
    rat_func = rat_func.evalf(prec)

    plt.clf()
    fig, ax = plt.subplots()

    fig.set_size_inches(4, 4)

    plot_in_terminal(expr - exp(-t), (0, 100), prec=prec, points=points,
        axes=ax)

    # Hide grid lines
    ax.grid(False)

    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])

    plt.axis('off')
    plt.tight_layout()
    plt.savefig('logo.png', transparent=True, bbox_inches='tight', pad_inches=0)

if __name__ == '__main__':
    main()
