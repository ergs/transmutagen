"""
exp(-t) on [0, oo) best CRAM coefficients from the appendix of the paper
"Extended Numerical Computations on the '1/9' Conjecture in Rational
Approximation Theory", A. J. Carpenter, A. Ruttan, and R.S. Varga

The coefficients have been OCRed from https://finereaderonline.com which uses
Abbyy, and verified by hand.

p are the numerator coefficients and q are the denominator coefficients. They
are ordered from the 0th (constant) to the nth (t**n) term. The constant term
in the denominator is always normalized to 1.
"""

import os
import re
import pprint
import difflib
from ..partialfrac import t

def parse_crv_coeffs(file=os.path.join(os.path.dirname( __file__), 'data', 'crv_coeffs')):
    """
    Used to generate the below dictionaries from the OCRed file.
    """
    _coeffs = {}
    # Lines like
    # n = 12
    N = re.compile(r'n = (\d+)')
    # Lines like
    # 1 1.7271172505820169235 (+00) -1.1542504579210602494 (-01)
    COEFF = re.compile(r'(\d+) ([\d.-]+) \(([+-]\d\d)\) ([\d.-]+) \(([+-]\d\d)\)')

    with open(file) as f:
        coeffstxt = f.read()

    for line in coeffstxt.splitlines():
        m_n = N.match(line)
        m_coeff = COEFF.match(line)
        if m_n:
            n, = m_n.groups()
            n = int(n)
            if n in _coeffs:
                raise ValueError("n given twice: " + n)
            _coeffs[n] = {'p': [], 'q': []}
            continue
        elif m_coeff:
            i, qb, qe, pb, pe = m_coeff.groups()
            i = int(i)
            if not (len(_coeffs[n]['p']) == len(_coeffs[n]['q']) == i):
                raise ValueError('Unexpected i: ' + repr(i))
            qblen = 22 if '-' in qb else 21
            pblen = 22 if '-' in pb else 21
            if len(qb) != qblen:
                raise ValueError('Unexpected length for q: ' + repr(line))
            if len(pb) != pblen:
                raise ValueError('Unexpected length for p: ' + repr(line))

            p = pb + 'e' + pe
            q = qb + 'e' + qe
            _coeffs[n]['p'].append(p)
            _coeffs[n]['q'].append(q)
        else:
            raise ValueError("Line did not match: " + repr(line))

    return _coeffs

coeffs = {}

coeffs[17] = {'p': [
     "1.0000000000000000229e+00",
    "-2.7729414537863365694e-01",
     "3.5585465335003458111e-02",
    "-2.7955519228564120076e-03",
     "1.4986697693060245133e-04",
    "-5.7867107664997448463e-06",
     "1.6558969231502529775e-07",
    "-3.5600331752523462572e-09",
     "5.7694610687453000688e-11",
    "-7.0111135292795387525e-13",
     "6.3018857979045448319e-15",
    "-4.0934546068922329038e-17",
     "1.8540997178336935538e-19",
    "-5.5478662451290864179e-22",
     "1.0077964568000473113e-24",
    "-9.6311299616434657123e-28",
     "3.6381008264169349244e-31",
    "-2.2774706078188437603e-35",
    ],
    'q': [
    "1.0000000000000000000e+00",
    "7.2270585462136953087e-01",
    "2.5829131995630046565e-01",
    "6.0809507390072969545e-02",
    "1.0598023489381217368e-02",
    "1.4566258090943959608e-03",
    "1.6421910387421514909e-04",
    "1.5592669068294270482e-05",
    "1.2701896118955897646e-06",
    "8.9952033533504590178e-08",
    "5.5866285320057224635e-09",
    "3.0690410653109898959e-10",
    "1.4763544085912781412e-11",
    "6.5692146982166977577e-13",
    "2.1929711332561360156e-14",
    "9.9803140529347369232e-16",
    "1.0076443559489818370e-17",
    "9.9533336444995010637e-19",
    ]
}

# Incomplete (I only have page 400 so far)
coeffs[18] = {'p': [
     "9.9999999999999999754e-01",
    "-2.7789748837786202942e-01",
     "3.5908020274519729805e-02",
    "-2.8562580858878249220e-03",
     "1.5609017337127136364e-04",
    "-6.1940471597288549472e-06",
     "1.8396482187296904295e-07",
    "-4.1548085790428176917e-09",
     "7.1794537028167480115e-11",
    "-9.4773868843647724074e-13",
     "9.4758123475451790323e-15",
    "-7.0617684389507100553e-17",
     "3.8258499143063765432e-19",
    "-1.4520088830640599985e-21",

    ],
    'q': [
    "1.0000000000000000000e+00",
    "7.2210251162213760760e-01",
    "2.5801053189666607896e-01",
    "6.0769684666293474499e-02",
    "1.0604260828765657181e-02",
    "1.4607084457312575111e-03",
    "1.6523598870223731736e-04",
    "1.5764526871377682191e-05",
    "1.2926132501100778041e-06",
    "9.2334736432550722847e-08",
    "5.8031481469953154253e-09",
    "3.2288400362818480622e-10",
    "1.6023022340045240158e-11",
    "6.9925714572325997863e-13",

    ]
}


def create_expression(n):
    if n not in coeffs:
        raise ValueError("Don't have coefficients for {}".format(n))

    from sympy import Float, Add

    p = coeffs[n]['p']
    q = coeffs[n]['q']

    # Don't use Poly here, it loses precision
    num = Add(*[Float(p[i])*t**i for i in range(n+1)])
    den = Add(*[Float(q[i])*t**i for i in range(n+1)])

    return num/den

def _plot(args):
    n = args.degree
    from sympy import exp

    from .. import plot_in_terminal

    rat_func = create_expression(n)

    print(rat_func)

    plot_in_terminal(rat_func - exp(-t), (0, 100), prec=20)

def _parse_coeffs(args):
    file = args.file
    check_existing = args.check_existing

    _coeffs = parse_crv_coeffs(file=file)
    pprint.pprint(_coeffs, width=20)

    if check_existing:
        for n in sorted(coeffs):
            print('Checking against', n)
            if _coeffs[n] == coeffs[n]:
                print(n, "matches")
            else:
                print(n, "doesn't match")
                print('p diff:')
                print('\n'.join(difflib.ndiff(_coeffs[n]['p'], coeffs[n]['p'])))
                print('q diff')
                print('\n'.join(difflib.ndiff(_coeffs[n]['q'], coeffs[n]['q'])))

if __name__ == '__main__':
    # Run this with
    # PYTHONPATH=/path/to/development/sympy python -m transmutagen.tests.crv_coeffs

    import argparse
    parser = argparse.ArgumentParser(description=__doc__)

    subparser = parser.add_subparsers()

    plot = subparser.add_parser('plot', help='Plot a rational function')
    plot.add_argument('degree', type=int)
    plot.set_defaults(func=_plot)

    parse_coeffs = subparser.add_parser('parse-coeffs', help='Parse the coeffs in the data file')
    parse_coeffs.add_argument('--file', default=os.path.join(os.path.dirname(
        __file__), 'data', 'crv_coeffs'), help="Path to the crv_coeffs data file")
    parse_coeffs.add_argument('--check-existing', action='store_true',
        default=False, help="""Print a diff from the parsed coeffs and the
        existing ones in this file""")
    parse_coeffs.set_defaults(func=_parse_coeffs)

    # TODO: Add options for arguments to pass to various functions as needed.
    try:
        import argcomplete
        argcomplete.autocomplete(parser)
    except ImportError:
        pass
    args = parser.parse_args()
    args.func(args)
