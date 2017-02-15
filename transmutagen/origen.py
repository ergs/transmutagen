import argparse
import os
from subprocess import run

from pyne.utils import toggle_warnings
import warnings
toggle_warnings()
warnings.simplefilter('ignore')

from pyne.origen22 import (nlbs, write_tape5_irradiation, write_tape4,
    parse_tape9, merge_tape9, write_tape9, parse_tape6)
from pyne.material import from_atom_frac

ORIGEN = '/home/origen22/code/o2_therm_linux.exe'
decay_TAPE9 = "/home/origen22/libs/decay.lib"

def make_parser():
    p = argparse.ArgumentParser('origen', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('xs-tape9', help="path to the cross section TAPE9 file.")
    p.add_argument('time', help='the time in sec',
                   type=float)
    p.add_argument('phi', help='the neutron flux in [n/cm^2/sec]',
                   type=float, default=4e14)
    p.add_argument('--decay-tape9', help="path to the decay TAPE9 file.",
        default=decay_TAPE9)
    p.add_argument('--origen', help="Path to the origen executable",
        default=ORIGEN)
    return p

def main():
    p = make_parser()
    try:
        import argcomplete
        argcomplete.autocomplete(p)
    except ImportError:
        pass
    args = p.parse_args()

    xs_tape9 = args.xs_tape9
    time = args.time
    phi = args.phi
    decay_tape9 = args.decay_tape9
    origen = args.origen

    parsed_xs_tape9 = parse_tape9(xs_tape9)
    parsed_decay_tape9 = parse_tape9(decay_tape9)

    merged_tape9 = merge_tape9([parsed_decay_tape9, parsed_xs_tape9])

    # Can set outfile to change directory, but the file name needs to be
    # TAPE9.INP.
    write_tape9(merged_tape9)

    decay_nlb, xsfpy_nlb = nlbs(parsed_xs_tape9)

    # Can set outfile, but the file name should be called TAPE5.INP.
    write_tape5_irradiation("IRF", time/(60*60*24), phi,
        xsfpy_nlb=xsfpy_nlb, cut_off=0, out_table_num=[4],
        out_table_nes=[True, False, False])

    M = from_atom_frac({"U235": 1}, mass=1, atoms_per_molecule=1)

    write_tape4(M)

    run(origen)

    data = parse_tape6()

    print(data)

    filename = "{library} {time} {phi}.py".format(
        library=os.path.basename(xs_tape9),
        time=time,
        phi=phi,
        )
    with open('/data/' + filename, 'w') as f:
        f.write(repr(data))

if __name__ == '__main__':
    main()
