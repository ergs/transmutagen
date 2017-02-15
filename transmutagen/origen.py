from subprocess import run

from pyne.origen22 import (nlbs, write_tape5_irradiation, write_tape4,
    parse_tape9, merge_tape9, write_tape9, parse_tape6)
from pyne.material import from_atom_frac

def main():
    ORIGEN = '/home/origen22/code/o2_therm_linux.exe'

    xs_TAPE9 = "/home/origen22/libs/pwru50.lib"
    decay_TAPE9 = "/home/origen22/libs/decay.lib"

    parsed_xs_tape9 = parse_tape9(xs_TAPE9)
    parsed_decay_tape9 = parse_tape9(decay_TAPE9)

    merged_tape9 = merge_tape9([parsed_decay_tape9, parsed_xs_tape9])

    # Can set outfile to change directory, but the file name needs to be
    # TAPE9.INP.
    write_tape9(merged_tape9)

    decay_nlb, xsfpy_nlb = nlbs(parsed_xs_tape9)

    time = 2e6

    # Can set outfile, but the file name should be called TAPE5.INP.
    write_tape5_irradiation("IRF", time/(60*60*24), 4e14,
        xsfpy_nlb=xsfpy_nlb, cut_off=0, out_table_num=[4],
        out_table_nes=[True, False, False])

    M = from_atom_frac({"U235": 1}, mass=1, atoms_per_molecule=1)

    write_tape4(M)

    run(ORIGEN)

    data = parse_tape6()

    print(data)

if __name__ == '__main__':
    main()
