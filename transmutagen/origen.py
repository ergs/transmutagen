# PYTHON_ARGCOMPLETE_OK

import argparse
import os
from subprocess import run

import numpy as np
import tables

from pyne.utils import toggle_warnings
import warnings
toggle_warnings()
warnings.simplefilter('ignore')

import pyne.data
import pyne.material
from pyne.origen22 import (nlbs, write_tape5_irradiation, write_tape4,
    parse_tape9, merge_tape9, write_tape9, parse_tape6)
from pyne.material import from_atom_frac

from .tape9utils import origen_to_name

ORIGEN = '/home/origen22/code/o2_therm_linux.exe'
decay_TAPE9 = "/home/origen22/libs/decay.lib"
LIBS_DIR = "/home/origen22/libs"

NUCLIDE_KEYS = ['activation_products', 'actinides', 'fission_products']

def load_data(datafile):
    import pyne.data
    # Make pyne use naive atomic mass numbers to match ORIGEN
    for i in pyne.data.atomic_mass_map:
        pyne.data.atomic_mass_map[i] = float(pyne.nucname.anum(i))

    with open(datafile) as f:
        return eval(f.read(), {'array': np.array, 'pyne': pyne})

def origen_to_array(origen_dict, nucs):
    new_data = np.zeros((len(nucs), 1))
    nuc_to_idx = {v: i for i, v in enumerate(nucs)}
    for i in origen_dict:
        new_data[nuc_to_idx[origen_to_name(i)]] += origen_dict[i][1]

    return new_data

def origen_data_to_array_weighted(data, nucs, n_fission_fragments=2.004):
    # Table 5 is grams
    table_5_weights = {}
    table_5_nuclide = data['table_5']['nuclide']
    for key in NUCLIDE_KEYS:
        table_5_weights[key] = np.sum(origen_to_array(table_5_nuclide[key], nucs), axis=0)
    table_5_weights['fission_products'] *= n_fission_fragments

    # Table 4 is atom fraction
    table_4_nuclide = data['table_4']['nuclide']
    new_data = np.zeros((len(nucs), 1))
    for key in NUCLIDE_KEYS:
        new_data += table_5_weights[key]*origen_to_array(table_4_nuclide[key], nucs)

    return new_data

def origen_data_to_array_atom_fraction(data, nucs, n_fission_fragments=2):
    # Table 4 is atom fraction
    table_4_nuclide = data['table_4']['nuclide']
    new_data = np.zeros((len(nucs), 1))
    for key in NUCLIDE_KEYS:
        new_data += origen_to_array(table_4_nuclide[key], nucs)

    return new_data

def origen_data_to_array_materials(data, nucs):
    material = data['materials'][1]
    new_data = np.zeros((len(nucs), 1))
    nuc_to_idx = {v: i for i, v in enumerate(nucs)}

    for nuc, atom_frac in material.to_atom_frac().items():
        new_data[nuc_to_idx[pyne.nucname.name(nuc)]] = atom_frac

    return new_data

def create_hdf5_table(file, lib, nucs_size):
    transmutation_desc = np.dtype([
        ('initial vector', np.float64, (1, nucs_size)),
        ('time', np.float64),
        ('phi', np.float64),
        ('execution time ORIGEN', np.float64),
        ('execution time CRAM', np.float64),
        ('ORIGEN atom fraction', (1, nucs_size)),
        ('ORIGEN mass fraction', (1, nucs_size)),
        ('CRAM atom fraction', (1, nucs_size)),
        ('CRAM mass fraction', (1, nucs_size)),
        ])

    h5file = tables.open_file(file, mode="a", title="ORIGEN data", filters=tables.Filters(complevel=1))
    h5file.create_table('/', lib, transmutation_desc)

def save_file(file, data, lib, nucs_size, start_nuclide, time, phi):
    h5file = tables.open_file(file, mode="a", title="ORIGEN data", filters=tables.Filters(complevel=1))
    if lib not in h5file.root:
        create_hdf5_table(file, lib, nucs_size)


def make_parser():
    p = argparse.ArgumentParser('origen', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('xs_tape9', metavar='xs-tape9', help="""path to the cross section TAPE9 file. If
    the path is not absolute, defaults to looking in {LIBS_DIR}""".format(LIBS_DIR=LIBS_DIR))
    p.add_argument('time', help='the time in sec',
                   type=float)
    p.add_argument('--phi', help='the neutron flux in [n/cm^2/sec]',
                   type=float, default=4e14)
    p.add_argument('--nuclide', help="The initial starting nuclide.",
        default="U235")
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
    if not os.path.isabs(xs_tape9):
        xs_tape9 = os.path.join(LIBS_DIR, xs_tape9)
    time = args.time
    phi = args.phi
    nuclide = args.nuclide
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
        xsfpy_nlb=xsfpy_nlb, cut_off=0, out_table_num=[4, 5],
        out_table_nes=[True, False, False])

    M = from_atom_frac({nuclide: 1}, mass=1, atoms_per_molecule=1)

    write_tape4(M)

    run(origen)

    # Make pyne use naive atomic mass numbers to match ORIGEN
    for i in pyne.data.atomic_mass_map:
        pyne.data.atomic_mass_map[i] = float(pyne.nucname.anum(i))

    data = parse_tape6()

    filename = "{library} {time} {nuclide} {phi}.py".format(
        library=os.path.basename(xs_tape9),
        time=time,
        nuclide=nuclide,
        phi=phi,
        )
    with open('data/' + filename, 'w') as f:
        f.write(repr(data))
        print("Writing data to data/" + filename)

if __name__ == '__main__':
    main()
