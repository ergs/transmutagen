# PYTHON_ARGCOMPLETE_OK

import argparse
import os
from subprocess import run
import time

import numpy as np
from scipy.sparse import csr_matrix
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

from .util import load_sparse_csr
from .tape9utils import origen_to_name
from .codegen import CRAM_matrix_exp_lambdify

ORIGEN = '/home/origen22/code/o2_therm_linux.exe'
decay_TAPE9 = "/home/origen22/libs/decay.lib"
LIBS_DIR = "/home/origen22/libs"
DATA_DIR = os.path.abspath(os.path.join(__file__, os.path.pardir,
    os.path.pardir, 'docker', 'data'))

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

def origen_data_to_array_atom_fraction(data, nucs):
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
        ('n_fission_fragments', np.float64),
        ('execution time ORIGEN', np.float64),
        ('execution time CRAM', np.float64),
        ('ORIGEN atom fraction', (1, nucs_size)),
        ('ORIGEN mass fraction', (1, nucs_size)),
        ('CRAM atom fraction', (1, nucs_size)),
        ('CRAM mass fraction', (1, nucs_size)),
        ])

    h5file = tables.open_file(file, mode="a", title="ORIGEN data", filters=tables.Filters(complevel=1))
    h5file.create_table('/', lib, transmutation_desc)

def time_func(f, *args, **kwargs):
    """
    Times f(*args, **kwargs)

    Returns (time_elapsed, f(*args, **kwargs)).

    """
    t = time.perf_counter()
    res = f(*args, **kwargs)
    return time.perf_counter() - t, res

def save_file(file, data, lib, nucs, start_nuclide, time, phi, n_fission_fragments=2.004):
    h5file = tables.open_file(file, mode="a", title="ORIGEN data", filters=tables.Filters(complevel=1))
    if lib not in h5file.root:
        create_hdf5_table(file, lib, len(nucs))
    table = h5file.get_node(h5file.root, lib)
    # table.row['initial vector'] = initial_vector(start_nuclide)
    table.row['time'] = time
    table.row['phi'] = phi
    table.row['n_fission_fragments'] = n_fission_fragments
    # table.row['execution time ORIGEN']
    # table.row['execution time CRAM']
    table.row['ORIGEN atom fraction'] = origen_data_to_array_weighted(data, nucs, n_fission_fragments=n_fission_fragments)
    table.row['ORIGEN mass fraction'] = origen_data_to_array_materials(data, nucs)



def test_origen_against_CRAM():
    e_complex = CRAM_matrix_exp_lambdify()

    # ORIGEN returns 3 digits
    rtol=1e-3
    # ORIGEN stops the taylor expansion with the error term
    # exp(ASUM)*ASUM**n/n! (using Sterling's approximation), where n =
    # 3.5*ASUM + 6 and ASUM is the max of the column sums. The max of the
    # column sums is ~2 because of fission, giving ~1e-5 (see ORIGEN lines
    # 5075-5100)
    atol=1e-5

    for datafile in os.listdir(DATA_DIR):
        origen_data = load_data(os.path.join(DATA_DIR, datafile))
        tape9, time, nuc, phi = os.path.splitext(datafile)[0].split()
        print("Analyzing", tape9, "at time=", time, "nuc=", nuc, "phi=", phi)
        print('-'*80)

        npzfilename = os.path.join('data', os.path.splitext(tape9)[0] + '_' + phi + '.npz')
        nucs, mat = load_sparse_csr(npzfilename)

        nuc_to_idx = {v: i for i, v in enumerate(nucs)}
        b = csr_matrix(([1], [[nuc_to_idx[nuc]], [0]]), shape=[mat.shape[1],
        1])

        CRAM_res = np.asarray(e_complex(-mat.T*float(time), b))
        CRAM_res_normalized = CRAM_res/np.sum(CRAM_res)

        ORIGEN_res_weighted = origen_data_to_array_weighted(origen_data, nucs,)
        ORIGEN_res_materials = origen_data_to_array_materials(origen_data, nucs)
        # ORIGEN_res_atom_fraction = origen_data_to_array_atom_fraction(origen_data, nucs)

        for C, O, units in [
            (CRAM_res, ORIGEN_res_weighted, 'atom fractions'),
            (CRAM_res_normalized, ORIGEN_res_materials, 'mass fractions'),
            # (CRAM_res_normalized, ORIGEN_res_atom_fraction, 'atom fraction'),
            ]:

            print("Units:", units)
            try:
                np.testing.assert_allclose(C, O, rtol=rtol, atol=atol)
            except AssertionError as e:
                print(e)
                print("Mismatching elements sorted by error (CRAM, ORIGEN, symmetric relative error)")
                A = np.isclose(C, O, rtol=rtol, atol=atol)
                rel_error = abs(C - O)/(C + O)
                for i, in np.argsort(rel_error, axis=0)[::-1]:
                    if A[i]:
                        continue
                    print(nucs[i], C[i], O[i], rel_error[i])
            else:
                print("Arrays match with rtol=", rtol, "atol=", atol)

            print()

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
