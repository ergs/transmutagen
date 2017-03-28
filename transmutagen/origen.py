# PYTHON_ARGCOMPLETE_OK

import argparse
import os
from subprocess import run
import logging
from itertools import combinations

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

from .util import load_sparse_csr, time_func
from .tape9utils import origen_to_name
from .codegen import CRAM_matrix_exp_lambdify

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
# Change to WARN for less output
logger.setLevel(logging.INFO)

# ORIGEN = '/home/origen22/code/o2_therm_linux.exe'
ORIGEN = '/home/o2prec/o2prec'
decay_TAPE9 = "/home/origen22/libs/decay.lib"
LIBS_DIR = "/home/origen22/libs"
DATA_DIR = os.path.abspath(os.path.join(__file__, os.path.pardir,
    os.path.pardir, 'data'))

NUCLIDE_KEYS = ['activation_products', 'actinides', 'fission_products']

def execute_origen(xs_tape9, time, nuclide, phi, origen, decay_tape9):
    xs_tape9 = xs_tape9
    if not os.path.isabs(xs_tape9):
        xs_tape9 = os.path.join(LIBS_DIR, xs_tape9)

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

    origen_time, data = time_func(run, origen)

    # Make pyne use naive atomic mass numbers to match ORIGEN
    for i in pyne.data.atomic_mass_map:
        pyne.data.atomic_mass_map[i] = float(pyne.nucname.anum(i))

    data = parse_tape6()

    logger.info("ORIGEN time: %s", origen_time)
    return origen_time, data

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

def origen_data_to_array_weighted(ORIGEN_data, nucs, n_fission_fragments=2.004):
    # Table 5 is grams
    table_5_weights = {}
    table_5_nuclide = ORIGEN_data['table_5']['nuclide']
    for key in NUCLIDE_KEYS:
        table_5_weights[key] = np.sum(origen_to_array(table_5_nuclide[key], nucs), axis=0)
    table_5_weights['fission_products'] *= n_fission_fragments

    # Table 4 is atom fraction
    table_4_nuclide = ORIGEN_data['table_4']['nuclide']
    new_data = np.zeros((len(nucs), 1))
    for key in NUCLIDE_KEYS:
        new_data += table_5_weights[key]*origen_to_array(table_4_nuclide[key], nucs)

    return new_data

def origen_data_to_array_atom_fraction(ORIGEN_data, nucs):
    # Table 4 is atom fraction
    table_4_nuclide = ORIGEN_data['table_4']['nuclide']
    new_data = np.zeros((len(nucs), 1))
    for key in NUCLIDE_KEYS:
        new_data += origen_to_array(table_4_nuclide[key], nucs)

    return new_data

def origen_data_to_array_materials(ORIGEN_data, nucs):
    material = ORIGEN_data['materials'][1]
    new_data = np.zeros((len(nucs), 1))
    nuc_to_idx = {v: i for i, v in enumerate(nucs)}

    for nuc, atom_frac in material.to_atom_frac().items():
        new_data[nuc_to_idx[pyne.nucname.name(nuc)]] = atom_frac

    return new_data

def hash_data(vec, library, time, phi, n_fission_fragments):
    return hash((tuple(vec.flat), library, time, phi, n_fission_fragments))

def initial_vector(start_nuclide, nucs):
    nuc_to_idx = {v: i for i, v in enumerate(nucs)}
    return csr_matrix(([1], [[nuc_to_idx[start_nuclide]], [0]]),
        shape=(len(nucs), 1))

def test_origen_data_sanity(ORIGEN_data):
    for table in ['table_4', 'table_5']:
        assert table in ORIGEN_data, table
        assert 'nuclide' in ORIGEN_data[table]

        nuclide = ORIGEN_data['table_4']['nuclide']

        # Sanity check
        for comb in combinations(NUCLIDE_KEYS, 2):
            a, b = comb
            for common in set.intersection(set(nuclide[a]), set(nuclide[b])):
                array_a, array_b = nuclide[a][common], nuclide[b][common]
                assert np.allclose(array_a, 0) \
                    or np.allclose(array_b, 0)
                    # or np.allclose(array_a, array_b)

def create_hdf5_table(file, lib, nucs):
    nucs_size = len(nucs)
    desc_common = [
        ('hash', np.int64),
        ('library', 'S8'),
        ('initial vector', np.float64, (nucs_size, 1)),
        ('time', np.float64),
        ('phi', np.float64),
        ('n_fission_fragments', np.float64),
    ]
    desc_origen = [
        ('execution time ORIGEN', np.float64),
        ('ORIGEN atom fraction', np.float64, (nucs_size, 1)),
        ('ORIGEN mass fraction', np.float64, (nucs_size, 1)),
        ]
    desc_cram = [
        ('execution time CRAM', np.float64),
        ('CRAM atom fraction', np.float64, (nucs_size, 1)),
        ('CRAM mass fraction', np.float64, (nucs_size, 1)),
        ]

    h5file = tables.open_file(file, mode="a", title="CRAM/ORIGEN test run data", filters=tables.Filters(complevel=1))
    h5file.create_group('/', lib, '%s data' % lib)
    h5file.create_table('/' + lib, 'origen', np.dtype(desc_common + desc_origen))
    h5file.create_table('/' + lib, 'cram', np.dtype(desc_common + desc_cram))
    h5file.create_array('/' + lib, 'nucs', np.array(nucs, 'S6'))

def save_file_origen(file, *, ORIGEN_data, lib, nucs, start_nuclide, time,
    phi, ORIGEN_time, n_fission_fragments=2.004):

    with tables.open_file(file, mode="a", title="ORIGEN and CRAM data",
        filters=tables.Filters(complevel=1)) as h5file:

        if lib not in h5file.root:
            create_hdf5_table(file, lib, nucs)

        table = h5file.get_node(h5file.root, lib + '/origen')
        table.row['initial vector'] = vec = initial_vector(start_nuclide, nucs).toarray()
        table.row['library'] = lib
        table.row['hash'] = hash_data(vec, lib, time, phi, n_fission_fragments)
        table.row['time'] = time
        table.row['phi'] = phi
        table.row['n_fission_fragments'] = n_fission_fragments
        table.row['execution time ORIGEN'] = ORIGEN_time
        table.row['ORIGEN atom fraction'] = origen_data_to_array_weighted(ORIGEN_data, nucs, n_fission_fragments=n_fission_fragments)
        table.row['ORIGEN mass fraction'] = origen_data_to_array_materials(ORIGEN_data, nucs)
        table.row.append()
        table.flush()

def save_file_cram(file, *, CRAM_res, lib, nucs, start_nuclide, time,
    phi, CRAM_time, n_fission_fragments=2.004):
    assert len(CRAM_res) == len(nucs)
    with tables.open_file(file, mode="a", title="ORIGEN and CRAM data",
        filters=tables.Filters(complevel=1)) as h5file:

        if lib not in h5file.root:
            create_hdf5_table(file, lib, nucs)

        table = h5file.get_node(h5file.root, lib + '/cram')
        table.row['initial vector'] = vec = initial_vector(start_nuclide, nucs).toarray()
        table.row['library'] = lib
        table.row['hash'] = hash_data(vec, lib, time, phi, n_fission_fragments)
        table.row['time'] = time
        table.row['phi'] = phi
        table.row['n_fission_fragments'] = n_fission_fragments
        table.row['execution time CRAM'] = CRAM_time
        table.row['CRAM atom fraction'] = CRAM_res
        CRAM_res_normalized = CRAM_res/np.sum(CRAM_res)
        table.row['CRAM mass fraction'] = CRAM_res_normalized
        table.row.append()
        table.flush()

def test_origen_against_CRAM(xs_tape9, time, nuclide, phi):
    e_complex = CRAM_matrix_exp_lambdify()

    logger.info("Running CRAM %s at time=%s, nuclide=%s, phi=%s", xs_tape9, time, nuclide, phi)
    logger.info('-'*80)

    npzfilename = os.path.join('data', os.path.splitext(os.path.basename(xs_tape9))[0] + '_' +
    str(phi) + '.npz')

    nucs, mat = load_sparse_csr(npzfilename)
    assert mat.shape[1] == len(nucs)
    b = initial_vector(nuclide, nucs)

    CRAM_time, CRAM_res = time_func(e_complex, -mat*float(time), b)
    CRAM_res = np.asarray(CRAM_res)

    logger.info("CRAM time: %s", CRAM_time)

    return CRAM_time, CRAM_res

def compute_mismatch(ORIGEN_data, CRAM_res, nucs, rtol=1e-3, atol=1e-5):
    """
    Computes a mismatch analysis for an ORIGEN run vs. CRAM

    The default rtol is 1e-3 because ORIGEN returns 3 digits.

    The default atol is 1e-5 because ORIGEN stops the taylor expansion with
    the error term exp(ASUM)*ASUM**n/n! (using Sterling's approximation),
    where n = 3.5*ASUM + 6 and ASUM is the max of the column sums. The max of
    the column sums is ~2 because of fission, giving ~1e-5 (see ORIGEN lines
    5075-5100)

    """
    CRAM_res_normalized = CRAM_res/np.sum(CRAM_res)

    ORIGEN_res_weighted = origen_data_to_array_weighted(ORIGEN_data, nucs,)
    ORIGEN_res_materials = origen_data_to_array_materials(ORIGEN_data, nucs)
    # ORIGEN_res_atom_fraction = origen_data_to_array_atom_fraction(origen_data, nucs)

    for C, O, units in [
        (CRAM_res, ORIGEN_res_weighted, 'atom fractions'),
        (CRAM_res_normalized, ORIGEN_res_materials, 'mass fractions'),
        # (CRAM_res_normalized, ORIGEN_res_atom_fraction, 'atom fraction'),
        ]:

        logger.info("Units: %s", units)
        try:
            np.testing.assert_allclose(C, O, rtol=rtol, atol=atol)
        except AssertionError as e:
            logger.info(e)
            logger.info("Mismatching elements sorted by error (CRAM, ORIGEN, symmetric relative error)")
            A = np.isclose(C, O, rtol=rtol, atol=atol)
            rel_error = abs(C - O)/(C + O)
            for i, in np.argsort(rel_error, axis=0)[::-1]:
                if A[i]:
                    continue
                logger.info("%s %s %s %s", nucs[i], C[i], O[i], rel_error[i])
        else:
            logger.info("Arrays match with rtol=%s atol=%s", rtol, atol)

        logger.info('')

    # TODO: return some information here
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
    p.add_argument('--no-run-origen', action='store_false', dest='run_origen',
        help="Don't run origen")
    p.add_argument('--no-run-cram', action='store_false', dest='run_cram',
        help="Don't run cram")
    p.add_argument('--hdf5-file', default='data/results.hdf5', help="""hdf5 file
    to write results to""")
    return p

def execute(xs_tape9, time, phi, nuclide, hdf5_file='data/results.hdf5',
    decay_tape9=decay_TAPE9, origen=ORIGEN, run_origen=True, run_cram=True):
    lib = os.path.splitext(os.path.basename(xs_tape9))[0]

    npzfilename = os.path.join('data', lib + '_' + str(phi) + '.npz')
    nucs, mat = load_sparse_csr(npzfilename)

    if run_origen:
        ORIGEN_time, ORIGEN_data = execute_origen(xs_tape9, time, nuclide, phi,
            origen, decay_tape9)
        test_origen_data_sanity(ORIGEN_data)
        save_file_origen(hdf5_file,
            ORIGEN_data=ORIGEN_data,
            lib=lib,
            nucs=nucs,
            start_nuclide=nuclide,
            time=time,
            phi=phi,
            ORIGEN_time=ORIGEN_time,
        )

    if run_cram:
        CRAM_time, CRAM_res = test_origen_against_CRAM(xs_tape9, time, nuclide, phi)
        save_file_cram(hdf5_file,
            CRAM_res=CRAM_res,
            lib=lib,
            nucs=nucs,
            start_nuclide=nuclide,
            time=time,
            phi=phi,
            CRAM_time=CRAM_time,
        )

    if run_origen and run_cram:
        compute_mismatch(ORIGEN_data, CRAM_res, nucs)

def main():
    p = make_parser()
    try:
        import argcomplete
        argcomplete.autocomplete(p)
    except ImportError:
        pass
    args = p.parse_args()
    execute(**vars(args))

if __name__ == '__main__':
    main()
