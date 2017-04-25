"""Some utilities for dealing with Origen 2.2 TAPE9 files."""
import os
import warnings
from collections import defaultdict

import numpy as np

import scipy.sparse

from pyne import utils
utils.toggle_warnings()
warnings.simplefilter('ignore')
from pyne import data
from pyne import rxname
from pyne import nucname
from pyne.origen22 import parse_tape9

LN2 = np.log(2.0)
DECAY_RXS = ['bminus', 'bplus', 'ec', 'alpha', 'it', 'sf', 'bminus_n']
PAROFF = {
    'frac_beta_minus': 10000000,
    'frac_beta_minus_x': 10000001,
    'frac_beta_plus_or_electron_capture': -10000000,
    'frac_beta_plus_or_electron_capture_x': -9999999,
    'frac_alpha': -20040000,
    'frac_isomeric_transition': -1,
    #'frac_spont_fiss': 0,
    'frac_beta_n': 9990000,
    }
PAROFFM = {
    'frac_beta_minus': 9999999,
    'frac_beta_minus_x': 10000000,
    'frac_beta_plus_or_electron_capture': -10000001,
    'frac_beta_plus_or_electron_capture_x': -10000000,
    'frac_alpha': -20040001,
    'frac_isomeric_transition': -1,
    #'frac_spont_fiss': 0,
    'frac_beta_n': 9989999,
    }
# This threshold is a decay constant for a half-life that is 10x the age of
# the universe.
THRESHOLD = LN2/(10 * 13.7e9 * 365.25* 24 * 3600)


def origen_to_name(nuc):
    """Takes a nuclide in Origen format and returns it's human readable name."""
    return nucname.name(nucname.zzaaam_to_id(int(nuc)))


def decay_data(t9, nlb=(1, 2, 3), threshold=THRESHOLD, nucs=None):
    """Gets decay data from a TAPE9.

    Parameters
    ----------
    t9 : dict
        A TAPE9 dict that contains the decay data.
    nlb : tuple of ints, optional
        The decay library numbers. Usually this is just (1, 2, 3).
    threshold : float, optional
        A cutoff for when we consider nuclides stable or a decay mode
        forbidden. This is given as a decay constant and the default is
        for a half-life that is 10x the age of the universe.
    nucs : set or None, optional
        The known set of nuclides.

    Returns
    -------
    nucs : set
        The set of nuclide names.
    decay_consts : dict
        Mapping from nuclide names to decay constants [1/sec]
    gammas : dict
        Mapping from (i, j) nuclide name tuples to the branch ratio
        fraction. This is [unitless] and the sum over all j for a given
        i is guarenteed to sum to 1.
    """
    nucs = set() if nucs is None else nucs
    decay_consts = {}
    gammas = {}
    for n in nlb:
        decay_consts.update({origen_to_name(nuc): LN2/val
                             for nuc, val in t9[n]['half_life'].items()
                             if nuc != 162500})
        for key in t9[n]:
            if not key.startswith('frac_') or key == 'frac_natural_abund' or  \
                                              key == 'frac_spont_fiss':
                continue
            for nuc, val in t9[n][key].items():
                if val < threshold:
                    continue  # forbidden decay
                nname = origen_to_name(nuc)
                if decay_consts[nname] < threshold:
                    continue  # stable nuclide
                poff = PAROFF if int(nuc)%10 == 0 else PAROFFM
                child = nucname.zzaaam_to_id(int(nuc)) + poff[key]
                cname = nucname.name(child)
                nucs.add(nname)
                nucs.add(cname)
                gammas[nname, cname] = val
    # add β- decays
    for nname, val in decay_consts.items():
        if val < threshold:
            continue  # stable nuclide
        gamma_total = sum([v for (n, c), v in gammas.items() if n == nname])
        if gamma_total < 1.0:
            parid = nucname.id(nname)
            poff = PAROFF if parid%10 == 0 else PAROFFM
            child = parid + poff['frac_beta_minus']
            cname = nucname.name(child)
            gammas[nname, cname] = 1.0 - gamma_total
            gamma_total = 1.0
            nucs.add(nname)
            nucs.add(cname)
        if gamma_total > 1.0:
            biggest_gamma = max([(i, j) for i, j in gammas if i == nname],
                                key=lambda t: gammas[t])
            gammas[biggest_gamma] = gammas[biggest_gamma] + 1 - gamma_total
    return nucs, decay_consts, gammas


XS_RXS = ['gamma', 'z_2n', 'z_3n', 'alpha', 'fission', 'proton', 'gamma_1', 'z_2n_1']
XS_TO_ORIGEN = {'gamma': 'gamma', 'z_2n': '2n', 'z_3n': '3n', 'alpha': 'alpha',
                'fission': 'f', 'proton': 'p', 'gamma_1': 'gamma_x', 'z_2n_1': '2n_x'}
ORIGEN_TO_XS = {v: k for k, v in XS_TO_ORIGEN.items()}


def find_nlb(t9, nlb=None):
    """Finds the library numbers in the TAPE9 file."""
    if nlb is not None:
        return nlb
    nlb = set(t9.keys()) - {1, 2, 3}
    if len(nlb) > 3:
        raise ValueError("Too many libraries found in TAPE9 file.")
    return tuple(sorted(nlb))


def cross_section_data(t9, nlb=None, threshold=THRESHOLD, nucs=None):
    """Gets decay data from a TAPE9.

    Parameters
    ----------
    t9 : dict
        A TAPE9 dict that contains the decay data.
    nlb : tuple of ints or None, optional
        The cross section library numbers. If None, this will attempt to discover
        the numbers in the library.
    threshold : float, optional
        A cutoff for when we consider nuclides stable or a decay mode
        forbidden. This is given as a decay constant and the default is
        for a half-life that is 10x the age of the universe.
    nucs : set or None, optional
        The known set of nuclides.

    Returns
    -------
    nucs : set
        The set of nuclide names.
    sigma_ij : dict
        Mapping from (i, j) nuclide name tuples to the cross section for this
        reaction. Note that this does not include the fission cross section [barns].
    sigma_fission : dict
        Mapping from fissionable nuclide names to the fission cross section [barns].
    fission_product_yields : dict
        Mapping from (i, j) nuclide name tuples to the fission product yields for
        that nuclide.
    """
    nlb = find_nlb(t9, nlb=nlb)
    nucs = set() if nucs is None else nucs
    sigma_ij = {}
    sigma_fission = {}
    fission_product_yields = {}
    for n in nlb:
        # grab sigma_ij cross sections
        for rx in t9[n]:
            if not rx.startswith('sigma_') or rx == 'sigma_f':
                continue
            _, _, orx = rx.partition('_')
            xs_rs = ORIGEN_TO_XS[orx]
            for nuc in t9[n][rx]:
                val = t9[n][rx][nuc]
                if val < threshold:
                    continue
                nname = nucname.name(int(nuc))
                try:
                    child = nucname.name(rxname.child(nname, xs_rs))
                except RuntimeError:
                    continue
                nucs.add(nname)
                nucs.add(child)
                sigma_ij[nname, child] = val
        # grab the fission cross section
        if 'sigma_f' in t9[n]:
            rx = 'sigma_f'
            for nuc in t9[n][rx]:
                val = t9[n][rx][nuc]
                if val < threshold:
                    continue
                nname = nucname.name(int(nuc))
                nucs.add(nname)
                sigma_fission[nname] = val
        # grab the fission product yields
        for rx in t9[n]:
            if not rx.endswith('_fiss_yield'):
                continue
            fromnuc, *_ = rx.partition('_')
            fromnuc = nucname.name(fromnuc)
            for k, v in t9[n][rx].items():
                if v < threshold:
                    continue
                tonuc = nucname.name(nucname.zzaaam_to_id(int(k)))
                # origen yields are in percent
                nucs.add(fromnuc)
                nucs.add(tonuc)
                fission_product_yields[fromnuc, tonuc] = v / 100
    return nucs, sigma_ij, sigma_fission, fission_product_yields


def sort_nucs(nucs):
    """Returns the canonical ordering of a collection of nulcides."""
    return sorted(nucs, key=nucname.id)


def create_dok(phi, nucs, decay_consts, gammas, sigma_ij, sigma_fission,
               fission_product_yields):
    """Creates a dictionary-of-keys representation of the transumation data.

    Parameters
    ----------
    phi : float
        The neutron flux in [n / cm^2 / sec]
    nucs : list
        The list of nuclide names in canonical order.
    decay_consts : dict
        Mapping from nuclide names to decay constants [1/sec]
    gammas : dict
        Mapping from (i, j) nuclide name tuples to the branch ratio
        fraction. This is [unitless] and the sum over all j for a given
        i is guarenteed to sum to 1.
    sigma_ij : dict
        Mapping from (i, j) nuclide name tuples to the cross section for this
        reaction. Note that this does not include the fission cross section [barns].
    sigma_fission : dict
        Mapping from fissionable nuclide names to the fission cross section [barns].
    fission_product_yields : dict
        Mapping from (i, j) nuclide name tuples to the fission product yields for
        that nuclide.

    Returns
    -------
    dok : defaultdict
        Mapping from (i, j) nuclide name tuples to the transmutation rate.
    """
    phi = phi * 1e-24  # flux, n / barn / s
    dok = defaultdict(float) # indexed by nuclide name
    # let's first add the cross section channels
    for i, j in sigma_ij:
        v = sigma_ij.get((i, j), 0.0) * phi
        dok[j, i] += v
        dok[i, i] -= v
    # now let's add the fission products
    for (i, j), fpy in fission_product_yields.items():
        dok[j, i] += fpy * sigma_fission.get(i, 0.0) * phi
    for i, sigf in sigma_fission.items():
        dok[i, i] -= sigf * phi
    # now let's add the decay consts
    for (i, j), g in gammas.items():
        dok[j, i] += g * decay_consts[i]
    for i, v in decay_consts.items():
        dok[i, i] -= v
    return dok


def dok_to_sparse_info(nucs, dok):
    """Converts the dict of keys to a sparse info

    Parameters
    ----------
    nucs : list
        The list of nuclide names in canonical order.
    dok : dict
        Dictionary of keys representing a sparse transmutation matrix.

    Returns
    -------
    rows : list
        Row indexs
    cols : list
        Column indexes
    vals : list
        Values at cooresponding row/col index
    shape : 2-tuple of ints
        Represents the size of the matirx
    """
    shape = (len(nucs), len(nucs))
    nuc_idx = {n: i for i, n in enumerate(nucs)}
    rows = []
    cols = []
    vals = []
    for (i, j), v in dok.items():
        if (v == 0.0) or (i not in nuc_idx) or (j not in nuc_idx):
            continue
        rows.append(nuc_idx[i])
        cols.append(nuc_idx[j])
        vals.append(v)
    return rows, cols, vals, shape


def find_decaylib(t9, tape9, decaylib):
    """Finds and loads a decay lib."""
    if len({1, 2, 3} & set(t9.keys())) == 3:
        return t9
    if not os.path.isabs(decaylib):
        d = os.path.dirname(tape9)
        decaylib = os.path.join(d, decaylib)
    decay = parse_tape9(decaylib)
    return decay


# not all sparse matrices accept row, col, val, shape data
SPMAT_FORMATS = {
    'bsr': scipy.sparse.bsr_matrix,
    'coo': scipy.sparse.coo_matrix,
    'csc': scipy.sparse.csc_matrix,
    'csr': scipy.sparse.csr_matrix,
    }


def tape9_to_sparse(tape9s, phi, format='csr', decaylib='decay.lib',
                    include_fission=True, threshold=THRESHOLD):
    """Converts a TAPE9 file to a sparse matrix.

    Parameters
    ----------
    tape9s : str or list of str
        The filename(s) of the tape file(s).
    phi : float
        The neutron flux in [n / cm^2 / sec]
    format : str, optional
        Format of the sparse matrix created.
    decaylib : str, optional
        A path to TAPE9 containg the decay data libraries if the decay libraries
        are not present in tape9. If this is a relative path, it is taken
        relative to the given tape9 location.
    include_fission : bool
        Flag for whether or not the fission data should be included in the
        resultant matrix.
    threshold : float, optional
        A cutoff for when we consider nuclides stable or a decay mode
        forbidden. This is given as a decay constant and the default is
        for a half-life that is 10x the age of the universe.

    Returns
    -------
    mat : scipy.sparse matrix
        A sparse matrix in the specified layout.
    nucs : list
        The list of nuclide names in canonical order.
    """
    all_decays_consts, all_gammas, all_sigma_ij, all_sigma_fission, all_fission_product_yields = [], [], [], [], []
    nucs = set()
    mats = []
    # seed initial nucs with known atomic masses
    data.atomic_mass('u235')
    for tape9 in tape9s:
        # print("Getting data for", tape9)
        t9 = parse_tape9(tape9)
        decay = find_decaylib(t9, tape9, decaylib)

        for i in data.atomic_mass_map.keys():
            if nucname.iselement(i):
                continue
            try:
                nucs.add(nucname.name(i))
            except RuntimeError:
                pass

        # get the tape 9 data
        nucs, decays_consts, gammas = decay_data(decay, threshold=threshold, nucs=nucs)
        nucs, sigma_ij, sigma_fission, fission_product_yields = cross_section_data(t9,
                                                                    threshold=threshold,
                                                                    nucs=nucs)

        if not include_fission:
            sigma_fission = {}
            fission_product_yields = {}
        all_decays_consts.append(decays_consts)
        all_gammas.append(gammas)
        all_sigma_ij.append(sigma_ij)
        all_sigma_fission.append(sigma_fission)
        all_fission_product_yields.append(fission_product_yields)


    nucs = sort_nucs(nucs)
    for i in range(len(tape9s)):
        dok = create_dok(phi, nucs, all_decays_consts[i], all_gammas[i], all_sigma_ij[i], all_sigma_fission[i],
                     all_fission_product_yields[i])
        rows, cols, vals, shape = dok_to_sparse_info(nucs, dok)
        mats.append(SPMAT_FORMATS[format]((vals, (rows, cols)), shape=shape))
    return mats, nucs


def normalize_tape9s(tape9s):
    from .origen_all import ALL_LIBS

    if isinstance(tape9s, str):
        tape9s = [tape9s]

    _tape9s = []
    for tape9 in tape9s[:]:
        if os.path.isdir(tape9):
            _tape9s.extend([os.path.join(tape9, i) for i in ALL_LIBS])
        else:
            _tape9s.append(tape9)
    return _tape9s
