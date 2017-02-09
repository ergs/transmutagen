"""Some utilities for dealing with Origen 2.2 TAPE9 files."""
import os
import json
import warnings
import numpy as np

from pyne import utils
utils.toggle_warnings()
warnings.simplefilter('ignore')
from pyne.origen22 import parse_tape9
from pyne import nucname


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


def decay_data(t9, nlb=(1, 2, 3), threshold=THRESHOLD):
    """Loads decay data.

    Parameters
    ----------
    t9 : dict
        A TAPE9 dict that contains the decay data.
    nlb : tuple of ints, optional
        The decay library numbers. Usually this is just (1, 2, 3).
    threshold : float
        A cutoff for when we consider nuclides stable or a decay mode
        forbidden. This is given as a decay constant and the default is
        for a half-life that is 10x the age of the universe.

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
    nucs = set()
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
                gammas[nname, cname] = val
                nucs.add(nname)
                nucs.add(cname)
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
            biggest_gamma = max([(i, j) for i, j in gammas if i == nuc],
                                key=lambda t: gammas[t])
            gammas[biggest_gamma] = gammas[biggest_gamma] + 1 - gamma_total
    return nucs, decays_consts, gammas
