from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

from .cram import (general_rat_func, nsolve_intervals, nsolve_points,
    CRAM_exp, get_CRAM_from_cache, CRAM_coeffs)

__all__ = ['general_rat_func', 'nsolve_intervals', 'nsolve_points',
    'CRAM_exp', 'get_CRAM_from_cache', 'CRAM_coeffs']

from .partialfrac import (t, thetas_alphas, thetas_alphas_to_expr_real,
    thetas_alphas_to_expr_complex, thetas_alphas_to_expr_complex2,
    thetas_alphas_to_expr_expanded, allroots,
    multiply_vector, customre)

__all__ += ['t', 'thetas_alphas', 'thetas_alphas_to_expr_real',
    'thetas_alphas_to_expr_complex', 'thetas_alphas_to_expr_complex2',
    'thetas_alphas_to_expr_expanded', 'allroots', 'multiply_vector',
    'customre']

from .codegen import (MatrixNumPyPrinter, autoeye, numpy_solve_with_autoeye,
    scipy_sparse_solve_with_autoeye, scipy_translations,
    scipy_translations_autoeye, CRAM_matrix_exp_lambdify)

__all__ += ['MatrixNumPyPrinter', 'autoeye', 'numpy_solve_with_autoeye',
    'scipy_sparse_solve_with_autoeye', 'scipy_translations',
    'scipy_translations_autoeye', 'CRAM_matrix_exp_lambdify']

from .origen import (execute_origen, origen_to_array,
    origen_data_to_array_weighted, origen_data_to_array_materials,
    origen_data_to_array_atom_fraction, initial_vector, compute_mismatch)

__all__ += ['execute_origen', 'origen_to_array',
    'origen_data_to_array_weighted', 'origen_data_to_array_materials',
    'origen_data_to_array_atom_fraction', 'initial_vector',
    'compute_mismatch']

from .origen_all import (ALL_LIBS, INITIAL_NUCS, DAY, MONTH, YEAR, TEN_YEARS,
    THOUSAND_YEARS, MILLION_YEARS, TIME_STEPS, PHI)

__all__ += ['ALL_LIBS', 'INITIAL_NUCS', 'DAY', 'MONTH', 'YEAR', 'TEN_YEARS',
    'THOUSAND_YEARS', 'MILLION_YEARS', 'TIME_STEPS', 'PHI']

from .gensolve import generate

__all__ += ['generate']

from .generate_json import common_mat, generate_json

__all__ += ['common_mat', 'generate_json']

from .tape9utils import (LN2, DECAY_RXS, PAROFF, PAROFFM, origen_to_name,
    decay_data, XS_RXS, XS_TO_ORIGEN, ORIGEN_TO_XS, find_nlb,
    cross_section_data, sort_nucs, create_dok, dok_to_sparse_info,
    find_decaylib, SPMAT_FORMATS, tape9_to_sparse, normalize_tape9s)

__all__ += ['LN2', 'DECAY_RXS', 'PAROFF', 'PAROFFM', 'origen_to_name',
    'decay_data', 'XS_RXS', 'XS_TO_ORIGEN', 'ORIGEN_TO_XS', 'find_nlb',
    'cross_section_data', 'sort_nucs', 'create_dok', 'dok_to_sparse_info',
    'find_decaylib', 'SPMAT_FORMATS', 'tape9_to_sparse', 'normalize_tape9s']

from .util import (diff_strs, relative_error, mean_log10_relative_error,
    plot_in_terminal, plt_show_in_terminal, log_function_args,
    save_sparse_csr, load_sparse_csr, time_func, memoize)

__all__ += ['diff_strs', 'relative_error', 'mean_log10_relative_error',
    'plot_in_terminal', 'plt_show_in_terminal', 'log_function_args',
    'save_sparse_csr', 'load_sparse_csr', 'time_func', 'memoize']
