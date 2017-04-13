import datetime
import difflib
import inspect
import os
import logging
import time
from functools import wraps
import io

import numpy as np
from scipy.sparse import csr_matrix
import mpmath
from sympy import lambdify, symbols

from sympy.utilities.decorator import conserve_mpmath_dps

t = symbols('t', real=True)

def diff_strs(a, b):
    """
    Print a colored character-by-character diff of a and b.

    Requires colorama to be installed.

    Differing characters from the a string are printed in red and those from
    the b string are printed in green.

    Note that this can be slow for large strings (more than 10000 characters).

    """
    try:
        import colorama
    except ImportError:
        raise ImportError("colorama is required to use diff_strs")

    s = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    for op, i1, j1, i2, j2 in s.get_opcodes():
        if op == 'equal':
            print(a[i1:j1], end='')
        elif op == 'replace':
            print(colorama.Fore.RED, a[i1:j1], colorama.Fore.GREEN, b[i2:j2], sep='', end='')
        elif op == 'insert':
            print(colorama.Fore.GREEN, b[i2:j2], sep='', end='')
        elif op == 'delete':
            print(colorama.Fore.RED, a[i1:j1], sep='', end='')
        print(colorama.Style.RESET_ALL, end='')
    print()

def relative_error(exact, approx):
    return abs(exact - approx)/exact

def mean_log10_relative_error(exact, approx):
    import numpy as np
    return np.mean(np.log10(abs(relative_error(exact, approx))))

# Function to create plot like in "Computing the Matrix Exponential in Burnup
# Calculations", Pusa and Leppänen:
# mpmath.cplot(lambdify(t, rat_func14 - exp(-t), 'mpmath'), re=[0, 100], im=[-30, 30], color=lambda i: -mpmath.floor(mpmath.log(abs(i), 10))/(30 - mpmath.floor(mpmath.log(abs(i), 10))), points=100000, verbose=True)

@conserve_mpmath_dps
def plot_in_terminal(expr, *args, prec=None, logname=None, **kwargs):
    """
    Run plot() but show in terminal if possible
    """
    from mpmath import plot
    if prec:
        mpmath.mp.dps = prec
    f = lambdify(t, expr, mpmath)
    try:
        from iterm2_tools.images import display_image_bytes
    except ImportError:
        if logname:
            os.makedirs('plots', exist_ok=True)
            file = 'plots/%s.png' % logname
        else:
            file = None
        plot(f, *args, file=file, **kwargs)
    else:
        from io import BytesIO
        b = BytesIO()
        plot(f, *args, **kwargs, file=b)
        if logname:
            os.makedirs('plots', exist_ok=True)
            with open('plots/%s.png' % logname, 'wb') as f:
                f.write(b.getvalue())
        print(display_image_bytes(b.getvalue()))

def plt_show_in_terminal(logname=None):
    import matplotlib.pyplot as plt
    try:
        from iterm2_tools.images import display_image_bytes
    except ImportError:
        plt.show()
    else:
        b = io.BytesIO()
        plt.savefig(b, format='png')
        print(display_image_bytes(b.getvalue()))
    if logname:
        os.makedirs('plots', exist_ok=True)
        plt.savefig('plots/%s.png' % logname, format='png')

def _get_log_file_name(locals_dict):
    d = locals_dict.copy()
    kwargs = d.pop('kwargs')
    d.update(kwargs)
    d.setdefault('maxsteps')
    d.setdefault('division')
    d.pop('log_to_file', None)
    degree = d.pop('degree')
    prec = d.pop('prec')
    info = 'degree=%s prec=%s ' % (degree, prec)
    info += ' '.join('%s=%s' % (i, d[i]) for i in sorted(d))
    return info


def log_function_args(func):
    """
    Decorator to log the arguments to func, and other info
    """
    logger = func.__globals__['logger']

    @wraps(func)
    def _func(*args, **kwargs):
        func_name = func.__name__
        logger.info("%s with arguments %s", func_name, args)
        logger.info("%s with keyword arguments %s", func_name, kwargs)

        os.makedirs('logs', exist_ok=True)
        binding = inspect.signature(func).bind(*args, **kwargs)
        binding.apply_defaults()
        logname = _get_log_file_name(binding.arguments)
        if kwargs.get('log_to_file', False):
            logger.addHandler(logging.FileHandler('logs/%s.log' % logname, delay=True))
        logger.info("Logging to file 'logs/%s.log'", logname)

        kwargs['logname'] = logname

        starttime = datetime.datetime.now()
        logger.info("Start time: %s", starttime)
        try:
            return func(*args, **kwargs)
        except BaseException as e:
            logger.error("Exception raised", exc_info=True)
            raise
        finally:
            endtime = datetime.datetime.now()
            logger.info("End time: %s", endtime)
            logger.info("Total time: %s", endtime - starttime)

    return _func


def save_sparse_csr(filename, array, nucs, phi):
    """Saves a sparse CSR matrix to disk"""
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape, nucs=nucs,
             phi=phi)

def load_sparse_csr(filename):
    loader = np.load(filename)
    return (loader['nucs'], csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                        shape=loader['shape']))


def time_func(f, *args, **kwargs):
    """
    Times f(*args, **kwargs)

    Returns (time_elapsed, f(*args, **kwargs)).

    """
    t = time.perf_counter()
    res = f(*args, **kwargs)
    return time.perf_counter() - t, res

def memoize(f):
    memo = {}
    @wraps(f)
    def inner(*args, **kwargs):
        hashable_kwargs = frozenset(kwargs.items())
        if (args, hashable_kwargs) not in memo:
            memo[args, hashable_kwargs] = f(*args, **kwargs)

        return memo[args, hashable_kwargs]

    return inner
