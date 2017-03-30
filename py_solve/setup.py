#!/usr/bin/env python

"""
This is a dummy module, for testing. To generate it, run

   python -m transmutagen.gensolve

"""
import os
import sys
try:
    from setuptools import setup
    HAVE_SETUPTOOLS = True
except ImportError:
    from distutils.core import setup
    HAVE_SETUPTOOLS = False

import numpy as np

VERSION = '1.0'

setup_kwargs = {
    "version": VERSION,
    "description": 'Code geneartion tools for transmutation solvers',
    "license": 'BSD 3-clause',
    "author": 'ERGS',
    "author_email": 'ergsonomic@googlegroups.com',
    "url": 'https://github.com/ergs/transmutagen',
    "download_url": "https://github.com/ergs/transmutagen/zipball/" + VERSION,
    "classifiers": [
        "License :: OSI Approved",
        "Programming Language :: Python",
        ],
    "zip_safe": False,
    "data_files": [("", ['LICENSE', 'README.md']),],
    }


# Cython parts
if not os.path.exists('py_solve/solve.c'):
    sys.exit("py_solve/solve.c not found. Run python -m transmutagen.gensolve to generate the solver")

from Cython.Build import cythonize
from distutils.extension import Extension

sourcefiles = ['py_solve/py_solve.pyx', 'py_solve/solve.c']
extensions = [Extension("py_solve/py_solve", sourcefiles,
                        extra_compile_args=['-O0', '-fcx-fortran-rules',
                                            '-fcx-limited-range'])]
setup_kwargs['ext_modules'] = cythonize(extensions)


if __name__ == '__main__':
    setup(
        name='py_solve',
        long_description=__doc__,
        include_dirs = [np.get_include()],
        **setup_kwargs
        )
