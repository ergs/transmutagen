#! /usr/bin/env python
try:
    from setuptools import setup
    HAVE_SETUPTOOLS = True
except ImportError:
    from distutils.core import setup
    HAVE_SETUPTOOLS = False

import numpy as np

# We have to import transmutagen.gensolve to build py_solve
import transmutagen
from distutils.command.build_ext import build_ext

from Cython.Build import cythonize
from distutils.extension import Extension

import versioneer

VERSION = versioneer.get_version()
setup_kwargs = {
    "version": VERSION,
    "description": 'Code generation tools for transmutation solvers',
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
    "package_data": {
        'transmutagen': ['data/*.json', 'data/CRAM_cache/*'],
        'transmutagen.tests': ['data/*.npz', 'data/crv_coeffs'],
        'transmutagen.py_solve': ['*.pyx', '*.pxd'],
        },
     }

if HAVE_SETUPTOOLS:
    setup_kwargs['install_requires'] = [
        'mpmath',
        'sympy',
        'matplotlib',
        'numpy',
        'scipy',
        'pyne',
        'jinja2',
        'gmpy2',
        'Cython',
        'scikit-umfpack',
        ]

sourcefiles = ['transmutagen/py_solve/py_solve.pyx',
    'transmutagen/py_solve/solve.c']
extensions = [Extension("transmutagen.py_solve.py_solve", sourcefiles,
    # TODO: Use CLANG_COMPILER_FLAGS with clang
    extra_compile_args=transmutagen.gensolve.GCC_COMPILER_FLAGS)]
setup_kwargs['ext_modules'] = cythonize(extensions)

# TODO: Allow passing gensolve options from setup.py
class transmutagen_build_ext(build_ext):
    def run(self):
        transmutagen.gensolve.generate(py_solve=True)
        return super().run()

if __name__ == '__main__':
    setup(
        cmdclass={**versioneer.get_cmdclass(), 'build_ext': transmutagen_build_ext},
        name='transmutagen',
        packages=['transmutagen', 'transmutagen.tests',
            'transmutagen.py_solve', 'transmutagen.py_solve.tests'],
        long_description=open('README.md').read(),
        include_dirs = [np.get_include()],
        **setup_kwargs
        )
