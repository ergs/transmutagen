#! /usr/bin/env python
try:
    from setuptools import setup
    HAVE_SETUPTOOLS = True
except ImportError:
    from distutils.core import setup
    HAVE_SETUPTOOLS = False

import numpy as np

import versioneer

VERSION = versioneer.get_version()
setup_kwargs = {
    "version": VERSION,
    "cmdclass": versioneer.get_cmdclass(),
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
    "package_data": {'transmutagen': ['data/gensolve.json', 'data/CRAM_cache/*']},
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
        'Cython'
        ]

if __name__ == '__main__':
    setup(
        name='transmutagen',
        packages=['transmutagen'],
        long_description=open('README.md').read(),
        include_dirs = [np.get_include()],
        **setup_kwargs
        )
