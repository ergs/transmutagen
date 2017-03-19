#! /usr/bin/env python
import os
try:
    from setuptools import setup
    HAVE_SETUPTOOLS = True
except ImportError:
    from distutils.core import setup
    HAVE_SETUPTOOLS = False


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
if os.path.exists('transmutagen/solve.c'):
    from Cython.Build import cythonize
    from distutils.extension import Extension

    sourcefiles = ['transmutagen/py_solve.pyx', 'transmutagen/solve.c']
    extensions = [Extension("transmutagen.solve", sourcefiles)]
    setup_kwargs['ext_modules'] = cythonize(extensions)


if __name__ == '__main__':
    setup(
        name='transmutagen',
        packages=['transmutagen'],
        long_description=open('README.md').read(),
        **setup_kwargs
        )
