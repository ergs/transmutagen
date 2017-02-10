#! /usr/bin/env python
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


if __name__ == '__main__':
    setup(
        name='transmutagen',
        packages=['transmutagen'],
        long_description=open('README.md').read(),
        **setup_kwargs
        )
