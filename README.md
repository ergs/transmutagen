# Transmutagen

Transmutation SymPy Code Generator

[![Build Status](https://travis-ci.org/ergs/transmutagen.svg?branch=master)](https://travis-ci.org/ergs/transmutagen)

# Usage

First, get the git master version of SymPy and set `PYTHONPATH` to point to
it.

Then run

    python -m transmutagen D N

where `D` is the degree of the approximation and `N` is the number of digits.
A typical run would be something like

    python -m transmutagen 18 30

There are many options, but it is not recommending changing them unless you
know what you are doing. See `python -m transmutagen --help`. To
increase/reduce the verbosity of the output, use the `--log-level` flag.

If you use iTerm2, install `iterm2-tools` (from conda-forge) to get plots in
the terminal.

Note: all output and plots are logged to the `logs` and `plots` directories.
