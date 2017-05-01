# Transmutagen

Transmutation SymPy Code Generator

[![Build Status](https://travis-ci.org/ergs/transmutagen.svg?branch=master)](https://travis-ci.org/ergs/transmutagen)

# Usage

First, get the git master version of SymPy and set `PYTHONPATH` to point to
it.

Transmutagen also depends on

 - mpmath
 - sympy (git master)
 - matplotlib
 - numpy
 - scipy
 - pyne
 - jinja2
 - gmpy2
 - Cython

## Generating the CRAM approximation to exp(-x) on [0, oo)

Run

    python -m transmutagen.cram D N

where `D` is the degree of the approximation and `N` is the number of digits.
A typical run would be something like

    python -m transmutagen.cram 14 200

Note that the digits returned are not necessarily all accurate. To compute `N`
correct digits one generally needs to many more working digits in the
computation. Furthermore, when roots are taken, the precision may decrease
even further. So it is recommend to always compute the CRAM expression with a
very high number of digits.

There are many options, but it is not recommending changing them unless you
know what you are doing. See `python -m transmutagen --help`. To
increase/reduce the verbosity of the output, use the `--log-level` flag.

If you use iTerm2, install `iterm2-tools` (from conda-forge) to get plots in
the terminal.

Note: all output and plots are logged to the `logs` and `plots` directories.

## Generating solver code

To generate solver code, use

    python -m transmutagen.gensolve

This will generate ``solve.c`` and ``solve.h``. Use

    python -m transmutagen.gensolve --help

to see various options, such as how to change the degrees that are generated,
and the namespace of the generated functions.

This will use a default list of nuclides and sparsity pattern. To add or
remove nuclides, modify the JSON file, and pass it in with

    python -m transmutagen.gensolve --json-file gensolve.json

The format of the JSON file is

``` json
{
    "nucs": ["H1", "H2", ...],
    "tofrom": [["H1", "H1"], ["H1", "H2"], ...]
}
```

Where ``"nucs"`` is a list of nuclides and ``"tofrom"`` is a list of lists of
every possible reaction product pair.

To generate a JSON file from ORIGEN libraries, run

    python -m transmutagen.generate_json /path/to/origen/libs/ --outfile gensolve.json

This will save the JSON to ``gensolve.json``.

The resulting solve.c will have functions
``{namespace}_expm_multiply{N}(double* A, double* b, double* x)``, where
``{namespace}`` is the namespace specified by the ``--namespace`` flag to
``python -m transmutagen.gensolve`` (the default is ``transmutagen``), and
``{N}`` is the degree of the approximation used in the solve, specified by the
``--degree`` flag (the default is ``14``). The function computes ``exp(A)*b``
and stores the result in ``x``.  ``A`` should be in a flattened format,
according to the sparsity pattern the solver was generated from.

## Converting ORIGEN libraries to a sparse matrix representation

If you'd like to convert an origen file to a matrix representation, please
use something like:

    python -m transmutagen.tape9sparse ~/origen22/libs/pwru50.lib 4e14

See `--help` and the `transmutagen.tape9utils` docs for more details.

## Running tests against ORIGEN

Put `ORIGEN.zip` in the `docker/` directory. Also clone `o2prec` to in the
`docker/` directory. Then run

    ./docker/build_and_run.sh

This requires the docker daemon to be running, and may require `sudo`. There
are various options, which you can see with

    ./docker/build_and_run.sh --help

This will run both ORIGEN and transmutagen (CRAM) on a suite of ORIGEN
libraries, starting nuclides, and times, writing the results to
`data/results.hdf5`. The output will also be logged to `logs/origen_all.log`.
Be warned total suite takes over 24 hours to run.
