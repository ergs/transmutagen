from argparse import ArgumentParser
import json
import os
import sys

from jinja2 import Environment
from sympy import im
import numpy as np

import pyne.utils
import pyne.data
from pyne import nucname

from .cram import get_CRAM_from_cache
from .partialfrac import thetas_alphas
from . import __version__

HEADER = """\
/* This file was generated automatically with transmutagen version {{__version__}}. */
/* The command used to generate this file was: python -m transmutagen.gensolve {{' '.join(sys.argv[1:])}}*/
#ifndef {{namespace.upper()}}_SOLVE_C
#define {{namespace.upper()}}_SOLVE_C

{% if py_solve %}
#include <complex.h>
{%- endif %}

typedef struct {{namespace}}_transmute_info_tag {
  int n;
  int nnz;
  int* i;
  int* j;
  char** nucs;
  int* nucids;
  double* decay_matrix;
} {{namespace}}_transmute_info_t;

extern {{namespace}}_transmute_info_t {{namespace}}_transmute_info;

int {{namespace}}_transmute_ij(int i, int j);

int {{namespace}}_transmute_nucid_to_i(int nucid);

{% if py_solve %}
{%- for type, typefuncname in types %}
void {{namespace}}_solve_{{typefuncname}}({{type}}* A, {{type}}* b, {{type}}* x);
void {{namespace}}_diag_add_{{typefuncname}}({{type}}* A, {{type}} alpha);
void {{namespace}}_dot_{{typefuncname}}({{type}}* A, {{type}}* x, {{type}}* y);
void {{namespace}}_scalar_times_vector_{{typefuncname}}({{type}}, {{type}}*);
{% endfor %}
{%- endif %}
{%- for degree in degrees %}
void {{namespace}}_expm_multiply{{degree}}(double* A, double* b, double* x);
{%- endfor %}
#endif

"""

SRC = """\
/* This file was generated automatically with transmutagen version {{__version__}}. */
/* The command used to generate this file was: python -m transmutagen.gensolve {{' '.join(sys.argv[1:])}}*/
#include <string.h>

#include <complex.h>

{% if timing_test %}
#include <time.h>
#include <stdio.h>
{% endif %}

#include "{{headerfilename}}"

const int {{namespace.upper()}}_I[{{NNZ}}] =
  { {%- for i, j in sorted(ij) %}{{i}},{% endfor -%} };

const int {{namespace.upper()}}_J[{{NNZ}}] =
  { {%- for i, j in sorted(ij) %}{{j}},{% endfor -%} };

const char* {{namespace.upper()}}_NUCS[{{N}}] =
  { {%- for nuc in nucs %}"{{nuc}}",{% endfor -%} };

const int {{namespace.upper()}}_NUCIDS[{{N}}] =
  { {%- for nuc in nucs %}{{nucname.id(nuc)}},{% endfor -%} };

const double {{namespace.upper()}}_DECAY_MATRIX[{{NNZ}}] =
  {%- if len(decay_matrix) > 0 %}
  { {%- for x in decay_matrix %}{{x.hex()}},{% endfor -%} };
  {%- else -%}
  { {%- for i in range(NNZ) %}0,{% endfor -%} };
  {% endif %}

{{namespace}}_transmute_info_t {{namespace}}_transmute_info = {
  .n = {{N}},
  .nnz = {{NNZ}},
  .i = (int*) {{namespace.upper()}}_I,
  .j = (int*) {{namespace.upper()}}_J,
  .nucs = (char**) {{namespace.upper()}}_NUCS,
  .nucids = (int*) {{namespace.upper()}}_NUCIDS,
  .decay_matrix = (double*) {{namespace.upper()}}_DECAY_MATRIX,
};

int {{namespace}}_transmute_ij(int i, int j) {
  int n = (i << 16) + j;
  switch (n) {
    {%- for i, j in sorted(ij) %}
    case {{(i * 2**16) + j}}:
        return {{ij[i, j]}};
    {%- endfor %}
    default:
        return -1;
  }
}

int {{namespace}}_transmute_nucid_to_i(int nucid) {
  switch (nucid) {
    {%- for i, nuc in enumerate(nucs) %}
    case {{nucname.id(nuc)}}:
        return {{i}};
    {%- endfor %}
    default:
        return -1;
  }
}


{%- if py_solve %}
{%- for type, typefuncname in types %}
void {{namespace}}_solve_{{typefuncname}}({{type}}* A, {{type}}* b, {{type}}* x) {
  /* Decompose first */
  {{type}} LU [{{NIJK}}];
  memcpy(LU, A, {{NNZ}}*sizeof({{type}}));
  memset(LU+{{NNZ}}, 0, {{NIJK-NNZ}}*sizeof({{type}}));
  {%- for i in range(N) %}
  {%- for j in range(i+1, N) %}
  {%- if (j, i) in ijk %}
  LU[{{ijk[j, i]}}] /= LU[{{ijk[i, i]}}];
  {%- for k in range(i+1, N) %}
  {%- if (i, k) in ijk %}
  LU[{{ijk[j, k]}}] -= LU[{{ijk[j, i]}}] * LU[{{ijk[i, k]}}];
  {%- endif %}
  {%- endfor %}
  {%- endif %}
  {%- endfor %}
  {%- endfor %}

  /* Perform Solve */
  memcpy(x, b, {{N}}*sizeof({{type}}));
  {%- for i in range(N) %}{% if more_than_fore[i] %}
  x[{{i}}] = x[{{i}}]{% for j in range(i) %}{%if (i, j) in ijk%} - LU[{{ijk[i, j]}}]*x[{{j}}]{%endif%}{% endfor %};
  {%- endif %}
  {%- endfor %}
  /* Backward calc */
  {%- for i in range(N-1, -1, -1) %}{%if more_than_back[i]%}
  x[{{i}}] = x[{{i}}]{% for j in range(i+1, N) %}{%if (i, j) in ijk%} - LU[{{ijk[i, j]}}]*x[{{j}}]{%endif%}{% endfor %};
  {%- endif %}
  x[{{i}}] /= LU[{{ijk[i, i]}}];
  {%- endfor %}
}

void {{namespace}}_diag_add_{{typefuncname}}({{type}}* A, {{type}} theta) {
  /* In-place, performs the addition A + theta I, for a scalar theta. */
  {%- for i in range(N) %}
  A[{{ij[i, i]}}] += theta;
  {%- endfor %}
}

void {{namespace}}_dot_{{typefuncname}}({{type}}* A, {{type}}* x, {{type}}* y) {
  /* Performs the caclulation Ax = y and returns y */
  {%- for i in range(N) %}
  y[{{i}}] ={% for j in range(N) %}{% if (i,j) in ij %} + A[{{ij[i, j]}}]*x[{{j}}]{% endif %}{% endfor %};
  {%- endfor %}
}

void {{namespace}}_scalar_times_vector_{{typefuncname}}({{type}} alpha, {{type}}* v) {
  /* In-place, performs alpha*v, for a scalar alpha and vector v. */
  {%- for i in range(N) %}
  v[{{i}}] *= alpha;
  {%- endfor %}
}

{%- endfor %}
{%- endif %}

void {{namespace}}_solve_special(double* A, double complex theta, double complex alpha, double* b, double complex* x) {
  /* Solves (A + theta*I)x = alpha*b and stores the result in x */
  double complex LU [{{NIJK}}];

  /* LU = A + theta*I */
  {%- for i in range(NNZ) %}
  {%- if i in diagonals %}
  LU[{{i}}] = theta + A[{{i}}];
  {%- else %}
  LU[{{i}}] = A[{{i}}];
  {%- endif %}
  {%- endfor %}

  memset(LU+{{NNZ}}, 0, {{NIJK-NNZ}}*sizeof(double complex));

  /* Decompose first */
  {%- for i in range(N) %}
  {%- for j in range(i+1, N) %}
  {%- if (j, i) in ijk %}
  LU[{{ijk[j, i]}}] /= LU[{{ijk[i, i]}}];
  {%- for k in range(i+1, N) %}
  {%- if (i, k) in ijk %}
  LU[{{ijk[j, k]}}] -= LU[{{ijk[j, i]}}] * LU[{{ijk[i, k]}}];
  {%- endif %}
  {%- endfor %}
  {%- endif %}
  {%- endfor %}
  {%- endfor %}

  /* Multiply x by alpha and perform Solve */
  {%- for i in range(N) %}
  x[{{i}}] = alpha*b[{{i}}]{% for j in range(i) %}{%if (i, j) in ijk%} - LU[{{ijk[i, j]}}]*x[{{j}}]{%endif%}{% endfor %};
  {%- endfor %}
  /* Backward calc */
  {%- for i in range(N-1, -1, -1) %}{%if more_than_back[i]%}
  x[{{i}}] = x[{{i}}]{% for j in range(i+1, N) %}{%if (i, j) in ijk%} - LU[{{ijk[i, j]}}]*x[{{j}}]{%endif%}{% endfor %};
  {%- endif %}
  x[{{i}}] /= LU[{{ijk[i, i]}}];
  {%- endfor %}
}

{% for degree in degrees %}
void {{namespace}}_expm_multiply{{degree}}(double* A, double* b, double* x) {
    /* Computes exp(A)*b and stores the result in x */
    {%- for i in range(degree//2) %}
    double complex x{{i}} [{{N}}];
    {%- endfor %}

    {% set thetas, alphas, alpha0 = get_thetas_alphas(degree) -%}
    {% for theta, alpha in sorted(zip(thetas, alphas), key=abs0) if im(theta) >= 0 %}
    {{namespace}}_solve_special(A, {{ -theta}}, {{2*alpha}}, b, x{{loop.index0}});
    {%- endfor %}

    {% for i in range(N) %}
    x[{{i}}] = (double)creal({%- for j in range(degree//2) %}+x{{j}}[{{i}}]{%- endfor %}) + {{alpha0}}*b[{{i}}];
    {%- endfor %}
}

{% endfor %}

{% if timing_test %}

int main(int argc, const char* argv[]) {
    double A[{{NNZ}}];
    double b[{{N}}];
    double x[{{N}}];
    int i;
    double sum = 0.0;
    clock_t start, diff;

    memcpy(A, {{namespace.upper()}}_DECAY_MATRIX, {{NNZ}}*sizeof(double));

    for (i=0; i <= {{N}}; i++) {
        b[i] = 0.0;
    }

    /* U235 */
    b[{{namespace}}_transmute_nucid_to_i(922350000)] = 1.0;

    /* CPU time */
    start = clock();

    {{namespace}}_expm_multiply14(A, b, x);

    diff = clock() - start;
    float msec = (float)diff / CLOCKS_PER_SEC;
    printf("Took %f seconds\\n", msec);

    for (i=0; i <= {{N}}; i++) {
        sum += x[i];
    }

    printf("Sum of resulting vector: %f\\n", sum);

    return(0);
}

{% endif %}
"""

def make_ijk(ij, N):
    ijk = ij.copy()
    idx = len(ij)
    for i in range(N):
        for j in range(i+1, N):
            if (j, i) not in ijk:
                continue
            for k in range(i+1, N):
                if (i, k) in ijk and (j, k) not in ijk:
                    ijk[j, k] = idx
                    idx += 1
    return ijk


def get_thetas_alphas(degree, prec=200, use_cache=True):
    print("Computing coefficients for degree", degree)
    rat_func = get_CRAM_from_cache(degree, prec, log=True, plot=False, use_cache=use_cache)

    thetas, alphas, alpha0 = thetas_alphas(rat_func, prec)
    return thetas, alphas, alpha0


def pyne_decay_matrix(fromto, ijnucs):
    if pyne.utils.use_warnings():
        pyne.utils.toggle_warnings()
    dm = np.zeros(len(fromto), dtype='f8')
    for f, t in fromto:
        decay_const = pyne.data.decay_const(f)
        if decay_const <= 0.0 or np.isnan(decay_const):
            continue
        i = ijnucs[t, f]
        if f == t:
            dm[i] = -decay_const
        else:
            br = pyne.data.branch_ratio(f, t)
            dm[i] = decay_const * br
    return dm


def make_decay_matrix(kind, fromto, ijnucs):
    """makes a decay matrix, if it can, for a given set of allowable fromto reactions."""
    kind = kind.lower()
    if kind == 'none':
        return None
    elif kind == 'pyne':
        return pyne_decay_matrix(fromto, ijnucs)
    else:
        raise ValueError('method for generating decay matrix not understood. Must be '
                         'either "pyne" or "none", got ' + str(kind))


def write_if_diff(filename, contents, verbose=True):
    """Only writes the file if it is different. This prevents touching the file needlessly."""
    if not os.path.isfile(filename):
        existing = None
    else:
        with open(filename, 'r') as f:
            existing = f.read()
    if contents == existing:
        if verbose:
            print(filename + " generated is the same as existing file, skipping.")
        return
    with open(filename, 'w') as f:
        if verbose:
            print("Writing", filename)
        f.write(contents)


def generate(json_file=os.path.join(os.path.dirname(__file__), 'data/gensolve.json'),
    outfile=None, degrees=None, py_solve=False, namespace='transmutagen',
    decay_matrix_kind='pyne', timing_test=False):

    if degrees is None:
        degrees = [6, 8, 10, 12, 14, 16, 18] if py_solve else [14]
    if not outfile:
        outfile='py_solve/py_solve/solve.c' if py_solve else 'solve.c'
    # outfile should always end in .c
    headerfile = outfile[:-2] + '.h'
    headerfilename = os.path.basename(headerfile)

    with open(json_file) as f:
        json_data = json.load(f)

    nucs = json_data['nucs']
    N = len(nucs)
    ijkeys = [(nucs.index(j), nucs.index(i)) for i, j in json_data['fromto']]
    ij = {k: l for l, k in enumerate(sorted(ijkeys))}
    ijk = make_ijk(ij, N)
    ijnucs = {(nucs[i], nucs[j]): k for (i, j), k in ijk.items()}
    diagonals = {ij[i, i]: i for i in range(N)}
    more_than_fore = [len([j for j in range(i+1) if (i, j) in ijk]) > 1 for i in range(N)]
    more_than_back = [len([j for j in range(i, N) if (i, j) in ijk]) > 1 for i in range(N)]
    decay_matrix = make_decay_matrix(decay_matrix_kind, json_data['fromto'], ijnucs)
    types = [  # C type, type function name
             ('double', 'double'),
             ('double complex', 'complex')]
    env = Environment()
    src_template = env.from_string(SRC, globals=globals())
    src = src_template.render(N=N, ij=ij, ijk=ijk, nucs=nucs, sorted=sorted,
        len=len, more_than_back=more_than_back, NNZ=len(ij), NIJK=len(ijk),
        more_than_fore=more_than_fore, types=types, namespace=namespace,
        diagonals=diagonals, degrees=degrees, py_solve=py_solve,
        get_thetas_alphas=get_thetas_alphas, im=im, abs0=lambda i:abs(i[0]),
        zip=zip, enumerate=enumerate, headerfilename=headerfilename,
        __version__=__version__, sys=sys, decay_matrix=decay_matrix,
        nucname=nucname, timing_test=timing_test)
    header_template = env.from_string(HEADER, globals=globals())
    header = header_template.render(types=types, degrees=degrees,
        py_solve=py_solve, namespace=namespace)
    write_if_diff(outfile, src)
    write_if_diff(headerfile, header)
    # If this changes, also update py_solve/setup.py
    gcc_compiler_flags = ['-O0', '-fcx-fortran-rules', '-fcx-limited-range',
        '-ftree-sra', '-ftree-ter', '-fexpensive-optimizations']
    clang_compiler_flags = ['-O0', '--ffast-math']

    print("With gcc, it is recommended to compile the following flags:", ' '.join(gcc_compiler_flags))
    print("With clang, it is recommended to compile the following flags:", ' '.join(clang_compiler_flags))

def main(args=None):
    p = ArgumentParser('gensolver')
    p.add_argument('--json-file',
        default=os.path.join(os.path.dirname(__file__), 'data/gensolve.json'),
        help="""Location of the json input file. An input file can be
        generated from ORIGEN libraries with python -m
        transmutagen.generate_json. The default is %(default)r.""")
    p.add_argument('--py-solve', action='store_true', help="""Generate code for
        py_solve.""")
    p.add_argument('--degrees', nargs='+', default=None, help="""expm_multiply
        degrees to generate. The default is 14, unless --py-solve is
        specified, in which case the default is '6 8 10 12 14 16 18'. The
        orders should be even integers only.""", metavar="DEGREE", type=int)
    p.add_argument('--outfile', help="""Location to write the C file to.
        Should end in '.c'. The default is 'solve.c', unless --py-solve is
        specified, in which case the default is 'py_solve/py_solve/solve.c'.
        The header file will be generated alongside it.""")
    p.add_argument('--namespace', default='transmutagen', help="""Namespace
        for the generated solver. The default is %(default)r.""")
    p.add_argument('--decay-matrix', default='pyne', dest='decay_matrix_kind',
                   choices={'none', 'None', 'NONE', 'pyne', 'Pyne', 'PyNE', 'PYNE'},
                   help='method for generating included decay matrix, default "pyne".')
    p.add_argument("--timing-test", action='store_true', default=False,
        help="""Generate a main() function that does a timing test on the decay
        matrix.""")

    ns = p.parse_args(args=args)
    if ns.outfile and not ns.outfile.endswith('.c'):
        p.error("--outfile should end with '.c'")
    arguments = ns.__dict__.copy()

    generate(**arguments)


if __name__ == "__main__":
    main()
