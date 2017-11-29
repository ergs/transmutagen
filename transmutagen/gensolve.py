from argparse import ArgumentParser
import json
import os
import sys
import time
import inspect
import textwrap
import subprocess

from jinja2 import Environment
from sympy import im
import numpy as np

import pyne.utils
import pyne.data
from pyne import nucname

from .cram import get_CRAM_from_cache
from .partialfrac import thetas_alphas
from . import __version__

# If this changes, also update py_solve/setup.py
GCC_COMPILER_FLAGS = ['-O0', '-fcx-fortran-rules', '-fcx-limited-range',
        '-ftree-sra', '-ftree-ter', '-fexpensive-optimizations']
CLANG_COMPILER_FLAGS = ['-O0', '--ffast-math']

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
void {{namespace}}_expm_multiply{{degree}}(double* A, double* b, double* x {%- if include_lost_bits %}, double* lost_bits {% endif %});
{%- endfor %}
#endif

"""

SRC = """\
/* This file was generated automatically with transmutagen version {{__version__}}. */
/* The command used to generate this file was: python -m transmutagen.gensolve {{' '.join(sys.argv[1:])}}*/
#include <string.h>

#include <complex.h>
{%- if include_lost_bits %}
#include <math.h>
#include <stdlib.h>
{%- endif %}

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

void {{namespace}}_solve_special(double* A, double complex theta, double complex alpha, double* b, double complex* x {%- if include_lost_bits %}, double* lost_bits{% endif %}) {
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

  {%- if include_lost_bits %}
  x[{{i}}] = alpha*b[{{i}}];
  {%- for j in range(i) %}
  {% if (i, j) in ijk %}
  x[{{i}}] -= LU[{{ijk[i, j]}}]*x[{{j}}];
  if (creal(x[{{i}}]) && creal(LU[{{ijk[i, j]}}]*x[{{j}}]) && creal(x[{{i}}])*creal(LU[{{ijk[i, j]}}]*x[{{j}}]) < 0) {
      if (abs(creal(x[{{i}}])) > abs(creal(LU[{{ijk[i, j]}}]*x[{{j}}]))) {
          lost_bits[{{i}}] += log2(1 - abs(creal(LU[{{ijk[i, j]}}]*x[{{j}}]))/abs(creal(x[{{i}}])));
      } else {
          lost_bits[{{i}}] += log2(1 - abs(creal(x[{{i}}]))/abs(creal(LU[{{ijk[i, j]}}]*x[{{j}}])));
      }
  }
  {%- endif %}
  {%- endfor %}
  {%- else %}
  x[{{i}}] = alpha*b[{{i}}]{% for j in range(i) %}{%if (i, j) in ijk%} - LU[{{ijk[i, j]}}]*x[{{j}}]{%endif%}{% endfor %};
  {%- endif %}

  {%- endfor %}

  /* Backward calc */
  {%- for i in range(N-1, -1, -1) %}{%if more_than_back[i]%}

  {%- if include_lost_bits %}
  x[{{i}}] = x[{{i}}];
  {%- for j in range(i+1, N) %}
  {%- if (i, j) in ijk %}
  x[{{i}}] -= LU[{{ijk[i, j]}}]*x[{{j}}];
  if (creal(x[{{i}}]) && creal(LU[{{ijk[i, j]}}]*x[{{j}}]) && creal(x[{{i}}])*creal(LU[{{ijk[i, j]}}]*x[{{j}}]) < 0) {
      if (abs(creal(x[{{i}}])) > abs(creal(LU[{{ijk[i, j]}}]*x[{{j}}]))) {
          lost_bits[{{i}}] += log2(1 - abs(creal(LU[{{ijk[i, j]}}]*x[{{j}}]))/abs(creal(x[{{i}}])));
      } else {
          lost_bits[{{i}}] += log2(1 - abs(creal(x[{{i}}]))/abs(creal(LU[{{ijk[i, j]}}]*x[{{j}}])));
      }
  }
  {%- endif %}
  {%- endfor %}
  {%- else %}
  x[{{i}}] = x[{{i}}]{% for j in range(i+1, N) %}{%if (i, j) in ijk%} - LU[{{ijk[i, j]}}]*x[{{j}}]{%endif%}{% endfor %};
  {%- endif %}

  {%- endif %}
  x[{{i}}] /= LU[{{ijk[i, i]}}];
  {%- endfor %}
}

{% for degree in degrees %}
void {{namespace}}_expm_multiply{{degree}}(double* A, double* b, double* x {%- if include_lost_bits %}, double* lost_bits{% endif %}) {
    /* Computes exp(A)*b and stores the result in x */
    {%- for i in range(degree//2) %}
    double complex x{{i}} [{{N}}];
    {%- endfor %}

    {%- if include_lost_bits %}
    {%- for i in range(N) %}
    lost_bits[{{i}}] = 0;
    {%- endfor %}
    {%- endif %}

    {% set thetas, alphas, alpha0 = get_thetas_alphas(degree) -%}
    {% for theta, alpha in sorted(zip(thetas, alphas), key=abs0) if im(theta) >= 0 %}
    {{namespace}}_solve_special(A, {{ -theta}}, {{2*alpha}}, b, x{{loop.index0}} {%- if include_lost_bits %}, lost_bits {% endif %});
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

def make_solve_special(ij, N):
    import math
    ijk = make_ijk(ij, N)
    NIJK = len(ijk)
    more_than_back = [len([j for j in range(i, N) if (i, j) in ijk]) > 1 for i in range(N)]

    diagonals = {ij[i, i]: i for i in range(N)}

    _pre_decomposed = {}
    _pre_decomposed_A = None

    def decompose(A, theta):
        nonlocal _pre_decomposed_A
        NNZ = A.shape[0]
        LU = np.zeros(NIJK, dtype=complex)

        if np.all(_pre_decomposed_A == A):
            if theta in _pre_decomposed:
                return _pre_decomposed[theta].copy()
        else:
            _pre_decomposed_A = A.copy()

        # LU = A + theta*I
        LU[:NNZ] = A
        for i in diagonals:
            LU[i] += theta

        # Decompose first
        for i in range(N):
            for j in range(i+1, N):
                if (j, i) in ijk:
                    LU[ijk[j, i]] /= LU[ijk[i, i]]
                    for k in range(i+1, N):
                        if (i, k) in ijk:
                            LU[ijk[j, k]] -= LU[ijk[j, i]] * LU[ijk[i, k]]

        _pre_decomposed[theta] = LU.copy()
        return LU


    def solve_special(A, theta, alpha, b):
        """
        Solves (A + theta*I)x = alpha*b for x
        """
        if len(A.shape) != 1:
            raise TypeError("A should be 1-dimensional")

        if len(b) != N:
            raise TypeError("b should be length %d" % N)

        LU = decompose(A, theta)

        # Multiply x by alpha and perform Solve
        x = alpha*b
        x_lost_bits = np.zeros(x.shape)

        for i in range(N):
            xvals = [x[i]]
            for j in range(i):
                if (i, j) in ijk:
                    rhs = LU[ijk[i, j]]*x[j]
                    if np.real(x[i]) and np.real(rhs):
                        l = abs(np.real(x[i]))
                        r = abs(np.real(rhs))
                        if l > r:
                            x_lost_bits[i] += (l - r)*2**(-53)*(1 - r/l)
                        else:
                            x_lost_bits[i] += (r - l)*2**(-53)*(1 - l/r)
                    xvals.append(-rhs)
            x[i] = math.fsum(np.real(xvals)) + math.fsum(np.imag(xvals))*1j

        # Backward calc
        for i in range(N-1, -1, -1):
            xvals = [x[i]]
            if more_than_back[i]:
                for j in range(i+1, N):
                    if (i, j) in ijk:
                        rhs = LU[ijk[i, j]]*x[j]
                        if np.real(x[i]) and np.real(rhs):
                            l = abs(np.real(x[i]))
                            r = abs(np.real(rhs))
                            if l > r:
                                x_lost_bits[i] += (l - r)*2**(-53)*(1 - r/l)
                            else:
                                x_lost_bits[i] += (r - l)*2**-53*(1 - l/r)
                        xvals.append(-rhs)
            x[i] = math.fsum(np.real(xvals)) + math.fsum(np.imag(xvals))*1j

            x[i] /= LU[ijk[i, i]]

        return x, x_lost_bits

    return solve_special

def make_expm_multiply(degree, solve_special):
    thetas, alphas, alpha0 = get_thetas_alphas(degree)
    thetas = np.array(thetas, dtype=complex)
    alphas = np.array(alphas, dtype=complex)
    alpha0 = np.array([alpha0], dtype=float)

    def expm_multiply(A, b, *, debug=False):
        """Computes exp(A)*b"""
        N = len(b)

        X = np.zeros((degree//2, N, 1), dtype=complex)

        x_lost_bits = np.zeros(b.shape)
        i = 0
        for theta, alpha in sorted(zip(thetas, alphas), key=lambda i:abs(i[0])):
            if im(theta) >= 0:
                if debug:
                    print("Doing solve special", i)
                    print("theta", theta)
                    print("alpha", alpha)
                X[i], lost_bits = solve_special(A, -theta, 2*alpha, b)
                x_lost_bits += lost_bits
                i += 1

        x = np.real(np.sum(X, axis=0)) + alpha0*b

        return x, lost_bits
    return expm_multiply

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


def generate_python_solver(json_file=os.path.join(os.path.dirname(__file__),
    'data/gensolve.json'), json_data=None, degree=14):
    if not json_data:
        with open(json_file) as f:
            json_data = json.load(f)

    nucs = json_data['nucs']
    N = len(nucs)
    ijkeys = [(nucs.index(j), nucs.index(i)) for i, j in json_data['fromto']]
    ij = {k: l for l, k in enumerate(sorted(ijkeys))}

    solve_special = make_solve_special(ij, N)
    expm_multiply = make_expm_multiply(degree, solve_special)
    return expm_multiply

def generate_python_solver_module(json_file=os.path.join(os.path.dirname(__file__),
    'data/gensolve.json'), json_data=None, degree=14, outfile=None):
    """
    A quick and dirty way to convert Python gensolve functions in this file
    into a module, which won't use nested functions.
    """
    import pyflakes.api

    FUNCS = [make_solve_special, make_expm_multiply]
    IMPORTS = """
import numpy as np
from sympy import im
from transmutagen.gensolve import make_ijk, get_thetas_alphas
"""

    if not outfile:
        outfile = 'python_gensolve.py'

    if not json_data:
        with open(json_file) as f:
            json_data = json.load(f)

    nucs = json_data['nucs']
    N = len(nucs)
    ijkeys = [(nucs.index(j), nucs.index(i)) for i, j in json_data['fromto']]
    ij = {k: l for l, k in enumerate(sorted(ijkeys))}

    SRC = IMPORTS
    for func in FUNCS:
        func_name = func.__name__
        func_val = func_name[len('make_'):]
        s = inspect.signature(func)
        source = inspect.getsource(func)
        source = textwrap.dedent('\n'.join(source.splitlines()[1:]))
        source = source.replace('\nreturn ', '\n' + func_val + ' = ')
        for param in s.parameters:
            if param in locals():
                SRC += "{param} = {param_val}\n".format(param=param,
                    param_val=locals()[param])

        SRC += '\n' + source + '\n'

    SRC = SRC.replace('nonlocal ', 'global ')
    errors = pyflakes.api.check(SRC, outfile)
    if errors:
        print("Warning, the generated file has some errors (see above).")

    with open(outfile, 'w') as f:
        f.write(SRC)


def assemble_gnu(outfile):
    """Assembles the solver with GCC. Returns the filename that was generated."""
    base, _ = os.path.splitext(outfile)
    asmfile = base + '-gnu.s'
    cmd = ['gcc', '-fPIC']
    cmd.extend(GCC_COMPILER_FLAGS)
    cmd.extend(['-S', '-o', asmfile, '-c', outfile])
    print('Running command:\n  $ ' + ' '.join(cmd))
    t0 = time.time()
    subprocess.check_call(cmd)
    t1 = time.time()
    print('Assembled in {0:.3} seconds'.format(t1 - t0))
    return asmfile


def archive(filename, files):
    """Creates archive of list of files with a name."""
    import tarfile
    with tarfile.open(filename, 'w:gz') as tar:
        for f in files:
            print('  compressing ' + f)
            tar.add(f)



def generate(json_file=os.path.join(os.path.dirname(__file__),
    'data/gensolve.json'), json_data=None,
    outfile=None, degrees=None, py_solve=False, namespace='transmutagen',
    decay_matrix_kind='pyne', timing_test=False, include_lost_bits=False,
    gnu_asm=False, tar=False):

    if degrees is None:
        degrees = [6, 8, 10, 12, 14, 16, 18] if py_solve else [14]
    if not outfile:
        outfile = 'py_solve/py_solve/solve.c' if py_solve else 'solve.c'
    # outfile should always end in .c
    headerfile = outfile[:-2] + '.h'
    headerfilename = os.path.basename(headerfile)

    if not json_data:
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
        nucname=nucname, timing_test=timing_test,
        include_lost_bits=include_lost_bits)
    header_template = env.from_string(HEADER, globals=globals())
    header = header_template.render(types=types, degrees=degrees,
        py_solve=py_solve, namespace=namespace, include_lost_bits=include_lost_bits)
    write_if_diff(outfile, src)
    write_if_diff(headerfile, header)

    print("With gcc, it is recommended to compile the following flags:", ' '.join(GCC_COMPILER_FLAGS))
    print("With clang, it is recommended to compile the following flags:", ' '.join(CLANG_COMPILER_FLAGS))

    generated = [outfile, headerfile]
    if gnu_asm:
        print("Compiling GNU Assembly with GCC...")
        filename = assemble_gnu(outfile=outfile)
        generated.append(filename)
    if tar:
        base, _ = os.path.splitext(outfile)
        tarfile = base + '.tar.gz'
        print("Archiving generated files in " + tarfile)
        archive(tarfile, generated)



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
    p.add_argument('-o', '--outfile', help="""Location to write the C file to.
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
    p.add_argument("--include-lost-bits", action='store_true', default=False,
        help="""Add an additional argument to the generated expm_multiply functions to keep track of how many floating point bits are potentially lost in the calculation (experimental).""")
    p.add_argument("--gcc-asm", "--gnu-asm", action='store_true', default=False, dest='gnu_asm',
        help="Creates GCC assembly, so that users don't have to go through full compile.")
    p.add_argument("--tar", action='store_true', default=False, dest='tar',
        help="Creates an archive of all files that were generated.")
    p.add_argument("--no-tar", '--dont-tar', action='store_false', dest='tar',
        help="Does not creates an archive of generated files (default).")

    ns = p.parse_args(args=args)
    if ns.outfile and not ns.outfile.endswith('.c'):
        p.error("--outfile should end with '.c'")
    arguments = ns.__dict__.copy()

    generate(**arguments)


if __name__ == "__main__":
    main()
