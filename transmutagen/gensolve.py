from argparse import ArgumentParser
import json
import os
import sys

from jinja2 import Environment
from sympy import im

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
} {{namespace}}_transmute_info_t;

extern {{namespace}}_transmute_info_t {{namespace}}_transmute_info;

int {{namespace}}_transmute_ij(int i, int j);

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

#include "{{headerfilename}}"

const int {{namespace.upper()}}_I[{{NNZ}}] =
  { {%- for i, j in sorted(ij) %}{{i}},{% endfor -%} };

const int {{namespace.upper()}}_J[{{NNZ}}] =
  { {%- for i, j in sorted(ij) %}{{j}},{% endfor -%} };

const char* {{namespace.upper()}}_NUCS[{{N}}] =
  { {%- for nuc in nucs %}"{{nuc}}",{% endfor -%} };

{{namespace}}_transmute_info_t {{namespace}}_transmute_info = {
  .n = {{N}},
  .nnz = {{NNZ}},
  .i = (int*) {{namespace.upper()}}_I,
  .j = (int*) {{namespace.upper()}}_J,
  .nucs = (char**) {{namespace.upper()}}_NUCS,
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
    rat_func = get_CRAM_from_cache(degree, prec, log=True, plot=False, use_cache=use_cache)

    thetas, alphas, alpha0 = thetas_alphas(rat_func, prec)
    return thetas, alphas, alpha0

def generate(json_file=os.path.join(os.path.dirname(__file__), 'data/gensolve.json'),
    outfile=None, degrees=None, py_solve=False, namespace='transmutagen'):

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
    ijkeys = [(nucs.index(j), nucs.index(i)) for i, j in json_data['tofrom']]
    ij = {k: l for l, k in enumerate(sorted(ijkeys))}
    ijk = make_ijk(ij, N)
    diagonals = {ij[i, i]: i for i in range(N)}
    more_than_fore = [len([j for j in range(i+1) if (i, j) in ijk]) > 1 for i in range(N)]
    more_than_back = [len([j for j in range(i, N) if (i, j) in ijk]) > 1 for i in range(N)]
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
        __version__=__version__, sys=sys)
    header_template = env.from_string(HEADER, globals=globals())
    header = header_template.render(types=types, degrees=degrees,
        py_solve=py_solve, namespace=namespace)
    print("Writing", outfile)
    with open(outfile, 'w') as f:
        f.write(src)
    print("Writing", headerfile)
    with open(headerfile, 'w') as f:
        f.write(header)
    # If this changes, also update py_solve/setup.py
    compiler_flags = ['-O0', '-fcx-fortran-rules', '-fcx-limited-range',
        '-ftree-sra', '-ftree-ter', '-fexpensive-optimizations']

    print("With gcc, it is recommended to compile the following flags:", ' '.join(compiler_flags))

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

    ns = p.parse_args(args=args)
    if ns.outfile and not ns.outfile.endswith('.c'):
        p.error("--outfile should end with '.c'")
    arguments = ns.__dict__.copy()

    generate(**arguments)


if __name__ == "__main__":
    main()
