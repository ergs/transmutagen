from argparse import ArgumentParser

from jinja2 import Environment
from scipy.sparse import eye, csr_matrix
from sympy import im
import numpy as np

from transmutagen.tape9utils import tape9_to_sparse, normalize_tape9s

from .cram import get_CRAM_from_cache, CRAM_exp
from .partialfrac import thetas_alphas

SRC = """/* unrolled solvers */
#include <string.h>

#include "solve.h"

const int TRANSMUTAGEN_I[{{NNZ}}] =
  { {%- for i, j in sorted(ij) %}{{i}},{% endfor -%} };

const int TRANSMUTAGEN_J[{{NNZ}}] =
  { {%- for i, j in sorted(ij) %}{{j}},{% endfor -%} };

const char* TRANSMUTAGEN_NUCS[{{N}}] =
  { {%- for nuc in nucs %}"{{nuc}}",{% endfor -%} };


transmutagen_info_t transmutagen_info = {
  .n = {{N}},
  .nnz = {{NNZ}},
  .i = (int*) TRANSMUTAGEN_I,
  .j = (int*) TRANSMUTAGEN_J,
  .nucs = (char**) TRANSMUTAGEN_NUCS,
};

int transmutagen_ij(int i, int j) {
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

{%- for type, typefuncname in types %}
void transmutagen_solve_{{typefuncname}}({{type}}* A, {{type}}* b, {{type}}* x) {
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

void transmutagen_diag_add_{{typefuncname}}({{type}}* A, {{type}} theta) {
  /* In-place, performs the addition A + theta I, for a scalar theta. */
  {%- for i in range(N) %}
  A[{{ij[i, i]}}] += theta;
  {%- endfor %}
}

void transmutagen_dot_{{typefuncname}}({{type}}* A, {{type}}* x, {{type}}* y) {
  /* Performs the caclulation Ax = y and returns y */
  {%- for i in range(N) %}
  y[{{i}}] ={% for j in range(N) %}{% if (i,j) in ij %} + A[{{ij[i, j]}}]*x[{{j}}]{% endif %}{% endfor %};
  {%- endfor %}
}

void transmutagen_scalar_times_vector_{{typefuncname}}({{type}} alpha, {{type}}* v) {
  /* In-place, performs alpha*v, for a scalar alpha and vector v. */
  {%- for i in range(N) %}
  v[{{i}}] *= alpha;
  {%- endfor %}
}

{%- endfor %}

void transmutagen_solve_special(double* A, double complex theta, double complex alpha, double* b, double complex* x) {
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
void expm{{degree}}(double* A, double* b, double* x) {
    {%- for i in range(degree//2) %}
    double complex x{{i}} [{{N}}];
    {%- endfor %}

    {% set thetas, alphas, alpha0 = get_thetas_alphas(degree) -%}
    {% for theta, alpha in zip(thetas, alphas) if im(theta) >= 0 %}
    transmutagen_solve_special(A, {{ -theta}}, {{2*alpha}}, b, x{{loop.index0}});
    {%- endfor %}

    {% for i in range(N) %}
    x[{{i}}] = (double)creal({%- for j in range(degree//2) %}+x{{j}}[{{i}}]{%- endfor %}) + {{alpha0}}*b[{{i}}];
    {%- endfor %}
}

{% endfor %}
"""


def csr_ij(mat):
    ij = {}
    for i, l, u in zip(range(mat.shape[0]), mat.indptr[:-1], mat.indptr[1:]):
        for p in range(l, u):
            ij[i, mat.indices[p]] = p
    return ij


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
    if use_cache:
        rat_func = get_CRAM_from_cache(degree, prec)
    else:
        rat_func = CRAM_exp(degree, prec, plot=False)

    thetas, alphas, alpha0 = thetas_alphas(rat_func, prec)
    return thetas, alphas, alpha0

def common_mat(mats):
    assert len({i.shape for i in mats}) == 1
    mats = [i.tocoo() for i in mats] + [eye(mats[0].shape[0], format='coo')]
    rows = np.hstack([i.row for i in mats])
    cols = np.hstack([i.col for i in mats])
    data = np.ones(len(rows))
    return csr_matrix((data, (rows, cols)))

def generate(tape9s, decaylib, outfile='py_solve/py_solve/solve.c'):
    mats, nucs = tape9_to_sparse(tape9s, phi=1.0, format='csr', decaylib=decaylib)
    mat = common_mat(mats)
    N = mat.shape[0]
    ij = csr_ij(mat)
    ijk = make_ijk(ij, N)
    diagonals = {ij[i, i]: i for i in range(N)}
    more_than_fore = [len([j for j in range(i+1) if (i, j) in ijk]) > 1 for i in range(N)]
    more_than_back = [len([j for j in range(i, N) if (i, j) in ijk]) > 1 for i in range(N)]
    types = [  # C type, type function name
             ('double', 'double'),
             ('double complex', 'complex')]
    env = Environment()
    template = env.from_string(SRC, globals=globals())
    src = template.render(N=mat.shape[0], ij=ij, ijk=ijk, nucs=nucs, sorted=sorted, len=len,
                          more_than_back=more_than_back, NNZ=len(ij), NIJK=len(ijk),
                          more_than_fore=more_than_fore, types=types,
                          diagonals=diagonals, degrees=[6, 8, 10, 12, 14, 16, 18],
                          get_thetas_alphas=get_thetas_alphas, im=im, zip=zip, enumerate=enumerate)
    print("Writing", outfile)
    with open(outfile, 'w') as f:
        f.write(src)


def main(args=None):
    p = ArgumentParser('gensolver')
    p.add_argument('tape9s', nargs='+', help="""Paths to the TAPE9 files. If a
    path is a directory, a set of default libraries will be gathered from that
    directory (transmutagen.origen_all.ALL_LIBS)""")
    p.add_argument('-d', '--decay', help='path to the decay file, if needed',
                   default='decay.lib', dest='decaylib')

    ns = p.parse_args(args=args)
    tape9s = normalize_tape9s(ns.tape9s)
    generate(tape9s, ns.decaylib)


if __name__ == "__main__":
    main()
