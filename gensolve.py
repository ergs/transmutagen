import os
import sys
from argparse import ArgumentParser

from jinja2 import Environment
from scipy.sparse import eye

sys.path.insert(0, os.path.dirname(__file__))

from transmutagen.tape9utils import tape9_to_sparse


SRC = """/* unrolled solvers */
#include "solve.h"

const int TRANSMUTAGEN_I[{{NNZ}}] =
  { {%- for i, j in sorted(ij) %}{{i}},{% endfor -%} };

const int TRANSMUTAGEN_J[{{NNZ}}] =
  { {%- for i, j in sorted(ij) %}{{j}},{% endfor -%} };

const char* TRANSMUTAGEN_NUCS[{{NNZ}}] =
  { {%- for nuc in nucs %}"{{nuc}}",{% endfor -%} };


transmutagen_info_t transmutagen_info = {
  .n = {{N}},
  .nnz = {{NNZ}},
  .i = (int*) TRANSMUTAGEN_I,
  .j = (int*) TRANSMUTAGEN_J,
  .nucs = (char**) TRANSMUTAGEN_NUCS,
};


void transmutagen_solve_double(double* A, double* b, double* x) {
  /* Forward calc */
  {%- for i in range(N) %}
  x[{{i}}] = b[{{i}}]{% for j in range(i) %}{%if (i, j) in ij%} - A[{{ij[i, j]}}]*x[{{j}}]{%endif%}{% endfor %};
  {%- endfor %}
  /* Backward calc */
  {% for i in range(N-1, -1, -1) %}{%if more_than_back[i]%}
  x[{{i}}] = x[{{i}}]{% for j in range(i+1, N) %}{%if (i, j) in ij%} - A[{{ij[i, j]}}]*x[{{j}}]{%endif%}{% endfor %};
  {%-endif-%}{%- endfor %}
  /* divide by diag */
  {%- for i in range(N) %}
  x[{{i}}] /= A[{{ij[i, i]}}];
  {%- endfor %}
}
"""


def csr_ij(mat):
    ij = {}
    i = j = 0
    for i, l, u in zip(range(mat.shape[0]), mat.indptr[:-1], mat.indptr[1:]):
        for p in range(l, u):
            ij[i, mat.indices[p]] = p
    return ij


def generate(tape9, decaylib, outfile='transmutagen/solve.c'):
    mat, nucs = tape9_to_sparse(tape9, phi=1.0, format='csr', decaylib=decaylib)
    N = mat.shape[0]
    mat = mat + eye(N, format='csr')
    ij = csr_ij(mat)
    more_than_fore = [len([j for j in range(i+1) if (i, j) in ij]) > 1 for i in range(N)]
    more_than_back = [len([j for j in range(i, N) if (i, j) in ij]) > 1 for i in range(N)]
    env = Environment()
    template = env.from_string(SRC, globals=globals())
    src = template.render(N=mat.shape[0], ij=ij, nucs=nucs, sorted=sorted, len=len,
                          more_than_back=more_than_back, NNZ=len(ij))
    with open(outfile, 'w') as f:
        f.write(src)


def main(args=None):
    p = ArgumentParser('gensolver')
    p.add_argument('tape9', help="path to the TAPE9 file.")
    p.add_argument('-d', '--decay', help='path to the decay file, if needed',
                   default='decay.lib', dest='decaylib')

    ns = p.parse_args(args=args)
    generate(ns.tape9, ns.decaylib)


if __name__ == "__main__":
    main()
