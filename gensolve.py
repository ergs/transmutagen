import os
import sys
from argparse import ArgumentParser

from jinja2 import Environment

sys.path.insert(0, os.path.dirname(__file__))

from transmutagen.tape9utils import tape9_to_sparse


SRC = """
void solve_double(double* A, double* b, double* x) {
  {% for i in range(N) %}/* Forward calc */
  x[{{i}}] = b[{{i}}] {% for j in range(i) %} - A[{{ij[i, j]}}] * x[{{j}}] {% endfor %};
  {% endfor %}
  {% for i in range(N-1, -1, -1) %}/* Backward calc */
  x[{{i}}] = x[{{i}}] {% for j in range(i+1, N) %} - A[{{ij[i, j]}}] * x[{{j}}] {% endfor %};
  {% endfor %}
  {% for i in range(N) %}/* divide by diag */
  x[{{i}}] /= A[{{ij[i, i]}}];
  {% endfor %}
}
"""


def csr_ij(mat):
    ij = {}
    i = j = 0
    for i, l, u in zip(range(mat.shape[0]), mat.indptr[:-1], mat.indptr[1:]):
        for p in range(l, u):
            ij[i, mat.indices[p]] = p
    return ij


def generate(tape9, decaylib):
    mat, nucs = tape9_to_sparse(tape9, phi=1.0, format='csr', decaylib=decaylib)
    ij = csr_ij(mat)
    env = Environment()
    template = env.from_string(SRC, globals=globals())
    src = template.render(N=mat.shape[0], ij=ij)
    print(src)


def main(args=None):
    p = ArgumentParser('gensolver')
    p.add_argument('tape9', help="path to the TAPE9 file.")
    p.add_argument('-d', '--decay', help='path to the decay file, if needed',
                   default='decay.lib', dest='decaylib')

    ns = p.parse_args(args=args)
    generate(ns.tape9, ns.decaylib)


if __name__ == "__main__":
    main()
