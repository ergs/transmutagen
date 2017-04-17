from argparse import ArgumentParser

from jinja2 import Environment
from scipy.sparse import eye

from transmutagen.tape9utils import tape9_to_sparse


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

void expm14(double* A, double* b, double* x) {
    {%- for i in range(14//2) %}
    double complex x{{i}} [{{N}}];
    {%- endfor %}

    transmutagen_solve_special(A, 5.62314257274597712494520326004 + (-1.19406904634396697320055795941)*I, 27.8751619401456463960237466985 + (-102.147339990564514248579577671)*I, b, x0);
    transmutagen_solve_special(A, 2.26978382923111270000297467973 + (-8.46173797304022139646316695686)*I, -4.80711209883250887291965497626 + (-1.32097938374287242475211680928)*I, b, x1);
    transmutagen_solve_special(A, 3.99336971057856853025375498429 + (-6.00483164223503731596717948806)*I, 23.4982320910827012314190795608 + (-5.80835912971420750092857584133)*I, b, x2);
    transmutagen_solve_special(A, 5.08934506058062450150096345613 + (-3.58882402902700651552894109753)*I, -46.9332744888312930359089080827 + 45.6436497688277607413919781939*I, b, x3);
    transmutagen_solve_special(A, -0.20875863825013012197592074913 + (-10.9912605619012609176212156940)*I, 0.376360038782269688578990952196 + 0.335183470294501039620923752439*I, b, x4);
    transmutagen_solve_special(A, -8.89777318646888881871224673753 + (-16.6309826199020853044092653271)*I, 0.0000715428806358906730643236773918 + 0.000143610433495413001443873463755*I, b, x5);
    transmutagen_solve_special(A, -3.70327504942344806084144231316 + (-13.6563718714832681701880222932)*I, -0.00943902531073616885305862658337 + (-0.0171847919584830175365187052932)*I, b, x6);

    {%- for i in range(N) %}
    x[{{i}}] = (double)2*creal({%- for j in range(14//2) %}+x{{j}}[{{i}}]{%- endfor %}) + 1.83217437825404121359416895790e-14*b[{{i}}];
    {%- endfor %}
}

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


def generate(tape9, decaylib, outfile='py_solve/py_solve/solve.c'):
    mat, nucs = tape9_to_sparse(tape9, phi=1.0, format='csr', decaylib=decaylib)
    N = mat.shape[0]
    mat = mat + eye(N, format='csr')
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
                          more_than_fore=more_than_fore, types=types, diagonals=diagonals)
    print("Writing", outfile)
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
