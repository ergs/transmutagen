from jinja2 import Environment

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

