#ifndef TRANSMUTAGEN_SOLVE_C
#define TRANSMUTAGEN_SOLVE_C
struct matrix_info_t {
  int n;
  int nnz;
  int* i;
  int* j;
  char** nucs;
};

void transmutagen_solve_double(double* A, double* b, double* x);
#endif