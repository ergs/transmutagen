#ifndef TRANSMUTAGEN_SOLVE_C
#define TRANSMUTAGEN_SOLVE_C

#include <complex.h>

typedef struct transmutagen_info_tag {
  int n;
  int nnz;
  int* i;
  int* j;
  char** nucs;
} transmutagen_info_t;

extern transmutagen_info_t transmutagen_info;

void transmutagen_solve_double(double* A, double* b, double* x);
void transmutagen_diag_add_double(double* A, double alpha);
void transmutagen_dot_double(double* A, double* x, double* y);

void transmutagen_solve_complex(double complex* A, double complex* b, double complex* x);
void transmutagen_diag_add_complex(double complex* A, double complex alpha);
void transmutagen_dot_complex(double complex* A, double complex* x, double complex* y);
#endif