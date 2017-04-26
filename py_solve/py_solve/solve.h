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
void transmutagen_scalar_times_vector_double(double, double*);

void transmutagen_solve_complex(double complex* A, double complex* b, double complex* x);
void transmutagen_diag_add_complex(double complex* A, double complex alpha);
void transmutagen_dot_complex(double complex* A, double complex* x, double complex* y);
void transmutagen_scalar_times_vector_complex(double complex, double complex*);

void transmutagen_solve_special(double* A, double complex theta, double complex alpha, double* b, double complex* x);
void expm_multiply6(double* A, double* b, double* x);
void expm_multiply8(double* A, double* b, double* x);
void expm_multiply10(double* A, double* b, double* x);
void expm_multiply12(double* A, double* b, double* x);
void expm_multiply14(double* A, double* b, double* x);
void expm_multiply16(double* A, double* b, double* x);
void expm_multiply18(double* A, double* b, double* x);
#endif
