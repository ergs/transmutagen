#ifndef TRANSMUTAGEN_SOLVE_C
#define TRANSMUTAGEN_SOLVE_C
typedef struct transmutagen_info_tag {
  int n;
  int nnz;
  int* i;
  int* j;
  char** nucs;
} transmutagen_info_t;

extern transmutagen_info_t transmutagen_info;

void transmutagen_solve_double(double* A, double* b, double* x);
#endif