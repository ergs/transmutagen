cdef extern from "solve.h":

    cdef struct transmutagen_info_tag:
        int n
        int nnz
        int* i
        int* j
        char** nucs

    ctypedef transmutagen_info_tag transmutagen_info_t
    cdef transmutagen_info_t transmutagen_info

    void transmutagen_solve_double(double*, double*, double*)
    void transmutagen_diag_add_double(double*, double)
    void transmutagen_dot_double(double*, double*, double*)
    void transmutagen_vector_add_7_double(double*, double*, double*, double*, double*, double*, double*, double*)
    void transmutagen_scalar_times_vector_double(double, double*)

    void transmutagen_solve_complex(double complex*, double complex*, double complex*)
    void transmutagen_diag_add_complex(double complex*, double complex)
    void transmutagen_dot_complex(double complex*, double complex*, double complex*)
    void transmutagen_vector_add_7_complex(double complex*, double complex*, double complex*, double complex*, double complex*, double complex*, double complex*, double complex*)
    void transmutagen_scalar_times_vector_complex(double complex, double complex*)

    void expm14(double complex*, double complex*, double complex*)
