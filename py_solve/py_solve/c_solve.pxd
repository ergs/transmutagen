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
    void transmutagen_scalar_times_vector_double(double, double*)

    void transmutagen_solve_complex(double complex*, double complex*, double complex*)
    void transmutagen_diag_add_complex(double complex*, double complex)
    void transmutagen_dot_complex(double complex*, double complex*, double complex*)
    void transmutagen_scalar_times_vector_complex(double complex, double complex*)

    void expm_multiply6(double*, double*, double*)
    void expm_multiply8(double*, double*, double*)
    void expm_multiply10(double*, double*, double*)
    void expm_multiply12(double*, double*, double*)
    void expm_multiply14(double*, double*, double*)
    void expm_multiply16(double*, double*, double*)
    void expm_multiply18(double*, double*, double*)
