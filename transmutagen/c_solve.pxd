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
