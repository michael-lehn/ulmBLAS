/*
 *  mkl-info: oneMKL version and threading; run with MKL_VERBOSE=1 to see
 *  the instruction set of the code path that MKL dispatches to.
 */
#include <stdio.h>
#include <mkl.h>

int
main(void)
{
    char v[256];
    double A[64*64], B[64*64], C[64*64];
    mkl_get_version_string(v, sizeof v);
    printf("version %s\nmax_threads %d\n", v, mkl_get_max_threads());
    for (int i = 0; i < 64*64; ++i) A[i] = B[i] = C[i] = 1.0;
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 64, 64, 64, 1.0,
                A, 64, B, 64, 1.0, C, 64);
    printf("C[0] %g (expected 65)\n", C[0]);
    return 0;
}
