/*
 *  dgemm.c: C = alpha*A*B + beta*C with the CBLAS interface of ulmBLAS.
 *
 *  Build (in the top-level directory of ulmBLAS):
 *
 *      make
 *      cc -I. -c examples/dgemm.c -o dgemm.o
 *      c++ -o dgemm dgemm.o libcblas.a        # libcblas.a needs the C++ runtime
 *      ./dgemm
 */
#include <stdio.h>
#include <cblas.h>

int
main(void)
{
    /*  column major: A is 2 x 3, B is 3 x 2, C is 2 x 2  */
    double A[] = { 1, 4,    2, 5,    3, 6 };
    double B[] = { 1, 0, 2,    0, 1, 1 };
    double C[] = { 0, 0,    0, 0 };

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                2, 2, 3,                /* m, n, k */
                1.0, A, 2,              /* alpha, A, lda */
                B, 3,                   /* B, ldb */
                0.0, C, 2);             /* beta, C, ldc */

    /*  expected: C = [ 7 5 ; 16 11 ]  */
    printf("C = [ %g %g ; %g %g ]\n", C[0], C[2], C[1], C[3]);
    return (C[0]==7 && C[1]==16 && C[2]==5 && C[3]==11) ? 0 : 1;
}
