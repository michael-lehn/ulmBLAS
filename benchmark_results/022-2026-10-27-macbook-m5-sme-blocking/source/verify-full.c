/*
 *  verify-full: correctness of cblas_dgemm (column major, NoTrans, NoTrans,
 *  alpha = 1, beta = 1, leading dimensions m, m, k, as bench2) for shapes
 *  where bench2 does not check the result.  For each shape, C = A*B + C
 *  with random A, B, C in [-1, 1]; ALL elements of C are compared with a
 *  naive product computed in double, against the bound
 *  4*(k+2)*eps*(sum_l |A(i,l)*B(l,j)| + |C0(i,j)|) (covers the rounding
 *  errors of both computations).
 *
 *      verify-full M_N_K [M_N_K ...]
 *
 *  Prints one line per shape (max error / bound, number of violations).
 *  Exit status 1 if any element violates the bound.
 */
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>

static double
uniform(void)
{
    return 2.0*rand()/RAND_MAX - 1.0;
}

int
main(int argc, char **argv)
{
    int bad = 0;
    srand(2026);
    for (int a=1; a<argc; ++a) {
        int m, n, k;
        if (sscanf(argv[a], "%d_%d_%d", &m, &n, &k)!=3) {
            fprintf(stderr, "bad shape %s\n", argv[a]);
            return 2;
        }
        size_t  mk = (size_t)m*k, kn = (size_t)k*n, mn = (size_t)m*n;
        double *A = malloc(mk*sizeof(double)), *B = malloc(kn*sizeof(double));
        double *C = malloc(mn*sizeof(double)), *R = malloc(mn*sizeof(double));
        double *E = malloc(mn*sizeof(double));
        for (size_t i=0; i<mk; ++i) A[i] = uniform();
        for (size_t i=0; i<kn; ++i) B[i] = uniform();
        for (size_t i=0; i<mn; ++i) { C[i] = R[i] = uniform(); E[i] = fabs(R[i]); }
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0,
                    A, m, B, k, 1.0, C, m);
        for (int j=0; j<n; ++j) {
            for (int l=0; l<k; ++l) {
                double b = B[l+(size_t)j*k], ab = fabs(b);
                for (int i=0; i<m; ++i) {
                    R[i+(size_t)j*m] += A[i+(size_t)l*m]*b;
                    E[i+(size_t)j*m] += fabs(A[i+(size_t)l*m])*ab;
                }
            }
        }
        double worst = 0;
        long   violations = 0;
        for (size_t i=0; i<mn; ++i) {
            double bound = 4.0*(k+2)*DBL_EPSILON*E[i];
            double err = fabs(C[i]-R[i]);
            if (!(err <= bound)) {
                ++violations;
            }
            if (bound>0 && err/bound > worst) {
                worst = err/bound;
            }
        }
        printf("%d %d %d: %zu elements checked, max error/bound %.3g, %ld violations\n",
               m, n, k, mn, worst, violations);
        bad |= violations>0;
        free(A); free(B); free(C); free(R); free(E);
    }
    return bad;
}
