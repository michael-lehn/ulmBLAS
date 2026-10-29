/*
 *  verify-large: correctness of cblas_dgemm (column major, NoTrans, NoTrans,
 *  alpha = 1, beta = 1, lda = m, as bench2) for sizes where bench2 does not
 *  check the result (more than 2e8 multiply-adds).  For each n, C = A*B + C
 *  with random A, B, C in [-1, 1]; ENTRIES randomly chosen elements of C are
 *  compared with the sum computed in long double, against the bound
 *  4*(k+2)*eps*(sum |A(i,l)*B(l,j)| + |C0(i,j)|).
 *
 *      verify-large n1 n2 ...
 *
 *  Prints one line per n (max error / bound, number of violations).  Exit
 *  status 1 if any element violates the bound.
 */
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>

enum { ENTRIES = 2000 };

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
        int     n = atoi(argv[a]);
        size_t  nn = (size_t)n*n;
        double *A = malloc(nn*sizeof(double)), *B = malloc(nn*sizeof(double));
        double *C = malloc(nn*sizeof(double)), *C0 = malloc(nn*sizeof(double));
        for (size_t i=0; i<nn; ++i) {
            A[i] = uniform(); B[i] = uniform(); C[i] = C0[i] = uniform();
        }
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0,
                    A, n, B, n, 1.0, C, n);
        double worst = 0;
        int    violations = 0;
        for (int e=0; e<ENTRIES; ++e) {
            int i = rand() % n, j = rand() % n;
            long double s = C0[i+(size_t)j*n], bound = fabs(C0[i+(size_t)j*n]);
            for (int l=0; l<n; ++l) {
                long double p = (long double)A[i+(size_t)l*n]*B[l+(size_t)j*n];
                s += p;
                bound += fabsl(p);
            }
            bound *= 4.0L*(n+2)*DBL_EPSILON;
            long double err = fabsl(C[i+(size_t)j*n] - s);
            if (!(err <= bound)) {
                ++violations;
            }
            if (err/bound > worst) {
                worst = (double)(err/bound);
            }
        }
        printf("n %4d: %d elements checked, max error/bound %.3g, %d violations\n",
               n, ENTRIES, worst, violations);
        bad |= violations>0;
        free(A); free(B); free(C); free(C0);
    }
    return bad;
}
