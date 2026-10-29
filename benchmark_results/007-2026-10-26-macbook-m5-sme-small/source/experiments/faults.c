/*
 *  faults: page faults (minor, getrusage) and time per cblas_dgemm call
 *  (column major, NoTrans, alpha = beta = 1), after a warm-up.
 *
 *      faults n [calls]
 */
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <time.h>
#include <cblas.h>

static long
minflt(void)
{
    struct rusage u;
    getrusage(RUSAGE_SELF, &u);
    return u.ru_minflt;
}

static double
now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return t.tv_sec + 1e-9*t.tv_nsec;
}

int
main(int argc, char **argv)
{
    int  n     = argc>1 ? atoi(argv[1]) : 64;
    long calls = argc>2 ? atol(argv[2]) : 20000;
    double *A = malloc(sizeof(double)*n*n), *B = malloc(sizeof(double)*n*n),
           *C = malloc(sizeof(double)*n*n);
    for (long i=0; i<(long)n*n; ++i) { A[i] = 1e-3*i; B[i] = 1; C[i] = 0; }
    for (int w=0; w<100; ++w) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0,
                    A, n, B, n, 1.0, C, n);
    }
    long   f0 = minflt();
    double t0 = now();
    for (long c=0; c<calls; ++c) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0,
                    A, n, B, n, 1.0, C, n);
    }
    double t = now() - t0;
    printf("n %4d: %.2f page faults per call, %.3f us per call\n", n,
           (double)(minflt() - f0)/calls, 1e6*t/calls);
    return C[0]==-1;
}
