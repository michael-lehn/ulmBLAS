/*
 *  bench3: single-threaded level 3 timing besides gemm (column major).
 *
 *      bench3 SAMPLES N [N ...]
 *
 *  For each N (all matrices N x N): dsyrk (lower, NoTrans), dsymm (left,
 *  lower), dtrmm and dtrsm (left, lower, NoTrans, non-unit; A diagonally
 *  dominant).  Output per routine and size:
 *
 *      routine n median_GFLOPS min max calls_per_sample
 *
 *  GFLOP/s with the usual flop counts: syrk n^3, symm 2 n^3, trmm and
 *  trsm n^3.  Writes nothing but standard output.  On macOS the thread runs
 *  with QoS class user-interactive (performance cores).
 */
#if defined(__APPLE__)
#define _DARWIN_C_SOURCE        /* before cblas.h, which may set _POSIX_C_SOURCE */
#include <pthread/qos.h>
#endif
#include <cblas.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_SAMPLES 64
#define MIN_SAMPLE  0.2

static double
now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9*t.tv_nsec;
}

static int
cmp(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return x<y ? -1 : x>y;
}

static double *A, *B, *C, *B0;

static void
call(int r, int n)
{
    switch (r) {
    case 0:
        cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans, n, n,
                    1.0, A, n, 1.0, C, n);
        break;
    case 1:
        cblas_dsymm(CblasColMajor, CblasLeft, CblasLower, n, n,
                    1.0, A, n, B, n, 1.0, C, n);
        break;
    case 2:
        memcpy(B, B0, sizeof(double)*n*n);
        cblas_dtrmm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans,
                    CblasNonUnit, n, n, 1.0, A, n, B, n);
        break;
    case 3:
        memcpy(B, B0, sizeof(double)*n*n);
        cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans,
                    CblasNonUnit, n, n, 1.0, A, n, B, n);
        break;
    }
}

int
main(int argc, char **argv)
{
#if defined(__APPLE__)
    /*  macOS: run on a performance core (there is no CPU affinity)  */
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
    static const char  *name[]  = { "dsyrk", "dsymm", "dtrmm", "dtrsm" };
    static const double factor[] = { 1, 2, 1, 1 };

    if (argc<3) {
        fprintf(stderr, "usage: %s SAMPLES N [N ...]\n", argv[0]);
        return 2;
    }
    int samples = atoi(argv[1]);
    if (samples<1 || samples>MAX_SAMPLES) {
        return 2;
    }
    for (int a=2; a<argc; ++a) {
        int  n = atoi(argv[a]);
        long nn = (long)n*n;
        A  = malloc(sizeof(double)*nn);
        B  = malloc(sizeof(double)*nn);
        B0 = malloc(sizeof(double)*nn);
        C  = malloc(sizeof(double)*nn);
        for (long i=0; i<nn; ++i) {
            A[i]  = (double)rand()/RAND_MAX - 0.5;
            B0[i] = (double)rand()/RAND_MAX - 0.5;
            C[i]  = (double)rand()/RAND_MAX - 0.5;
        }
        for (int i=0; i<n; ++i) {
            A[i+(long)i*n] += n;
        }
        memcpy(B, B0, sizeof(double)*nn);

        for (int r=0; r<4; ++r) {
            double t[MAX_SAMPLES], flops = factor[r]*n*(double)n*n;
            long   calls = 0;
            double t0 = now(), t1;
            do {
                call(r, n);
                ++calls;
                t1 = now();
            } while (t1-t0 < 0.3);
            calls = (long)(calls*MIN_SAMPLE/(t1-t0)) + 1;
            for (int s=0; s<samples; ++s) {
                t0 = now();
                for (long c=0; c<calls; ++c) {
                    call(r, n);
                }
                t[s] = flops*calls/(now()-t0)*1e-9;
            }
            qsort(t, samples, sizeof(double), cmp);
            printf("%s %5d %8.3f %8.3f %8.3f %6ld\n", name[r], n,
                   t[samples/2], t[0], t[samples-1], calls);
            fflush(stdout);
        }
        free(A); free(B); free(B0); free(C);
    }
    return 0;
}
