/*
 *  bench2: single-threaded cblas_dgemm timing (column major, NoTrans).
 *
 *      bench2 SAMPLES BETA M N K [M N K ...]
 *
 *  For each size: one call checked against a naive product (skipped above
 *  2e8 multiply-adds), warm-up of at least 0.3 s, then SAMPLES samples of
 *  at least MIN_SAMPLE seconds each.  Output per size:
 *
 *      m n k beta median_GFLOPS min max calls_per_sample check
 *
 *  With BETA = 0, C is filled with NaN before each sample (C need not be
 *  set).  Writes nothing but standard output.  On macOS the thread runs
 *  with QoS class user-interactive (performance cores).
 */
#if defined(__APPLE__)
#define _DARWIN_C_SOURCE        /* before cblas.h, which may set _POSIX_C_SOURCE */
#include <pthread/qos.h>
#endif
#include <cblas.h>
#include <math.h>
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

static void
fill(double *x, long n, int nanValues)
{
    for (long i=0; i<n; ++i) {
        x[i] = nanValues ? NAN : (double)rand()/RAND_MAX - 0.5;
    }
}

static const char *
check(int m, int n, int k, const double *A, const double *B, double beta,
      const double *C0, const double *C)
{
    if ((double)m*n*k > 2e8) {
        return "unchecked";
    }
    for (int j=0; j<n; ++j) {
        for (int i=0; i<m; ++i) {
            double s = 0, bound = 0;
            for (int l=0; l<k; ++l) {
                s     += A[i+(long)l*m]*B[l+(long)j*k];
                bound += fabs(A[i+(long)l*m]*B[l+(long)j*k]);
            }
            if (beta!=0) {
                s     += beta*C0[i+(long)j*m];
                bound += fabs(beta*C0[i+(long)j*m]);
            }
            if (!(fabs(C[i+(long)j*m]-s) <= 4*(k+2)*2.3e-16*bound+1e-300)) {
                return "WRONG";
            }
        }
    }
    return "ok";
}

static void
run(int samples, double beta, int m, int n, int k)
{
    double *A  = malloc(sizeof(double)*m*k);
    double *B  = malloc(sizeof(double)*k*n);
    double *C  = malloc(sizeof(double)*m*n);
    double *C0 = malloc(sizeof(double)*m*n);
    double t[MAX_SAMPLES], flops = 2.0*m*n*k;
    long   calls = 0;

    fill(A, (long)m*k, 0);
    fill(B, (long)k*n, 0);
    fill(C0, (long)m*n, beta==0);
    memcpy(C, C0, sizeof(double)*m*n);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k,
                1.0, A, m, B, k, beta, C, m);
    const char *ok = check(m, n, k, A, B, beta, C0, C);

    double t0 = now(), t1;
    do {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k,
                    1.0, A, m, B, k, beta, C, m);
        ++calls;
        t1 = now();
    } while (t1-t0 < 0.3);
    calls = (long)(calls*MIN_SAMPLE/(t1-t0)) + 1;

    for (int s=0; s<samples; ++s) {
        if (beta==0) {
            fill(C, (long)m*n, 1);
        }
        t0 = now();
        for (long c=0; c<calls; ++c) {
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k,
                        1.0, A, m, B, k, beta, C, m);
        }
        t[s] = flops*calls/(now()-t0)*1e-9;
    }
    qsort(t, samples, sizeof(double), cmp);
    printf("%5d %5d %5d %g %8.3f %8.3f %8.3f %6ld %s\n", m, n, k, beta,
           t[samples/2], t[0], t[samples-1], calls, ok);
    fflush(stdout);
    free(A); free(B); free(C); free(C0);
}

int
main(int argc, char **argv)
{
#if defined(__APPLE__)
    /*  macOS: run on a performance core (there is no CPU affinity)  */
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
    if (argc<6 || (argc-3)%3) {
        fprintf(stderr, "usage: %s SAMPLES BETA M N K [M N K ...]\n", argv[0]);
        return 2;
    }
    int    samples = atoi(argv[1]);
    double beta    = atof(argv[2]);
    if (samples<1 || samples>MAX_SAMPLES) {
        return 2;
    }
    for (int i=3; i+2<argc; i+=3) {
        run(samples, beta, atoi(argv[i]), atoi(argv[i+1]), atoi(argv[i+2]));
    }
    return 0;
}
