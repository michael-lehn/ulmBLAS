/*
 *  ukernel-blis: throughput of the BLIS SME2 micro kernel for double
 *  (bli_dgemm_armsme_int_4SVLx2SVL of BLIS pull request #921, MR = 32,
 *  NR = 16), measured as ukernel.cc measures the ulmBLAS micro kernel.
 *
 *      ukernel-blis ALPHA BETA KC [KC ...]
 *
 *  NP x NQ = 8 x 4 pairs of packed panels of A (MR x KC) and B (KC x NR),
 *  B panel outer, A panels inner; C tiles with unit row stride (rs_c = 1,
 *  cs_c = MR).  The kernel is called with the full tile (m = MR, n = NR),
 *  so it does not use its temporary tile; it uses neither the auxinfo nor
 *  the context (NULL).  Output per KC:
 *
 *      kc MR NR median_GFLOPS min max
 */
#define _DARWIN_C_SOURCE
#include <pthread/qos.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void bli_dgemm_armsme_int_4SVLx2SVL(long m, long n, long k,
                                    const void *alpha, const void *a,
                                    const void *b, const void *beta,
                                    void *c, long rs_c, long cs_c,
                                    const void *data, const void *cntx);

static double
now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9*t.tv_nsec;
}

static int
cmp(const void *x, const void *y)
{
    double u = *(const double *)x, v = *(const double *)y;
    return u<v ? -1 : u>v;
}

int
main(int argc, char **argv)
{
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    const long MR = 32, NR = 16, NP = 8, NQ = 4;
    double alpha = atof(argv[1]), beta = atof(argv[2]), sum = 0;

    for (int i=3; i<argc; ++i) {
        long   kc = atol(argv[i]);
        double *A = aligned_alloc(128, 8*(NP*MR*kc+MR));
        double *B = aligned_alloc(128, 8*(NQ*NR*kc+NR));
        double *C = aligned_alloc(128, 8*(MR*NR*NP*NQ));
        for (long j=0; j<NP*MR*kc+MR; ++j) A[j] = (double)rand()/RAND_MAX-0.5;
        for (long j=0; j<NQ*NR*kc+NR; ++j) B[j] = (double)rand()/RAND_MAX-0.5;
        for (long j=0; j<MR*NR*NP*NQ; ++j) C[j] = 0;

        long   reps = 0;
        double t0 = now(), t1, g[7];
        do {
            for (long q=0; q<NQ; ++q)
                for (long p=0; p<NP; ++p)
                    bli_dgemm_armsme_int_4SVLx2SVL(MR, NR, kc, &alpha,
                        &A[p*MR*kc], &B[q*NR*kc], &beta,
                        &C[(q*NP+p)*MR*NR], 1, MR, NULL, NULL);
            ++reps; t1 = now();
        } while (t1-t0 < 0.3);
        reps = (long)(reps*0.2/(t1-t0)) + 1;
        for (int s=0; s<7; ++s) {
            t0 = now();
            for (long r=0; r<reps; ++r)
                for (long q=0; q<NQ; ++q)
                    for (long p=0; p<NP; ++p)
                        bli_dgemm_armsme_int_4SVLx2SVL(MR, NR, kc, &alpha,
                            &A[p*MR*kc], &B[q*NR*kc], &beta,
                            &C[(q*NP+p)*MR*NR], 1, MR, NULL, NULL);
            g[s] = 2.0*MR*NR*kc*NP*NQ*reps/(now()-t0)*1e-9;
        }
        qsort(g, 7, sizeof(double), cmp);
        printf("%5ld %ld %ld %8.3f %8.3f %8.3f\n", kc, MR, NR, g[3], g[0],
               g[6]);
        sum += C[0];
        free(A); free(B); free(C);
    }
    return sum==12345.0;
}
