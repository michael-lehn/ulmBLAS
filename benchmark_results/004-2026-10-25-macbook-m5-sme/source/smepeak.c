/*
 *  smepeak: throughput of SME FP64 outer products (FMOPA) on one core.
 *
 *      smepeak
 *
 *  1. FMOPA za.d with operands in registers, T = 1, 2, 4, 8 independent
 *     tiles (T = 1: latency bound if the latency exceeds one issue slot).
 *  2. The 32 x 16 pattern (per k: ld1d x4 of A, ld1d x2 of B, 8 FMOPA),
 *     panels of A and B of kc steps, repeated over the same panels; kc is
 *     varied so that the panels are in L1, L2 or memory.
 *
 *  GFLOP/s = 2*8*8 flops per FMOPA.  Thread with QoS user-interactive.
 */
#define _DARWIN_C_SOURCE
#include <arm_sme.h>
#include <pthread/qos.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double
now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9*t.tv_nsec;
}

#define MOPA(t, a, b) svmopa_za64_m(t, pt, pt, a, b)

__arm_new("za") __arm_locally_streaming static void
regs(long n, int tiles, const double *x)
{
    svbool_t    pt = svptrue_b64();
    svfloat64_t a0 = svld1(pt, x), a1 = svld1(pt, x+8);
    svfloat64_t b0 = svld1(pt, x+16), b1 = svld1(pt, x+24);
    svzero_za();
    switch (tiles) {
    case 1:
        for (long i=0; i<n; ++i) {
            MOPA(0,a0,b0); MOPA(0,a1,b1); MOPA(0,a0,b1); MOPA(0,a1,b0);
            MOPA(0,a0,b0); MOPA(0,a1,b1); MOPA(0,a0,b1); MOPA(0,a1,b0);
        }
        break;
    case 2:
        for (long i=0; i<n; ++i) {
            MOPA(0,a0,b0); MOPA(1,a1,b1); MOPA(0,a0,b1); MOPA(1,a1,b0);
            MOPA(0,a0,b0); MOPA(1,a1,b1); MOPA(0,a0,b1); MOPA(1,a1,b0);
        }
        break;
    case 4:
        for (long i=0; i<n; ++i) {
            MOPA(0,a0,b0); MOPA(1,a1,b1); MOPA(2,a0,b1); MOPA(3,a1,b0);
            MOPA(0,a0,b0); MOPA(1,a1,b1); MOPA(2,a0,b1); MOPA(3,a1,b0);
        }
        break;
    default:
        for (long i=0; i<n; ++i) {
            MOPA(0,a0,b0); MOPA(1,a1,b1); MOPA(2,a0,b1); MOPA(3,a1,b0);
            MOPA(4,a0,b0); MOPA(5,a1,b1); MOPA(6,a0,b1); MOPA(7,a1,b0);
        }
    }
}

__arm_new("za") __arm_locally_streaming static void
panel(long rep, long kc, const double *A, const double *B)
{
    svbool_t  pt = svptrue_b64();
    svcount_t pc = svptrue_c64();
    svzero_za();
    for (long r=0; r<rep; ++r) {
        const double *a = A, *b = B;
        for (long k=0; k<kc; ++k) {
            svfloat64x4_t x = svld1_f64_x4(pc, a);
            svfloat64x2_t y = svld1_f64_x2(pc, b);
            MOPA(0, svget4(x,0), svget2(y,0));
            MOPA(1, svget4(x,1), svget2(y,0));
            MOPA(2, svget4(x,2), svget2(y,0));
            MOPA(3, svget4(x,3), svget2(y,0));
            MOPA(4, svget4(x,0), svget2(y,1));
            MOPA(5, svget4(x,1), svget2(y,1));
            MOPA(6, svget4(x,2), svget2(y,1));
            MOPA(7, svget4(x,3), svget2(y,1));
            a += 32;
            b += 16;
        }
    }
}

int
main(void)
{
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    double x[32];
    for (int i=0; i<32; ++i) x[i] = 1e-9*i;

    for (int t=1; t<=8; t*=2) {
        long   n = 1L<<24;
        double best = 0;
        regs(1000, t, x);
        for (int s=0; s<5; ++s) {
            double t0 = now();
            regs(n, t, x);
            double g = 8.0*n*128/(now()-t0)*1e-9;
            if (g>best) best = g;
        }
        printf("regs   tiles %d            %8.1f GFLOP/s\n", t, best);
    }
    long kcs[] = {64, 256, 512, 1024, 2048, 4096, 16384, 65536, 262144};
    for (int i=0; i<9; ++i) {
        long   kc  = kcs[i];
        double *A  = aligned_alloc(128, 8*32*kc), *B = aligned_alloc(128, 8*16*kc);
        for (long j=0; j<32*kc; ++j) A[j] = 1e-9*j;
        for (long j=0; j<16*kc; ++j) B[j] = 1e-9*j;
        long   rep = (1L<<26)/kc/8;
        double best = 0;
        panel(rep/10+1, kc, A, B);
        for (int s=0; s<5; ++s) {
            double t0 = now();
            panel(rep, kc, A, B);
            double g = 8.0*rep*kc*128/(now()-t0)*1e-9;
            if (g>best) best = g;
        }
        printf("panel  kc %6ld (%7.0f KiB) %8.1f GFLOP/s\n", kc,
               8.0*48*kc/1024, best);
        free(A); free(B);
    }
    return 0;
}
