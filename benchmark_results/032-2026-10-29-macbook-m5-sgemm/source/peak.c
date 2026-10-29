/*
 *  peak: double precision FMA throughput of one core (NEON, AArch64).
 *
 *      peak SECONDS
 *
 *  Runs independent fmla v.2d instructions (24 accumulators) for about
 *  SECONDS seconds and prints the rate in GFLOP/s (4 flops per fmla).  The
 *  rate is proportional to the clock frequency of the core: it is measured
 *  before each round of the benchmarks to detect throttling.  On macOS the
 *  thread runs with QoS class user-interactive (performance cores).
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#if defined(__APPLE__)
#include <pthread/qos.h>
#endif

static double
now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9*t.tv_nsec;
}

#define FMA4(a,b,c,d) \
    "fmla v" #a ".2d, v24.2d, v25.2d\n\t" "fmla v" #b ".2d, v24.2d, v25.2d\n\t" \
    "fmla v" #c ".2d, v24.2d, v25.2d\n\t" "fmla v" #d ".2d, v24.2d, v25.2d\n\t"

static void
fmas(long iterations)       /* 24*iterations fmla */
{
    __asm__ volatile (
    "movi v24.2d, #0\n\t" "movi v25.2d, #0\n\t"
    "1:\n\t"
    FMA4(0,1,2,3) FMA4(4,5,6,7) FMA4(8,9,10,11)
    FMA4(12,13,14,15) FMA4(16,17,18,19) FMA4(20,21,22,23)
    "subs %[n], %[n], #1\n\t"
    "b.ne 1b\n\t"
    : [n] "+r"(iterations)
    :
    : "v0","v1","v2","v3","v4","v5","v6","v7","v8","v9","v10","v11",
      "v12","v13","v14","v15","v16","v17","v18","v19","v20","v21","v22",
      "v23","v24","v25","cc");
}

int
main(int argc, char **argv)
{
#if defined(__APPLE__)
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
    double seconds = argc>1 ? atof(argv[1]) : 1.0;
    long   n = 1000000;
    double t0, t;

    fmas(n);
    for (;;) {
        t0 = now(); fmas(n); t = now()-t0;
        if (t > 0.05) break;
        n *= 2;
    }
    n = (long)(n*seconds/t) + 1;
    t0 = now(); fmas(n); t = now()-t0;
    printf("peak %.3f GFLOP/s\n", 24.0*4*n/t*1e-9);
    return 0;
}
