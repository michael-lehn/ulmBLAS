/*
 *  peak: FMA throughput of one core with 256-bit (ymm) and 512-bit (zmm)
 *  vectors, double precision.  12 independent accumulators per width
 *  (latency 4 x 2 FMA units needs at least 8), about 1 s per test, best of
 *  REPEAT.  This is the upper bound for a DGEMM micro kernel of that width
 *  at the clock the core actually runs under this load (with AVX-512 on
 *  Intel CPUs usually lower than with AVX2).
 *
 *      gcc -O2 peak.c -o peak && ./peak
 */
#include <immintrin.h>
#include <stdio.h>
#include <time.h>

enum { REPEAT = 5 };

/*  volatile: the compiler must not fold a*1 + c into an addition  */
static volatile double B_ = 1.0, C_ = 1e-9;

static double
now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9*t.tv_nsec;
}

#define ACC(OP, S)                                                         \
    a0 = OP(a0, b, c); a1 = OP(a1, b, c); a2 = OP(a2, b, c);               \
    a3 = OP(a3, b, c); a4 = OP(a4, b, c); a5 = OP(a5, b, c);               \
    a6 = OP(a6, b, c); a7 = OP(a7, b, c); a8 = OP(a8, b, c);               \
    a9 = OP(a9, b, c); a10 = OP(a10, b, c); a11 = OP(a11, b, c);

__attribute__((target("avx2,fma"))) static double
fma256(long iter)
{
    __m256d b = _mm256_set1_pd(B_), c = _mm256_set1_pd(C_);
    /*  different start values: equal accumulators would be merged (CSE)  */
    __m256d a0 = _mm256_set1_pd(C_*1), a1 = _mm256_set1_pd(C_*2),
            a2 = _mm256_set1_pd(C_*3), a3 = _mm256_set1_pd(C_*4),
            a4 = _mm256_set1_pd(C_*5), a5 = _mm256_set1_pd(C_*6),
            a6 = _mm256_set1_pd(C_*7), a7 = _mm256_set1_pd(C_*8),
            a8 = _mm256_set1_pd(C_*9), a9 = _mm256_set1_pd(C_*10),
            a10 = _mm256_set1_pd(C_*11), a11 = _mm256_set1_pd(C_*12);
    for (long i = 0; i < iter; ++i) {
        ACC(_mm256_fmadd_pd, 0)
    }
    __m256d s = _mm256_add_pd(_mm256_add_pd(_mm256_add_pd(a0, a1),
                                            _mm256_add_pd(a2, a3)),
                              _mm256_add_pd(_mm256_add_pd(a4, a5),
                                            _mm256_add_pd(a6, a7)));
    s = _mm256_add_pd(s, _mm256_add_pd(_mm256_add_pd(a8, a9),
                                       _mm256_add_pd(a10, a11)));
    double r[4];
    _mm256_storeu_pd(r, s);
    return r[0] + r[1] + r[2] + r[3];
}

__attribute__((target("avx512f"))) static double
fma512(long iter)
{
    __m512d b = _mm512_set1_pd(B_), c = _mm512_set1_pd(C_);
    /*  different start values: equal accumulators would be merged (CSE)  */
    __m512d a0 = _mm512_set1_pd(C_*1), a1 = _mm512_set1_pd(C_*2),
            a2 = _mm512_set1_pd(C_*3), a3 = _mm512_set1_pd(C_*4),
            a4 = _mm512_set1_pd(C_*5), a5 = _mm512_set1_pd(C_*6),
            a6 = _mm512_set1_pd(C_*7), a7 = _mm512_set1_pd(C_*8),
            a8 = _mm512_set1_pd(C_*9), a9 = _mm512_set1_pd(C_*10),
            a10 = _mm512_set1_pd(C_*11), a11 = _mm512_set1_pd(C_*12);
    for (long i = 0; i < iter; ++i) {
        ACC(_mm512_fmadd_pd, 0)
    }
    __m512d s = _mm512_add_pd(_mm512_add_pd(_mm512_add_pd(a0, a1),
                                            _mm512_add_pd(a2, a3)),
                              _mm512_add_pd(_mm512_add_pd(a4, a5),
                                            _mm512_add_pd(a6, a7)));
    s = _mm512_add_pd(s, _mm512_add_pd(_mm512_add_pd(a8, a9),
                                       _mm512_add_pd(a10, a11)));
    return _mm512_reduce_add_pd(s);
}

static void
run(const char *name, double (*f)(long), int width)
{
    long iter = 1000;
    double t, best = 0, sink = 0;
    for (;;) {                       /* calibrate to about 1 s */
        t = now(); sink += f(iter); t = now() - t;
        if (t > 0.2) break;
        iter *= 4;
    }
    iter = (long)(iter / t);
    for (int r = 0; r < REPEAT; ++r) {
        t = now(); sink += f(iter); t = now() - t;
        double gf = 2.0*12*width*iter/t*1e-9;
        if (gf > best) best = gf;
    }
    printf("%s %8.2f GFLOP/s  (%.3f G FMA instructions/s; sink %g)\n",
           name, best, best/(2.0*width), sink);
}

int
main(void)
{
    __builtin_cpu_init();
    run("fma-ymm", fma256, 4);
    if (__builtin_cpu_supports("avx512f")) {
        run("fma-zmm", fma512, 8);
    }
    return 0;
}
