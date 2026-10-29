/*
 *  benchx: single-threaded cblas_?gemm timing (column major, NoTrans,
 *  alpha = 1, beta = 1, lda = m, ldb = k, ldc = m) for one data type,
 *  selected at compile time with -DPREC=s, d, c or z.
 *
 *      benchx SAMPLES N1 N2 ...        (square sizes m = n = k = Ni)
 *
 *  For each size: one call checked against sums in long double for CHECKS
 *  randomly chosen elements of C (bound 4*(k+2)*eps*(sum |A(i,l)*B(l,j)| +
 *  |C0(i,j)|), for complex types with the absolute values of the real
 *  products of the complex multiplication); this call is also the warm-up,
 *  extended to at least 0.3 s.  Then SAMPLES samples of at least MIN_SAMPLE
 *  seconds each.  GFLOP/s with 2mnk (s, d) or 8mnk (c, z) floating-point
 *  operations.  Output per size:
 *
 *      m n k median_GFLOPS min max calls_per_sample max_err/bound check
 *
 *  Writes nothing but standard output.
 */
#include <cblas.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_SAMPLES 64
#define MIN_SAMPLE  0.2
#define CHECKS      400

#define CAT_(a, b) a##b
#define CAT(a, b)  CAT_(a, b)

#if !defined(PREC)
#   error "define PREC as s, d, c or z"
#endif

#define s_ 1
#define d_ 2
#define c_ 3
#define z_ 4
#define P_ CAT(PREC, _)

#if P_==1 || P_==3
typedef float  real;
#   define EPS FLT_EPSILON
#else
typedef double real;
#   define EPS DBL_EPSILON
#endif
#if P_>=3
#   define COMPLEX 1
#   define NR_     2       /* reals per element */
#   define FLOPS   8.0
#else
#   define COMPLEX 0
#   define NR_     1
#   define FLOPS   2.0
#endif

#define GEMM CAT(CAT(cblas_, PREC), gemm)

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
fill(real *x, long n)
{
    for (long i=0; i<n; ++i) {
        x[i] = 2.0*rand()/RAND_MAX - 1.0;
    }
}

static void
gemm(int m, int n, int k, const real *A, const real *B, real *C)
{
#if COMPLEX
    static const real one[2] = { 1, 0 };
    GEMM(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, one, A, m, B, k,
         one, C, m);
#else
    GEMM(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, A, m, B, k,
         1, C, m);
#endif
}

/*  max over CHECKS elements of |C - (A*B + C0)| / bound, -1 if one is not
 *  finite */
static double
check(int m, int n, int k, const real *A, const real *B, const real *C0,
      const real *C)
{
    double worst = 0;
    for (int c=0; c<CHECKS; ++c) {
        int i = rand() % m, j = rand() % n;
        long ij = NR_*(i+(long)j*m);
        long double sr = C0[ij], br = fabsl((long double)C0[ij]);
#if COMPLEX
        long double si = C0[ij+1], bi = fabsl((long double)C0[ij+1]);
#endif
        for (int l=0; l<k; ++l) {
            long ia = NR_*(i+(long)l*m), ib = NR_*(l+(long)j*k);
#if COMPLEX
            long double ar = A[ia], ai = A[ia+1], xr = B[ib], xi = B[ib+1];
            sr += ar*xr - ai*xi;   br += fabsl(ar*xr) + fabsl(ai*xi);
            si += ar*xi + ai*xr;   bi += fabsl(ar*xi) + fabsl(ai*xr);
#else
            sr += (long double)A[ia]*B[ib];
            br += fabsl((long double)A[ia]*B[ib]);
#endif
        }
        double er = fabsl(C[ij]-sr)/(4*(k+2)*EPS*br+1e-300);
        if (!isfinite(C[ij]) || !(er==er)) return -1;
        if (er>worst) worst = er;
#if COMPLEX
        double ei = fabsl(C[ij+1]-si)/(4*(k+2)*EPS*bi+1e-300);
        if (!isfinite(C[ij+1]) || !(ei==ei)) return -1;
        if (ei>worst) worst = ei;
#endif
    }
    return worst;
}

static void
run(int samples, int n)
{
    int    m = n, k = n;
    real   *A  = malloc(sizeof(real)*NR_*(long)m*k);
    real   *B  = malloc(sizeof(real)*NR_*(long)k*n);
    real   *C  = malloc(sizeof(real)*NR_*(long)m*n);
    real   *C0 = malloc(sizeof(real)*NR_*(long)m*n);
    double t[MAX_SAMPLES], flops = FLOPS*m*n*k;
    long   calls = 1;

    fill(A, NR_*(long)m*k);
    fill(B, NR_*(long)k*n);
    fill(C0, NR_*(long)m*n);
    memcpy(C, C0, sizeof(real)*NR_*(long)m*n);

    double t0 = now();
    gemm(m, n, k, A, B, C);
    double t1 = now();
    double err = check(m, n, k, A, B, C0, C);
    double tc = now() - t1;           /* time of the check, not counted */
    while (t1-t0 < 0.3) {             /* warm-up of at least 0.3 s */
        gemm(m, n, k, A, B, C);
        ++calls;
        t1 = now() - tc;
    }
    calls = (long)(calls*MIN_SAMPLE/(t1-t0)) + 1;

    for (int s=0; s<samples; ++s) {
        double t0 = now();
        for (long c=0; c<calls; ++c) {
            gemm(m, n, k, A, B, C);
        }
        t[s] = flops*calls/(now()-t0)*1e-9;
    }
    qsort(t, samples, sizeof(double), cmp);
    printf("%5d %5d %5d %8.3f %8.3f %8.3f %6ld %8.4f %s\n", m, n, k,
           t[samples/2], t[0], t[samples-1], calls, err,
           (err>=0 && err<=1) ? "ok" : "WRONG");
    fflush(stdout);
    free(A); free(B); free(C); free(C0);
}

int
main(int argc, char **argv)
{
    if (argc<3) {
        fprintf(stderr, "usage: %s SAMPLES N1 N2 ...\n", argv[0]);
        return 2;
    }
    int samples = atoi(argv[1]);
    if (samples<1 || samples>MAX_SAMPLES) {
        return 2;
    }
    srand(2026);
    for (int i=2; i<argc; ++i) {
        run(samples, atoi(argv[i]));
    }
    return 0;
}
