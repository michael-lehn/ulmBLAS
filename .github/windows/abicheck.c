/*
   abicheck DLL: loads DLL (libulmblas_lbt.dll) as Julia does and checks

   - the ILP64 interface: idamax_64_ with n = 0xFFFFFFFF00000003 (the probe
     of libblastrampoline) returns 0 if n is read with 64 bits;
   - S, D, C and ZGEMM, DTRSM and ulmblas_dgemm_strided (row-major and
     negative strides) against naive loops;
   - that these calls preserve xmm6-xmm15, which are callee-saved in the
     Windows x64 ABI (xmmcall.S).

   Exit status 0 if all checks pass.
*/
#include <windows.h>
#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

typedef int64_t idx;

typedef idx  (*idamax_t)(const idx *, const double *, const idx *);
typedef void (*dgemm_t)(const char *, const char *, const idx *, const idx *,
                        const idx *, const double *, const double *,
                        const idx *, const double *, const idx *,
                        const double *, double *, const idx *, size_t, size_t);
typedef void (*sgemm_t)(const char *, const char *, const idx *, const idx *,
                        const idx *, const float *, const float *, const idx *,
                        const float *, const idx *, const float *, float *,
                        const idx *, size_t, size_t);
typedef void (*xgemm_t)(const char *, const char *, const idx *, const idx *,
                        const idx *, const void *, const void *, const idx *,
                        const void *, const idx *, const void *, void *,
                        const idx *, size_t, size_t);
typedef void (*dtrsm_t)(const char *, const char *, const char *, const char *,
                        const idx *, const idx *, const double *,
                        const double *, const idx *, double *, const idx *,
                        size_t, size_t, size_t, size_t);
typedef void (*dgemm_strided_t)(idx, idx, idx, double, const double *, idx,
                                idx, const double *, idx, idx, double,
                                double *, idx, idx);

int xmmcall(void (*f)(void));

static dgemm_t dgemm;
static sgemm_t sgemm;
static xgemm_t zgemm, cgemm;
static dtrsm_t dtrsm;
static dgemm_strided_t dgemm_strided;

/*  odd sizes: partial micro tiles; k > 256 (more than one k block) */
enum { M = 301, K = 419, N = 263 };
static double A[M*K], B[K*N], R[M*N], C[M*N], U[M*M], X[M*N];
static float  As[M*K], Bs[K*N], Cs[M*N];
static double complex Az[M*K], Bz[K*N], Cz[M*N];
static float complex  Ac[M*K], Bc[K*N], Cc[M*N];

static double err_d, err_s, err_z, err_c, err_trsm, err_rowmajor, err_neg;

static void
run(void)
{
    const idx m = M, n = N, k = K;
    const double one = 1, zero = 0;
    const float onef = 1, zerof = 0;
    const double complex onez = 1, zeroz = 0;
    const float complex onec = 1, zeroc = 0;

    dgemm("N", "N", &m, &n, &k, &one, A, &m, B, &k, &zero, C, &m, 1, 1);
    sgemm("N", "N", &m, &n, &k, &onef, As, &m, Bs, &k, &zerof, Cs, &m, 1, 1);
    zgemm("N", "N", &m, &n, &k, &onez, Az, &m, Bz, &k, &zeroz, Cz, &m, 1, 1);
    cgemm("N", "N", &m, &n, &k, &onec, Ac, &m, Bc, &k, &zeroc, Cc, &m, 1, 1);
    for (idx i = 0; i < M*N; ++i) {
        err_d = fmax(err_d, fabs(C[i] - R[i]));
        err_s = fmax(err_s, fabs(Cs[i] - R[i]));
        err_z = fmax(err_z, cabs(Cz[i] - (1 + 2*I)*R[i]));
        err_c = fmax(err_c, cabs(Cc[i] - (1 + 2*I)*R[i]));
    }

    /*  X = U^{-1} R, then compare U X with R */
    for (idx i = 0; i < M*N; ++i) X[i] = R[i];
    dtrsm("L", "U", "N", "N", &m, &n, &one, U, &m, X, &m, 1, 1, 1, 1);
    for (idx j = 0; j < N; ++j)
        for (idx i = 0; i < M; ++i) {
            double s = 0;
            for (idx l = i; l < M; ++l) s += U[i + l*M]*X[l + j*M];
            err_trsm = fmax(err_trsm, fabs(s - R[i + j*M])/(1 + fabs(R[i + j*M])));
        }

    /*  row-major C (incRowC = n, incColC = 1) */
    dgemm_strided(m, n, k, 1.0, A, 1, m, B, 1, k, 0.0, C, n, 1);
    for (idx i = 0; i < M; ++i)
        for (idx j = 0; j < N; ++j)
            err_rowmajor = fmax(err_rowmajor, fabs(C[i*N + j] - R[i + j*M]));

    /*  rows of A in reverse order: negative row stride */
    dgemm_strided(m, n, k, 1.0, A + (M - 1), -1, m, B, 1, k, 0.0, C, 1, m);
    for (idx i = 0; i < M; ++i)
        for (idx j = 0; j < N; ++j)
            err_neg = fmax(err_neg, fabs(C[i + j*M] - R[(M - 1 - i) + j*M]));
}

int
main(int argc, char **argv)
{
    HMODULE h = LoadLibraryA(argv[1]);
    if (!h) {
        fprintf(stderr, "LoadLibrary(%s) failed: %lu\n", argv[1], GetLastError());
        return 1;
    }
    idamax_t idamax = (idamax_t)GetProcAddress(h, "idamax_64_");
    dgemm  = (dgemm_t)GetProcAddress(h, "dgemm_64_");
    sgemm  = (sgemm_t)GetProcAddress(h, "sgemm_64_");
    zgemm  = (xgemm_t)GetProcAddress(h, "zgemm_64_");
    cgemm  = (xgemm_t)GetProcAddress(h, "cgemm_64_");
    dtrsm  = (dtrsm_t)GetProcAddress(h, "dtrsm_64_");
    dgemm_strided = (dgemm_strided_t)GetProcAddress(h, "ulmblas_dgemm_strided");
    if (!idamax || !dgemm || !sgemm || !zgemm || !cgemm || !dtrsm || !dgemm_strided) {
        fprintf(stderr, "missing symbols\n");
        return 1;
    }

    /*  ILP64: n = 0xFFFFFFFF00000003 is negative with 64 bits (result 0),
        3 with 32 bits (result 1..3) */
    double x[3] = {1, 3, 2};
    idx probe = (idx)0xFFFFFFFF00000003ULL, inc = 1;
    idx r = idamax(&probe, x, &inc);
    idx three = 3;
    idx r3 = idamax(&three, x, &inc);
    printf("ILP64 probe: idamax(n=0xFFFFFFFF00000003) = %lld (expect 0), "
           "idamax(n=3) = %lld (expect 2)\n", (long long)r, (long long)r3);

    for (idx i = 0; i < M*K; ++i) {
        A[i] = (double)((i*7) % 5) - 2;
        As[i] = (float)A[i]; Az[i] = A[i]*(1 + 2*I); Ac[i] = (float)A[i]*(1 + 2*I);
    }
    for (idx i = 0; i < K*N; ++i) {
        B[i] = (double)((i*3) % 7) - 3;
        Bs[i] = (float)B[i]; Bz[i] = B[i]; Bc[i] = (float)B[i];
    }
    for (idx j = 0; j < N; ++j)
        for (idx i = 0; i < M; ++i) {
            double s = 0;
            for (idx l = 0; l < K; ++l) s += A[i + l*M]*B[l + j*K];
            R[i + j*M] = s;
        }
    for (idx j = 0; j < M; ++j)
        for (idx i = 0; i < M; ++i)
            U[i + j*M] = (i < j) ? A[i + j*M]/M : (i == j) ? 2 + (i % 3) : 0;

    int clobbered = xmmcall(run);

    printf("max error: dgemm %g, sgemm %g, zgemm %g, cgemm %g, dtrsm %g (rel), "
           "strided row-major %g, negative stride %g\n",
           err_d, err_s, err_z, err_c, err_trsm, err_rowmajor, err_neg);
    printf("xmm6-xmm15 not preserved: %d\n", clobbered);

    int ok = r == 0 && r3 == 2 && err_d == 0 && err_s == 0 && err_z == 0
          && err_c == 0 && err_trsm < 1e-12 && err_rowmajor == 0 && err_neg == 0
          && clobbered == 0;
    printf("%s\n", ok ? "PASSED" : "FAILED");
    return ok ? 0 : 2;
}
