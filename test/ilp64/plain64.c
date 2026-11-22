/*
 *  libulmblas64: the interfaces with 64-bit integers under the usual names
 *  (dgemm_, cblas_dgemm, ...; interfaces/blas/shared with INTERFACE64=1, or
 *  the front of interfaces/blas/dispatch).  Results of a few routines of
 *  the Fortran 77 interface (real and complex, functions with real,
 *  complex and integer results, dgetrf_/dgetrs_ with 64-bit pivots,
 *  lsame_ with a 64-bit LOGICAL) and of the C interface (cblas.h of
 *  libulmblas64, 64-bit index type) against simple loops; that integer
 *  arguments are read with 64 bits (leading dimensions 2^32 with m = n = k
 *  = 0 are valid); and that the error handlers xerbla_ and cblas_xerbla of
 *  this program, with 64-bit integers, are called.  The Fortran
 *  prototypes are declared here.  Exit status 1 on failure.  Used by
 *  check-ilp64.sh.
 */
#include <cblas.h>
#include <complex.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef int64_t i64;

#if !defined(ULMBLAS_INDEX_TYPE)
#error "cblas.h of libulmblas64 expected (ULMBLAS_INDEX_TYPE)"
#endif

void
dgemm_(const char *transA, const char *transB, const i64 *m, const i64 *n,
       const i64 *k, const double *alpha, const double *A, const i64 *ldA,
       const double *B, const i64 *ldB, const double *beta, double *C,
       const i64 *ldC, size_t lenTransA, size_t lenTransB);
double
dnrm2_(const i64 *n, const double *x, const i64 *incX);
i64
idamax_(const i64 *n, const double *x, const i64 *incX);
double complex
zdotc_(const i64 *n, const double complex *x, const i64 *incX,
       const double complex *y, const i64 *incY);
i64
lsame_(const char *ca, const char *cb, size_t lenCa, size_t lenCb);
void
dgetrf_(const i64 *m, const i64 *n, double *A, const i64 *ldA, i64 *ipiv,
        i64 *info);
void
dgetrs_(const char *trans, const i64 *n, const i64 *nrhs, const double *A,
        const i64 *ldA, const i64 *ipiv, double *B, const i64 *ldB, i64 *info,
        size_t lenTrans);

static i64 failures = 0;

static void
check(i64 ok, const char *what)
{
    if (!ok) {
	printf("FAIL %s\n", what);
	++failures;
    } else {
	printf("PASS %s\n", what);
    }
}

/*  the error handlers of the program, with 64-bit integers */
static i64 f77Info = 0;
static char f77Name[8];
static i64 cInfo = 0;
static char cName[32];

void
xerbla_(const char *name, const i64 *info, size_t len)
{
    f77Info = *info;
    snprintf(f77Name, sizeof(f77Name), "%.*s", (int)(len < 7 ? len : 7), name);
}

void
cblas_xerbla(ULMBLAS_INDEX_TYPE info, const char *rout, const char *form, ...)
{
    (void)form;
    cInfo = info;
    snprintf(cName, sizeof(cName), "%s", rout);
}

static double
rnd(void)
{
    static uint64_t seed = 0x9e3779b97f4a7c15ULL;
    seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(seed >> 11) / 9007199254740992.0 - 0.5;
}

#define M 13
#define N 9
#define K 7
#define LD 17

int
main(void)
{
    double A[LD * LD], B[LD * LD], C[LD * LD], R[LD * LD];
    for (i64 i = 0; i < LD * LD; ++i) {
	A[i] = rnd();
	B[i] = rnd();
	C[i] = rnd();
    }

    /*  dgemm_: C = 2 A B + C (column major, leading dimension LD) */
    {
	i64 m = M, n = N, k = K, ld = LD;
	double alpha = 2, beta = 1;
	memcpy(R, C, sizeof(C));
	dgemm_("N", "N", &m, &n, &k, &alpha, A, &ld, B, &ld, &beta, R, &ld, 1,
	       1);
	double err = 0;
	for (i64 j = 0; j < N; ++j) {
	    for (i64 i = 0; i < M; ++i) {
		double s = 0;
		for (i64 l = 0; l < K; ++l) {
		    s += A[i + l * LD] * B[l + j * LD];
		}
		err = fmax(err, fabs(R[i + j * LD] - (2 * s + C[i + j * LD])));
	    }
	}
	check(err < 1e-13, "dgemm_");
    }

    /*  cblas_dgemm, row major */
    {
	memcpy(R, C, sizeof(C));
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0, A,
	            LD, B, LD, 0.0, R, LD);
	double err = 0;
	for (i64 i = 0; i < M; ++i) {
	    for (i64 j = 0; j < N; ++j) {
		double s = 0;
		for (i64 l = 0; l < K; ++l) {
		    s += A[i * LD + l] * B[j * LD + l];
		}
		err = fmax(err, fabs(R[i * LD + j] - s));
	    }
	}
	check(err < 1e-13, "cblas_dgemm (row major)");
    }

    /*  functions with real, integer and complex results */
    {
	i64 n = M, inc = 2;
	double s = 0;
	for (i64 i = 0; i < M; ++i) {
	    s += A[2 * i] * A[2 * i];
	}
	check(fabs(dnrm2_(&n, A, &inc) - sqrt(s)) < 1e-14, "dnrm2_");

	double x[M];
	for (i64 i = 0; i < M; ++i) {
	    x[i] = i == 9 ? -5.0 : 0.5;
	}
	i64 one = 1;
	check(idamax_(&n, x, &one) == 10, "idamax_ (1-based, 64-bit result)");
	check(cblas_idamax(M, x, 1) == 9, "cblas_idamax");

	double complex u[M], v[M], d = 0;
	for (i64 i = 0; i < M; ++i) {
	    u[i] = rnd() + rnd() * I;
	    v[i] = rnd() + rnd() * I;
	    d += conj(u[i]) * v[i];
	}
	check(cabs(zdotc_(&n, u, &one, v, &one) - d) < 1e-14, "zdotc_");
	double complex e;
	cblas_zdotc_sub(M, u, 1, v, 1, &e);
	check(cabs(e - d) < 1e-14, "cblas_zdotc_sub");
    }

    check(lsame_("q", "Q", 1, 1) == 1 && lsame_("q", "R", 1, 1) == 0,
          "lsame_ (64-bit LOGICAL)");

    /*  dgetrf_, dgetrs_: A x = b with 64-bit pivots */
    {
	i64 n = N, ld = LD, nrhs = 1, ipiv[N], info = -1;
	double LU[LD * LD], b[N], x[N];
	memcpy(LU, A, sizeof(A));
	for (i64 i = 0; i < N; ++i) {
	    LU[i + i * LD] += N;
	    x[i] = rnd();
	}
	for (i64 i = 0; i < N; ++i) {
	    b[i] = 0;
	    for (i64 j = 0; j < N; ++j) {
		b[i] += (A[i + j * LD] + (i == j ? N : 0)) * x[j];
	    }
	}
	dgetrf_(&n, &n, LU, &ld, ipiv, &info);
	check(info == 0, "dgetrf_");
	dgetrs_("N", &n, &nrhs, LU, &ld, ipiv, b, &n, &info, 1);
	double err = 0;
	for (i64 i = 0; i < N; ++i) {
	    err = fmax(err, fabs(b[i] - x[i]));
	}
	check(info == 0 && err < 1e-12, "dgetrs_");
    }

    /*  64-bit arguments: valid leading dimensions 2^32 with m = n = k = 0
     *  (quick return), then an invalid one: the handlers of the program */
    {
	i64 zero = 0, two = 2, one = 1, big = (i64)1 << 32;
	double alpha = 1, beta = 1;
	dgemm_("N", "N", &zero, &zero, &zero, &alpha, A, &big, B, &big, &beta,
	       R, &big, 1, 1);
	check(f77Info == 0, "dgemm_ reads 64-bit leading dimensions");
	dgemm_("N", "N", &two, &two, &two, &alpha, A, &one, B, &two, &beta, R,
	       &two, 1, 1);
	check(f77Info == 8 && strncmp(f77Name, "DGEMM", 5) == 0,
	      "xerbla_ of the program called (dgemm_, parameter 8)");
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 0, 0, 0, alpha,
	            A, big, B, big, beta, R, big);
	check(cInfo == 0, "cblas_dgemm reads 64-bit leading dimensions");
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, 2, 2, alpha,
	            A, 1, B, 2, beta, R, 2);
	check(cInfo != 0 && strstr(cName, "cblas_dgemm") != NULL,
	      "cblas_xerbla of the program called (cblas_dgemm)");
    }

    printf("%s: %lld failures\n", failures ? "FAIL" : "PASS",
           (long long)failures);
    return failures ? 1 : 0;
}
