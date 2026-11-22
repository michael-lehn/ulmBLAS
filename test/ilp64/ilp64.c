/*
 *  The ILP64 interfaces of the shared library (dgemm_64_, cblas_dgemm_64,
 *  ...) against the LP64 interfaces of the same library: results of a few
 *  routines of each level (real and complex, Fortran functions with
 *  real, complex and integer results), lsame_64_, and that the integer
 *  arguments are read with 64 bits: dgemm_64_ with m = n = k = 0 and the
 *  leading dimensions 2^32 is valid (an LP64 read would see 0 and report
 *  an error).  The error handlers xerbla_64_ and cblas_xerbla_64 of this
 *  program must be called by the library (also through the dispatching
 *  library).  Prototypes are declared here, independent of the headers of
 *  the build.  Exit status 1 on failure.  Used by check-ilp64.sh.
 */
#include <complex.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef int64_t i64;
typedef int32_t i32;

/*  Fortran 77, LP64 and ILP64 */
void
dgemm_(const char *transA, const char *transB, const i32 *m, const i32 *n,
       const i32 *k, const double *alpha, const double *A, const i32 *ldA,
       const double *B, const i32 *ldB, const double *beta, double *C,
       const i32 *ldC, size_t lenTransA, size_t lenTransB);
void
dgemm_64_(const char *transA, const char *transB, const i64 *m, const i64 *n,
          const i64 *k, const double *alpha, const double *A, const i64 *ldA,
          const double *B, const i64 *ldB, const double *beta, double *C,
          const i64 *ldC, size_t lenTransA, size_t lenTransB);
void
zherk_(const char *uplo, const char *trans, const i32 *n, const i32 *k,
       const double *alpha, const double complex *A, const i32 *ldA,
       const double *beta, double complex *C, const i32 *ldC, size_t lenUplo,
       size_t lenTrans);
void
zherk_64_(const char *uplo, const char *trans, const i64 *n, const i64 *k,
          const double *alpha, const double complex *A, const i64 *ldA,
          const double *beta, double complex *C, const i64 *ldC, size_t lenUplo,
          size_t lenTrans);
void
sgemv_(const char *trans, const i32 *m, const i32 *n, const float *alpha,
       const float *A, const i32 *ldA, const float *x, const i32 *incX,
       const float *beta, float *y, const i32 *incY, size_t lenTrans);
void
sgemv_64_(const char *trans, const i64 *m, const i64 *n, const float *alpha,
          const float *A, const i64 *ldA, const float *x, const i64 *incX,
          const float *beta, float *y, const i64 *incY, size_t lenTrans);
double
ddot_(const i32 *n, const double *x, const i32 *incX, const double *y,
      const i32 *incY);
double
ddot_64_(const i64 *n, const double *x, const i64 *incX, const double *y,
         const i64 *incY);
float complex
cdotc_(const i32 *n, const float complex *x, const i32 *incX,
       const float complex *y, const i32 *incY);
float complex
cdotc_64_(const i64 *n, const float complex *x, const i64 *incX,
          const float complex *y, const i64 *incY);
i32
izamax_(const i32 *n, const double complex *x, const i32 *incX);
i64
izamax_64_(const i64 *n, const double complex *x, const i64 *incX);
void
daxpy_64_(const i64 *n, const double *alpha, const double *x, const i64 *incX,
          double *y, const i64 *incY);
i64
lsame_64_(const char *ca, const char *cb, size_t lenCa, size_t lenCb);

/*  CBLAS, LP64 and ILP64 (the enums as int) */
void
cblas_dgemm(int order, int transA, int transB, i32 m, i32 n, i32 k,
            double alpha, const double *A, i32 ldA, const double *B, i32 ldB,
            double beta, double *C, i32 ldC);
void
cblas_dgemm_64(int order, int transA, int transB, i64 m, i64 n, i64 k,
               double alpha, const double *A, i64 ldA, const double *B, i64 ldB,
               double beta, double *C, i64 ldC);
void
cblas_ztrsm(int order, int side, int uplo, int transA, int diag, i32 m, i32 n,
            const void *alpha, const void *A, i32 ldA, void *B, i32 ldB);
void
cblas_ztrsm_64(int order, int side, int uplo, int transA, int diag, i64 m,
               i64 n, const void *alpha, const void *A, i64 ldA, void *B,
               i64 ldB);
size_t
cblas_idamax(i32 n, const double *x, i32 incX);
size_t
cblas_idamax_64(i64 n, const double *x, i64 incX);
void
cblas_zdotc_sub_64(i64 n, const void *x, i64 incX, const void *y, i64 incY,
                   void *dotc);
double
cblas_dnrm2_64(i64 n, const double *x, i64 incX);
double
cblas_dnrm2(i32 n, const double *x, i32 incX);

enum
{
    RowMajor = 101,
    ColMajor = 102
};
enum
{
    NoTrans = 111,
    Trans = 112,
    ConjTrans = 113
};
enum
{
    Upper = 121,
    Lower = 122
};
enum
{
    NonUnit = 131,
    Unit = 132
};
enum
{
    Left = 141,
    Right = 142
};

static int failures = 0;

static void
check(int ok, const char *what)
{
    if (!ok) {
	printf("FAIL %s\n", what);
	++failures;
    }
}

static int
same(const double *x, const double *y, i64 n)
{
    return memcmp(x, y, (size_t)n * sizeof(double)) == 0;
}

/*  the error handlers set at run time (only the setters and getters: the
 *  handlers of the program below replace the library's) */
typedef void (*CHandler64)(i64 info, const char *rout, const char *form,
                           va_list ap);
typedef void (*F77Handler64)(const char *srname, i64 *info, size_t lenSrname);
CHandler64
cblas_set_xerbla_64(CHandler64 handler, F77Handler64 f77Handler);
CHandler64
cblas_get_xerbla_64(void);
void
set_blas_xerbla_64_(F77Handler64 handler);
void
get_blas_xerbla_64_(F77Handler64 *handler);

static void
cHandler64(i64 info, const char *rout, const char *form, va_list ap)
{
    (void)info;
    (void)rout;
    (void)form;
    (void)ap;
}

static void
f77Handler64(const char *srname, i64 *info, size_t lenSrname)
{
    (void)srname;
    (void)info;
    (void)lenSrname;
}

/*  the error handlers of the program */
static i64 f77Info = 0;
static char f77Name[8];
static i64 cInfo = 0;
static char cName[32];

void
xerbla_64_(const char *name, const i64 *info, size_t len)
{
    f77Info = *info;
    snprintf(f77Name, sizeof(f77Name), "%.*s", (int)(len < 7 ? len : 7), name);
}

void
cblas_xerbla_64(i64 info, const char *rout, const char *form, ...)
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
    double A[LD * LD], B[LD * LD], C1[LD * LD], C2[LD * LD];
    double complex Z[LD * LD], Z1[LD * LD], Z2[LD * LD];
    float S[LD * LD], x[LD], y1[LD], y2[LD];
    float complex u[LD], v[LD];

    for (i64 i = 0; i < LD * LD; ++i) {
	A[i] = rnd();
	B[i] = rnd();
	C1[i] = C2[i] = rnd();
	Z[i] = rnd() + I * rnd();
	Z1[i] = Z2[i] = rnd() + I * rnd();
	S[i] = (float)rnd();
    }
    for (i64 i = 0; i < LD; ++i) {
	x[i] = (float)rnd();
	y1[i] = y2[i] = (float)rnd();
	u[i] = (float)rnd() + I * (float)rnd();
	v[i] = (float)rnd() + I * (float)rnd();
    }

    /*  level 3 */
    {
	i32 m = M, n = N, k = K, ld = LD;
	i64 m8 = M, n8 = N, k8 = K, ld8 = LD;
	double alpha = 1.5, beta = -0.5;
	dgemm_("N", "T", &m, &n, &k, &alpha, A, &ld, B, &ld, &beta, C1, &ld, 1,
	       1);
	dgemm_64_("N", "T", &m8, &n8, &k8, &alpha, A, &ld8, B, &ld8, &beta, C2,
	          &ld8, 1, 1);
	check(same(C1, C2, LD * LD), "dgemm_64_");

	cblas_dgemm(RowMajor, Trans, NoTrans, m, n, k, alpha, A, ld, B, ld,
	            beta, C1, ld);
	cblas_dgemm_64(RowMajor, Trans, NoTrans, m8, n8, k8, alpha, A, ld8, B,
	               ld8, beta, C2, ld8);
	check(same(C1, C2, LD * LD), "cblas_dgemm_64");

	zherk_("L", "C", &n, &k, &alpha, Z, &ld, &beta, Z1, &ld, 1, 1);
	zherk_64_("L", "C", &n8, &k8, &alpha, Z, &ld8, &beta, Z2, &ld8, 1, 1);
	check(memcmp(Z1, Z2, sizeof(Z1)) == 0, "zherk_64_");

	/*  a well-conditioned triangle */
	double complex T[LD * LD];
	memcpy(T, Z, sizeof(T));
	for (i64 i = 0; i < LD; ++i) {
	    T[i * LD + i] += 4;
	}
	double complex za = 0.5 - 2 * I;
	cblas_ztrsm(ColMajor, Right, Upper, ConjTrans, NonUnit, m, n, &za, T,
	            ld, Z1, ld);
	cblas_ztrsm_64(ColMajor, Right, Upper, ConjTrans, NonUnit, m8, n8, &za,
	               T, ld8, Z2, ld8);
	check(memcmp(Z1, Z2, sizeof(Z1)) == 0, "cblas_ztrsm_64");
    }

    /*  level 2 and 1 */
    {
	i32 m = M, n = N, ld = LD, inc = 1, inc2 = 2;
	i64 m8 = M, n8 = N, ld8 = LD, inc8 = 1, inc28 = 2;
	float alpha = 0.75f, beta = 2.0f;
	sgemv_("T", &m, &n, &alpha, S, &ld, x, &inc, &beta, y1, &inc2, 1);
	sgemv_64_("T", &m8, &n8, &alpha, S, &ld8, x, &inc8, &beta, y2, &inc28,
	          1);
	check(memcmp(y1, y2, sizeof(y1)) == 0, "sgemv_64_");

	check(ddot_(&m, A, &inc2, B, &inc) ==
	          ddot_64_(&m8, A, &inc28, B, &inc8),
	      "ddot_64_");
	float complex c1 = cdotc_(&n, u, &inc, v, &inc2);
	float complex c2 = cdotc_64_(&n8, u, &inc8, v, &inc28);
	check(c1 == c2 && c1 != 0, "cdotc_64_");
	check(izamax_(&n, Z, &inc2) == izamax_64_(&n8, Z, &inc28),
	      "izamax_64_");
	check(cblas_idamax(m, A, 3) == cblas_idamax_64(m8, A, 3),
	      "cblas_idamax_64");
	check(cblas_dnrm2(m, A, 2) == cblas_dnrm2_64(m8, A, 2),
	      "cblas_dnrm2_64");
	double complex d1 = 0, d2 = 0;
	for (i64 i = 0; i < N; ++i) {
	    d1 += conj(Z[i]) * Z1[2 * i];
	}
	cblas_zdotc_sub_64(n8, Z, 1, Z1, 2, &d2);
	check(cabs(d1 - d2) <= 1e-13 * (1 + cabs(d1)), "cblas_zdotc_sub_64");

	double a = 2;
	memcpy(C2, C1, sizeof(C1));
	daxpy_64_(&m8, &a, A, &inc8, C2, &inc28);
	int ok = 1;
	for (i64 i = 0; i < LD * LD; ++i) {
	    double expected =
	        (i % 2 == 0 && i / 2 < M) ? C1[i] + 2 * A[i / 2] : C1[i];
	    ok = ok && C2[i] == expected;
	}
	check(ok, "daxpy_64_");
    }

    check(lsame_64_("q", "Q", 1, 1) == 1 && lsame_64_("q", "R", 1, 1) == 0,
          "lsame_64_");

    /*  64-bit arguments: valid leading dimensions 2^32 with m = n = k = 0
     *  (quick return), then an invalid one */
    {
	i64 zero = 0, two = 2, big = (i64)1 << 32;
	double alpha = 1, beta = 1;
	dgemm_64_("N", "N", &zero, &zero, &zero, &alpha, A, &big, B, &big,
	          &beta, C2, &big, 1, 1);
	check(f77Info == 0, "dgemm_64_ reads 64-bit leading dimensions");
	i64 one = 1;
	dgemm_64_("N", "N", &two, &two, &two, &alpha, A, &one, B, &two, &beta,
	          C2, &two, 1, 1);
	check(f77Info == 8 && strncmp(f77Name, "DGEMM", 5) == 0,
	      "xerbla_64_ of the program called (dgemm_64_, parameter 8)");
	cblas_dgemm_64(ColMajor, NoTrans, NoTrans, zero, zero, zero, alpha, A,
	               big, B, big, beta, C2, big);
	check(cInfo == 0, "cblas_dgemm_64 reads 64-bit leading dimensions");
	cblas_dgemm_64(ColMajor, NoTrans, NoTrans, two, two, two, alpha, A, one,
	               B, two, beta, C2, two);
	check(cInfo != 0 && strstr(cName, "cblas_dgemm") != NULL,
	      "cblas_xerbla_64 of the program called (cblas_dgemm_64)");
    }

    {
	check(cblas_set_xerbla_64(cHandler64, NULL) == NULL &&
	          cblas_get_xerbla_64() == cHandler64 &&
	          cblas_set_xerbla_64(NULL, NULL) == cHandler64 &&
	          cblas_get_xerbla_64() == NULL,
	      "cblas_set_xerbla_64, cblas_get_xerbla_64");
	F77Handler64 initial = NULL, current = NULL;
	get_blas_xerbla_64_(&initial);
	set_blas_xerbla_64_(f77Handler64);
	get_blas_xerbla_64_(&current);
	check(initial != NULL && current == f77Handler64,
	      "set_blas_xerbla_64_, get_blas_xerbla_64_");
	set_blas_xerbla_64_(NULL);
	get_blas_xerbla_64_(&current);
	check(current == initial,
	      "get_blas_xerbla_64_: the default after reset");
    }

    printf("ilp64: %s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
