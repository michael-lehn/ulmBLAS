/*
 *  flexiblas-test.c: ulmBLAS as the backend of FlexiBLAS (linked with
 *  -lflexiblas, run with FLEXIBLAS=ULMBLAS; packaging/ci/flexiblas-test.sh
 *  builds and runs it).  Checks:
 *
 *  - dgemm, dgesv (dgetrf, dgetrs) and dposv (dpotrf, dpotrs) give the
 *    right results;
 *  - ulmBLAS is loaded (/proc/self/maps; on x86-64 a backend of the
 *    dispatching library);
 *  - flexiblas_set_num_threads sets the threads of ulmBLAS
 *    (ulmblas_get_num_threads of the loaded libulmblas.so.0);
 *  - errors reach the XERBLA of the program (__flexiblas_set_xerbla, or
 *    the xerbla_ of the program bound directly), from BLAS (dgemm) and
 *    from the LAPACK functions of ulmBLAS (dgetrf, dpotrf).
 *
 *  flexiblas_get_num_threads is only printed: FlexiBLAS 3.5 looks up the
 *  backend's getter under a misspelled name and returns 1.
 *
 *  The integer types are those of FlexiBLAS and of the LP64 Fortran
 *  interface (int).  With -DFLEXIBLAS64: FlexiBLAS with 64-bit integers
 *  (-lflexiblas64, FLEXIBLAS64=ULMBLAS) and libulmblas64 as its backend,
 *  INTEGER is int64_t, and dgemm with leading dimensions 2^32 (m = n = k =
 *  0) must be valid.
 */
#include <dlfcn.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#if defined(FLEXIBLAS64)
typedef int64_t FInt;
#define ULMBLAS_LIBRARY "libulmblas64.so.0"
#else
typedef int FInt;
#define ULMBLAS_LIBRARY "libulmblas.so.0"
#endif

void
dgemm_(const char *transA, const char *transB, const FInt *m, const FInt *n,
       const FInt *k, const double *alpha, const double *A, const FInt *ldA,
       const double *B, const FInt *ldB, const double *beta, double *C,
       const FInt *ldC, size_t lenTransA, size_t lenTransB);
void
dgesv_(const FInt *n, const FInt *nrhs, double *A, const FInt *ldA, FInt *ipiv,
       double *B, const FInt *ldB, FInt *info);
void
dposv_(const char *uplo, const FInt *n, const FInt *nrhs, double *A,
       const FInt *ldA, double *B, const FInt *ldB, FInt *info, size_t lenUplo);
void
dgetrf_(const FInt *m, const FInt *n, double *A, const FInt *ldA, FInt *ipiv,
        FInt *info);
void
dpotrf_(const char *uplo, const FInt *n, double *A, const FInt *ldA, FInt *info,
        size_t lenUplo);

void
flexiblas_set_num_threads(int num);
int
flexiblas_get_num_threads(void);

enum
{
    N = 200
};
static double A[N * N], B[N * N], C[N * N], M[N * N], x[N];
static FInt ipiv[N];

/*  the XERBLA of the program: records the last call */
static char xerblaName[8];
static FInt xerblaInfo;
static int xerblaCalls;

void
xerbla_(const char *srname, const FInt *info, size_t lenSrname)
{
    size_t len =
        lenSrname < sizeof(xerblaName) - 1 ? lenSrname : sizeof(xerblaName) - 1;
    memcpy(xerblaName, srname, len);
    xerblaName[len] = 0;
    while (len > 0 && xerblaName[len - 1] == ' ') {
	xerblaName[--len] = 0;
    }
    xerblaInfo = *info;
    ++xerblaCalls;
}

/*  1 if the last call of xerbla_ (since the last check) had name and info */
static int
xerblaCalled(const char *what, const char *name, FInt info)
{
    int ok =
        xerblaCalls == 1 && strcmp(xerblaName, name) == 0 && xerblaInfo == info;
    printf("xerbla, %s: %d call(s), %s %lld: %s\n", what, xerblaCalls,
           xerblaName, (long long)xerblaInfo, ok ? "ok" : "FAIL");
    xerblaCalls = 0;
    xerblaName[0] = 0;
    xerblaInfo = 0;
    return ok;
}

static double
maxError(const double *v, ptrdiff_t n)
{
    double err = 0;
    for (ptrdiff_t i = 0; i < n; ++i) {
	double e = v[i] - 1;
	err = e > err ? e : (-e > err ? -e : err);
    }
    return err;
}

/*  symmetric, diagonally dominant M, x = M * ones */
static void
makeSystem(void)
{
    for (ptrdiff_t j = 0; j < N; ++j) {
	x[j] = 0;
	for (ptrdiff_t i = 0; i < N; ++i) {
	    M[i + j * N] = (i == j) ? 2.0 * N : (double)((i + j) % 3) - 1;
	}
    }
    for (ptrdiff_t j = 0; j < N; ++j) {
	for (ptrdiff_t i = 0; i < N; ++i) {
	    x[i] += M[i + j * N];
	}
    }
}

int
main(void)
{
    int ok = 1;

    for (ptrdiff_t i = 0; i < N * N; ++i) {
	A[i] = (double)(i % 7) - 3;
	B[i] = (double)(i % 5) - 2;
    }
    FInt n = N, one = 1, info = -1;
    double alpha = 1, beta = 0;
    dgemm_("N", "N", &n, &n, &n, &alpha, A, &n, B, &n, &beta, C, &n, 1, 1);
    double err = 0;
    for (ptrdiff_t j = 0; j < N; ++j) {
	for (ptrdiff_t i = 0; i < N; ++i) {
	    double s = 0;
	    for (ptrdiff_t l = 0; l < N; ++l) {
		s += A[i + l * N] * B[l + j * N];
	    }
	    double e = s - C[i + j * N];
	    err = e > err ? e : (-e > err ? -e : err);
	}
    }
    printf("dgemm: max error %g\n", err);
    ok = ok && err == 0;

    makeSystem();
    dgesv_(&n, &one, M, &n, ipiv, x, &n, &info);
    err = maxError(x, N);
    printf("dgesv: info %lld, max error %g\n", (long long)info, err);
    ok = ok && info == 0 && err < 1e-12;

    makeSystem();
    info = -1;
    dposv_("L", &n, &one, M, &n, x, &n, &info, 1);
    err = maxError(x, N);
    printf("dposv: info %lld, max error %g\n", (long long)info, err);
    ok = ok && info == 0 && err < 1e-12;

    int loaded = 0;
    FILE *maps = fopen("/proc/self/maps", "r");
    char line[4096];
    while (maps && fgets(line, sizeof(line), maps)) {
	if (strstr(line, "ulmblas")) {
	    const char *path = strchr(line, '/');
#if defined(__x86_64__) || defined(_M_X64)
	    //  the dispatching libulmblas.so loads a backend libulmblas_ISA.so
	    if (strstr(line, "libulmblas_")) {
		loaded = 1;
	    }
#else
	    //  one library with the kernels (no dispatch)
	    loaded = 1;
#endif
	    printf("    %s", path ? path : line);
	}
    }
    if (maps) {
	fclose(maps);
    }
    printf("ulmBLAS backend loaded: %s\n", loaded ? "yes" : "no");
    ok = ok && loaded;

    //  the threads of ulmBLAS, from the library FlexiBLAS loaded
    void *ulmblas = dlopen(ULMBLAS_LIBRARY, RTLD_LAZY | RTLD_NOLOAD);
    int (*getThreads)(void) =
        ulmblas ? (int (*)(void))dlsym(ulmblas, "ulmblas_get_num_threads")
	        : NULL;
    if (!getThreads) {
	printf("threads: ulmblas_get_num_threads not found: FAIL\n");
	ok = 0;
    } else {
	for (int t = 2; t >= 1; --t) {
	    flexiblas_set_num_threads(t);
	    int got = getThreads();
	    printf("threads: flexiblas_set_num_threads(%d), ulmBLAS %d, "
	           "flexiblas_get_num_threads %d: %s\n",
	           t, got, flexiblas_get_num_threads(),
	           got == t ? "ok" : "FAIL");
	    ok = ok && got == t;
	}
    }

    //  invalid arguments
    xerblaCalls = 0;
    dgemm_("X", "N", &n, &n, &n, &alpha, A, &n, B, &n, &beta, C, &n, 1, 1);
    ok = xerblaCalled("dgemm", "DGEMM", 1) && ok;
    FInt minusOne = -1;
    dgetrf_(&minusOne, &n, M, &n, ipiv, &info);
    ok = xerblaCalled("dgetrf", "DGETRF", 1) && ok;
    dpotrf_("X", &n, M, &n, &info, 1);
    ok = xerblaCalled("dpotrf", "DPOTRF", 1) && ok;

#if defined(FLEXIBLAS64)
    //  64-bit integers: leading dimensions 2^32 are valid (m = n = k = 0)
    {
	FInt zero = 0, big = (FInt)1 << 32;
	dgemm_("N", "N", &zero, &zero, &zero, &alpha, A, &big, B, &big, &beta,
	       C, &big, 1, 1);
	printf("dgemm, leading dimensions 2^32: %s\n",
	       xerblaCalls == 0 ? "ok" : "FAIL");
	ok = ok && xerblaCalls == 0;
    }
#endif

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
