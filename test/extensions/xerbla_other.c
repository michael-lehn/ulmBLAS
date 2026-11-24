/*
 *  Another library with error handlers (xerbla_, cblas_xerbla), loaded
 *  before the library under test, must not take over its error handling:
 *  an invalid argument must reach the handler the program sets
 *  (set_blas_xerbla_, cblas_set_xerbla), not the other library's.  As in a
 *  Julia process (OpenBLAS is loaded first, ulmBLAS later with dlopen):
 *  the program is linked with libxerbla_other (xerbla_other_lib.c) and
 *  loads the library under test (argument: its path) with dlopen.
 *
 *      xerbla_other LIBRARY
 *
 *  Exit status 1 on failure.  Used by check-shared.sh on macOS (on Linux a
 *  library loaded before takes precedence, as the ELF symbol lookup).
 */
#include <cblas.h>
#include <dlfcn.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>

typedef void (*f77_xerbla_t)(const char *srname, ULMBLAS_INDEX_TYPE *info,
                             size_t lenSrname);
typedef void (*dgemm_t)(const char *transA, const char *transB,
                        const ULMBLAS_INDEX_TYPE *m,
                        const ULMBLAS_INDEX_TYPE *n,
                        const ULMBLAS_INDEX_TYPE *k, const double *alpha,
                        const double *A, const ULMBLAS_INDEX_TYPE *ldA,
                        const double *B, const ULMBLAS_INDEX_TYPE *ldB,
                        const double *beta, double *C,
                        const ULMBLAS_INDEX_TYPE *ldC, size_t lenTransA,
                        size_t lenTransB);
typedef void (*set_blas_xerbla_t)(f77_xerbla_t handler);

/*  calls of the handlers of libxerbla_other */
int
xerbla_other_calls(void);

static int f77Calls, cCalls;
static ptrdiff_t lastInfo;

static void
f77Handler(const char *srname, ULMBLAS_INDEX_TYPE *info, size_t lenSrname)
{
    (void)srname;
    (void)lenSrname;
    ++f77Calls;
    lastInfo = *info;
}

static void
cHandler(ULMBLAS_INDEX_TYPE info, const char *rout, const char *form,
         va_list ap)
{
    (void)rout;
    (void)form;
    (void)ap;
    ++cCalls;
    lastInfo = info;
}

static void *
sym(void *h, const char *name)
{
    void *p = dlsym(h, name);
    if (!p) {
	printf("FAIL  %s not found\n", name);
    }
    return p;
}

int
main(int argc, char **argv)
{
    if (argc != 2) {
	printf("FAIL  usage: xerbla_other LIBRARY\n");
	return 1;
    }
    void *h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) {
	printf("FAIL  %s\n", dlerror());
	return 1;
    }
    set_blas_xerbla_t setF77 = (set_blas_xerbla_t)sym(h, "set_blas_xerbla_");
    dgemm_t dgemm = (dgemm_t)sym(h, "dgemm_");
    __typeof__(&cblas_set_xerbla) setC =
        (__typeof__(&cblas_set_xerbla))sym(h, "cblas_set_xerbla");
    __typeof__(&cblas_dgemm) cdgemm =
        (__typeof__(&cblas_dgemm))sym(h, "cblas_dgemm");
    if (!setF77 || !dgemm || !setC || !cdgemm) {
	return 1;
    }

    int failures = 0;
    double A[16] = {0}, C[16] = {0};
    const ULMBLAS_INDEX_TYPE n4 = 4, ld1 = 1;
    const double one = 1, zero = 0;

    setF77(f77Handler);
    dgemm("N", "N", &n4, &n4, &n4, &one, A, &ld1, A, &n4, &zero, C, &n4, 1, 1);
    if (f77Calls != 1 || lastInfo != 8 || xerbla_other_calls() != 0) {
	printf("FAIL  dgemm_ ldA=1: handler of the program %d (info %ld), "
	       "of the other library %d\n",
	       f77Calls, (long)lastInfo, xerbla_other_calls());
	++failures;
    }
    setC(cHandler, NULL);
    cdgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 4, -1, 4, 1.0, A, 4, A, 4,
           0.0, C, 4);
    if (cCalls != 1 || lastInfo != 5 || xerbla_other_calls() != 0) {
	printf("FAIL  cblas_dgemm n=-1: handler of the program %d (info %ld), "
	       "of the other library %d\n",
	       cCalls, (long)lastInfo, xerbla_other_calls());
	++failures;
    }
    if (failures == 0) {
	printf("PASS  errors go to the handlers of the program, not to those "
	       "of a library loaded before\n");
    }
    return failures != 0;
}
