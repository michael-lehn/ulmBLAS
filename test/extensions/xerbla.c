/*
 *  Error handlers set at run time (as LAPACK's Reference-LAPACK/lapack#1407):
 *
 *  - cblas_set_xerbla, cblas_get_xerbla: the previous handler is returned,
 *    NULL restores the default; cblas_xerbla calls the handler with the
 *    position of the parameter in the call of the CBLAS function (for row
 *    major calls after the translation: cblas_dgemm with n < 0 reports 5
 *    in both orders), the name and form with a va_list;
 *  - set_blas_xerbla_, get_blas_xerbla_: xerbla_ calls the handler with the
 *    name (blank padded, with its length) and INFO; get_blas_xerbla_
 *    returns the default if no handler is set, never NULL.
 *
 *  The program does not define cblas_xerbla or xerbla_ (a definition would
 *  replace the library's, and with it the call of the handler).  Exit
 *  status 1 on failure.  Used by check-shared.sh.
 */
#include <cblas.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

typedef void (*f77_xerbla_t)(const char *srname, ULMBLAS_INDEX_TYPE *info,
                             size_t lenSrname);

void
dgemv_(const char *trans, const ULMBLAS_INDEX_TYPE *m,
       const ULMBLAS_INDEX_TYPE *n, const double *alpha, const double *A,
       const ULMBLAS_INDEX_TYPE *ldA, const double *x,
       const ULMBLAS_INDEX_TYPE *incX, const double *beta, double *y,
       const ULMBLAS_INDEX_TYPE *incY, size_t lenTrans);
void
set_blas_xerbla_(f77_xerbla_t handler);
void
get_blas_xerbla_(f77_xerbla_t *handler);

static ptrdiff_t failures = 0;

static void
check(ptrdiff_t ok, const char *what)
{
    if (!ok) {
	printf("FAIL  %s\n", what);
	++failures;
    }
}

static ptrdiff_t calls;
static ptrdiff_t lastInfo;
static char lastName[32];
static char lastForm[64];

static void
reset(void)
{
    calls = 0;
    lastInfo = 0;
    lastName[0] = lastForm[0] = '\0';
}

static void
handler(ULMBLAS_INDEX_TYPE info, const char *rout, const char *form, va_list ap)
{
    ++calls;
    lastInfo = (ptrdiff_t)info;
    snprintf(lastName, sizeof(lastName), "%s", rout);
    vsnprintf(lastForm, sizeof(lastForm), form, ap);
}

static void
other(ULMBLAS_INDEX_TYPE info, const char *rout, const char *form, va_list ap)
{
    (void)info;
    (void)rout;
    (void)form;
    (void)ap;
}

static void
f77Handler(const char *srname, ULMBLAS_INDEX_TYPE *info, size_t lenSrname)
{
    ++calls;
    lastInfo = (ptrdiff_t)*info;
    snprintf(lastName, sizeof(lastName), "%.*s",
             (int)(lenSrname < 31 ? lenSrname : 31), srname);
}

int
main(void)
{
    double A[16] = {0}, B[16] = {0}, C[16] = {0}, x[4] = {0}, y[4] = {0};
    const double one = 1, zero = 0;

    /* C interface */
    check(cblas_get_xerbla() == NULL, "cblas_get_xerbla: NULL at start");
    check(cblas_set_xerbla(handler, NULL) == NULL,
          "cblas_set_xerbla: returns NULL (no previous handler)");
    check(cblas_get_xerbla() == handler, "cblas_get_xerbla: the handler");

    reset();
    cblas_dgemv(CblasColMajor, CblasNoTrans, 4, 4, 1.0, A, 1, x, 1, 0.0, y, 1);
    check(calls == 1, "cblas_dgemv ldA=1: handler called once");
    check(lastInfo == 7, "cblas_dgemv ldA=1: info 7");
    check(strcmp(lastName, "cblas_dgemv") == 0,
          "cblas_dgemv ldA=1: name cblas_dgemv");

    reset();
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 4, -1, 4, 1.0, A, 4,
                B, 4, 0.0, C, 4);
    check(calls == 1 && lastInfo == 5, "cblas_dgemm col major n=-1: info 5");
    reset();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 4, -1, 4, 1.0, A, 4,
                B, 4, 0.0, C, 4);
    check(calls == 1 && lastInfo == 5,
          "cblas_dgemm row major n=-1: info 5 (translated)");
    check(lastForm[0] == '\0', "cblas_dgemm: form \"\" (no details)");

    check(cblas_set_xerbla(other, NULL) == handler,
          "cblas_set_xerbla: returns the previous handler");
    check(cblas_get_xerbla() == other, "cblas_get_xerbla: the new handler");
    check(cblas_set_xerbla(NULL, NULL) == other,
          "cblas_set_xerbla(NULL): returns the previous handler");
    check(cblas_get_xerbla() == NULL, "cblas_get_xerbla: NULL after reset");

    /* Fortran 77 interface */
    f77_xerbla_t initial = NULL, current = NULL;
    get_blas_xerbla_(&initial);
    check(initial != NULL, "get_blas_xerbla_: the default, not NULL");

    set_blas_xerbla_(f77Handler);
    get_blas_xerbla_(&current);
    check(current == f77Handler, "get_blas_xerbla_: the handler");

    reset();
    const ULMBLAS_INDEX_TYPE n4 = 4, ld1 = 1, inc1 = 1;
    dgemv_("N", &n4, &n4, &one, A, &ld1, x, &inc1, &zero, y, &inc1, 1);
    check(calls == 1, "dgemv_ ldA=1: handler called once");
    check(lastInfo == 6, "dgemv_ ldA=1: INFO 6");
    check(strncmp(lastName, "DGEMV", 5) == 0, "dgemv_ ldA=1: name DGEMV");

    /* the C handler is not called for errors of the Fortran interface */
    cblas_set_xerbla(handler, NULL);
    reset();
    dgemv_("N", &n4, &n4, &one, A, &ld1, x, &inc1, &zero, y, &inc1, 1);
    check(calls == 1 && lastInfo == 6 && strncmp(lastName, "DGEMV", 5) == 0,
          "dgemv_ with both handlers: only the Fortran handler");
    cblas_set_xerbla(NULL, NULL);

    set_blas_xerbla_(NULL);
    get_blas_xerbla_(&current);
    check(current == initial, "get_blas_xerbla_: the default after reset");

    /* the default prints and returns */
    ULMBLAS_INDEX_TYPE info = 3;
    initial("TEST  ", &info, 6);

    if (failures) {
	printf("%td failures\n", failures);
	return 1;
    }
    printf("PASS\n");
    return 0;
}
