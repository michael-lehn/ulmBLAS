/*
 *  libxerbla_other: a library with the error handlers of a BLAS (as
 *  OpenBLAS), for xerbla_other.c.  The handlers only count their calls.
 *
 *  With a weak definition, as OpenBLAS (___emutls_get_address of the GCC
 *  runtime): on macOS the dynamic loader binds the weak definitions of a
 *  library loaded later to definitions in libraries loaded before that
 *  have weak definitions themselves (header flag WEAK_DEFINES), there
 *  also to ordinary ones such as these handlers.
 */
#include <stddef.h>

__attribute__((weak)) int xerbla_other_weak = 0;

static int calls = 0;

void
xerbla_(const char *srname, const void *info, size_t lenSrname)
{
    (void)srname;
    (void)info;
    (void)lenSrname;
    ++calls;
}

void
xerbla_64_(const char *srname, const void *info, size_t lenSrname)
{
    (void)srname;
    (void)info;
    (void)lenSrname;
    ++calls;
}

void
cblas_xerbla(long info, const char *rout, const char *form, ...)
{
    (void)info;
    (void)rout;
    (void)form;
    ++calls;
}

int
xerbla_other_calls(void)
{
    return calls;
}
