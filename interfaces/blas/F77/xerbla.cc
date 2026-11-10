#include BLAS_HEADER
#include <interfaces/blas/F77/xerbla.h>
#include <cstdio>

extern "C" {

//
//  In the shared library (interfaces/blas/shared) the error handler is a
//  weak definition, so that a program can replace it (the Netlib test
//  programs do): on macOS calls inside a library are otherwise bound to
//  its own definition.
//
#if defined(ULMBLAS_REPLACEABLE_XERBLA)
#define ULMBLAS_XERBLA_ATTRIBUTE __attribute__((weak))
#else
#define ULMBLAS_XERBLA_ATTRIBUTE
#endif

ULMBLAS_XERBLA_ATTRIBUTE void
F77BLAS(xerbla)(const char *rout, const ULMBLAS_INDEX_TYPE *info,
                std::size_t len)
{
    fprintf(stderr, "Parameter %lld to routine %.*s was incorrect\n",
            (long long)*info, int(len), rout);
}

} // extern "C"
