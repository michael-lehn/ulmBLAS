#include BLAS_HEADER
#include <interfaces/blas/F77/xerbla.h>
#include <cstdio>

#if defined(ULMBLAS_REPLACEABLE_XERBLA) && defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

extern "C" {

//
//  In the shared library (interfaces/blas/shared) the error handler is a
//  weak definition, so that a program can replace it (the Netlib test
//  programs do): on macOS calls inside a library are otherwise bound to
//  its own definition.  A library that loads this one at run time (the
//  dispatching library, interfaces/blas/dispatch) passes the handler of
//  the program with ulmblas_set_f77_xerbla: the weak definition is not
//  replaced then.  On Windows a DLL always calls its own definition; the
//  handler of the program is used if the program exports it (MinGW:
//  -Wl,--export-all-symbols).
//
#if defined(ULMBLAS_REPLACEABLE_XERBLA)
#if defined(_WIN32)
#define ULMBLAS_XERBLA_ATTRIBUTE
#else
#define ULMBLAS_XERBLA_ATTRIBUTE __attribute__((weak))
#endif

typedef void (*ulmblas_f77_xerbla_t)(const char *, const ULMBLAS_INDEX_TYPE *,
                                     std::size_t);

static ulmblas_f77_xerbla_t f77_xerbla_replacement = nullptr;

void
ulmblas_set_f77_xerbla(ulmblas_f77_xerbla_t handler)
{
    f77_xerbla_replacement = handler;
}
#else
#define ULMBLAS_XERBLA_ATTRIBUTE
#endif

ULMBLAS_XERBLA_ATTRIBUTE void
F77BLAS(xerbla)(const char *rout, const ULMBLAS_INDEX_TYPE *info,
                std::size_t len)
{
#if defined(ULMBLAS_REPLACEABLE_XERBLA)
    ulmblas_f77_xerbla_t handler = f77_xerbla_replacement;
#if defined(_WIN32)
    if (!handler) {
	handler = reinterpret_cast<ulmblas_f77_xerbla_t>(
	    reinterpret_cast<void *>(
	        GetProcAddress(GetModuleHandleW(nullptr), "xerbla_")));
    }
#endif
    if (handler) {
	handler(rout, info, len);
	return;
    }
#endif
    fprintf(stderr, "Parameter %lld to routine %.*s was incorrect\n",
            (long long)*info, int(len), rout);
}

} // extern "C"
