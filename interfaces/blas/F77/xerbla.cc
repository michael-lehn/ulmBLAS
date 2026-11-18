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
//  -Wl,--export-all-symbols).  A library with two Fortran 77 interfaces
//  (../shared: LP64 xerbla_ and ILP64 xerbla_64_) has a setter per
//  interface: ULMBLAS_SET_F77_XERBLA names the one of this build.
//
#if defined(ULMBLAS_REPLACEABLE_XERBLA)
#ifndef ULMBLAS_SET_F77_XERBLA
#define ULMBLAS_SET_F77_XERBLA ulmblas_set_f77_xerbla
#endif
#define ULMBLAS_XERBLA_STR_(name) #name
#define ULMBLAS_XERBLA_STR(name) ULMBLAS_XERBLA_STR_(name)

#if defined(_WIN32)
#define ULMBLAS_XERBLA_ATTRIBUTE
#else
#define ULMBLAS_XERBLA_ATTRIBUTE __attribute__((weak))
#endif

typedef void (*ulmblas_f77_xerbla_t)(const char *srname,
                                     const ULMBLAS_INDEX_TYPE *info,
                                     std::size_t lenSrname);

static ulmblas_f77_xerbla_t f77_xerbla_replacement = nullptr;

void
ULMBLAS_SET_F77_XERBLA(ulmblas_f77_xerbla_t handler)
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
	handler =
	    reinterpret_cast<ulmblas_f77_xerbla_t>(reinterpret_cast<void *>(
	        GetProcAddress(GetModuleHandleW(nullptr),
		               ULMBLAS_XERBLA_STR(F77BLAS(xerbla)))));
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

void
F77BLAS(xerbla_array)(const char *srname_array,
                      const ULMBLAS_INDEX_TYPE *srname_len,
                      const ULMBLAS_INDEX_TYPE *info, std::size_t len)
{
    char srname[32];
    std::size_t n = 0;
    while (n < sizeof(srname) && static_cast<ULMBLAS_INDEX_TYPE>(n) < *srname_len) {
	srname[n] = srname_array[n];
	++n;
    }
    F77BLAS(xerbla)(srname, info, n);
}

} // extern "C"
