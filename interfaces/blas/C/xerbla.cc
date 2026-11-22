#include BLAS_HEADER
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <interfaces/blas/C/xerbla.h>

#if defined(ULMBLAS_REPLACEABLE_XERBLA) && defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

extern "C" {

//
//  The error handler is a weak definition (as in ../F77/xerbla.cc), so that
//  a program can replace it (the Netlib test programs do): in the shared
//  library (interfaces/blas/shared) on macOS calls inside a library are
//  otherwise bound to its own definition; in the static library this
//  object is also linked if the program calls cblas_set_xerbla.  On
//  Windows see ../F77/xerbla.cc.
//
#if defined(_WIN32)
#define ULMBLAS_XERBLA_ATTRIBUTE
#else
#define ULMBLAS_XERBLA_ATTRIBUTE __attribute__((weak))
#endif

//
//  Replaceable handler as in ../F77/xerbla.cc; ULMBLAS_SET_CBLAS_XERBLA
//  names the setter of this build (the ILP64 interface of ../shared,
//  cblas_xerbla_64, has its own).
//
#if defined(ULMBLAS_REPLACEABLE_XERBLA)
#ifndef ULMBLAS_SET_CBLAS_XERBLA
#define ULMBLAS_SET_CBLAS_XERBLA ulmblas_set_cblas_xerbla
#endif
#define ULMBLAS_XERBLA_STR_(name) #name
#define ULMBLAS_XERBLA_STR(name) ULMBLAS_XERBLA_STR_(name)

typedef void (*ulmblas_cblas_xerbla_t)(ULMBLAS_INDEX_TYPE info,
                                       const char *rout, const char *form, ...);

static ulmblas_cblas_xerbla_t cblas_xerbla_replacement = nullptr;

//  the handler of the program, passed by the dispatching library (see
//  ../F77/xerbla.cc)
void
ULMBLAS_SET_CBLAS_XERBLA(ulmblas_cblas_xerbla_t handler)
{
    cblas_xerbla_replacement = handler;
}
#endif

//
//  The handler set at run time (cblas_set_xerbla, as the Netlib CBLAS):
//  cblas_xerbla calls it with the position of the parameter in the call of
//  the CBLAS function (for row major calls after the translation below)
//  instead of printing.  A replacement of cblas_xerbla (a program that
//  defines it, see above) takes precedence: it replaces this function, and
//  with it the call of the handler.
//
static std::atomic<CBLAS_XERBLA_CALLBACK_T> cblas_xerbla_callback(nullptr);

//  f77Handler is not used: the functions of this interface report errors
//  only through cblas_xerbla, never through the Fortran 77 xerbla_ (the
//  Netlib CBLAS installs it as the handler of its Fortran BLAS); the
//  handler of xerbla_ is set with set_blas_xerbla_ (../F77/xerbla.cc).
CBLAS_XERBLA_CALLBACK_T
CBLAS(set_xerbla)(CBLAS_XERBLA_CALLBACK_T handler,
                  BLAS_FORTRAN_XERBLA_CALLBACK_T f77Handler)
{
    (void)f77Handler;
    return cblas_xerbla_callback.exchange(handler);
}

CBLAS_XERBLA_CALLBACK_T
CBLAS(get_xerbla)(void)
{
    return cblas_xerbla_callback.load();
}

void
ULMBLAS(xerbla)(int info, const char *rout, const char *form, ...)
{
    fprintf(stderr, "Parameter %d to routine %s was incorrect\n", info, rout);
}

//
//  For row major calls the wrappers report the position of a parameter in
//  the call of the column major routine (as the reference CBLAS).  As
//  cblas_xerbla of the reference CBLAS, the position in the call of the
//  CBLAS function is printed.  info has the index type (as CBLAS_INT of the
//  reference CBLAS: int64_t for cblas_xerbla_64).
//
ULMBLAS_XERBLA_ATTRIBUTE void
CBLAS(xerbla)(ULMBLAS_INDEX_TYPE info, const char *rout, const char *form, ...)
{
    extern int RowMajorStrg;

#if defined(ULMBLAS_REPLACEABLE_XERBLA)
    //  the interface passes no arguments after form
    ulmblas_cblas_xerbla_t handler = cblas_xerbla_replacement;
#if defined(_WIN32)
    if (!handler) {
	handler = reinterpret_cast<ulmblas_cblas_xerbla_t>(
	    reinterpret_cast<void *>(GetProcAddress(
	        GetModuleHandleW(nullptr), ULMBLAS_XERBLA_STR(CBLAS(xerbla)))));
    }
#endif
    if (handler) {
	handler(info, rout, form);
	return;
    }
#endif

    if (RowMajorStrg) {
	//  gemmtr first: its name contains "gemm" (row major swaps only A
	//  and B, as the reference)
	if (std::strstr(rout, "gemmtr")) {
	    if (info == 11)
		info = 9;
	    else if (info == 9)
		info = 11;
	} else if (std::strstr(rout, "gemm")) {
	    if (info == 5)
		info = 4;
	    else if (info == 4)
		info = 5;
	    else if (info == 11)
		info = 9;
	    else if (info == 9)
		info = 11;
	} else if (std::strstr(rout, "symm") || std::strstr(rout, "hemm")) {
	    if (info == 5)
		info = 4;
	    else if (info == 4)
		info = 5;
	} else if (std::strstr(rout, "trmm") || std::strstr(rout, "trsm")) {
	    if (info == 7)
		info = 6;
	    else if (info == 6)
		info = 7;
	} else if (std::strstr(rout, "gemv")) {
	    if (info == 4)
		info = 3;
	    else if (info == 3)
		info = 4;
	} else if (std::strstr(rout, "gbmv")) {
	    if (info == 4)
		info = 3;
	    else if (info == 3)
		info = 4;
	    else if (info == 6)
		info = 5;
	    else if (info == 5)
		info = 6;
	} else if (std::strstr(rout, "ger")) {
	    if (info == 3)
		info = 2;
	    else if (info == 2)
		info = 3;
	    else if (info == 8)
		info = 6;
	    else if (info == 6)
		info = 8;
	} else if ((std::strstr(rout, "her2") || std::strstr(rout, "hpr2")) &&
	           !std::strstr(rout, "her2k")) {
	    if (info == 8)
		info = 6;
	    else if (info == 6)
		info = 8;
	}
    }
    CBLAS_XERBLA_CALLBACK_T callback = cblas_xerbla_callback.load();
    if (callback) {
	va_list ap;
	va_start(ap, form);
	callback(info, rout, form, ap);
	va_end(ap);
	return;
    }
    fprintf(stderr, "Parameter %lld to routine %s was incorrect\n",
            static_cast<long long>(info), rout);
}

} // extern "C"
