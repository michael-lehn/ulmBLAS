#include <interfaces/blas/C/xerbla.h>
#include <cstdio>
#include <cstring>

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
//  its own definition.  On Windows see ../F77/xerbla.cc.
//
#if defined(ULMBLAS_REPLACEABLE_XERBLA)
#if defined(_WIN32)
#define ULMBLAS_XERBLA_ATTRIBUTE
#else
#define ULMBLAS_XERBLA_ATTRIBUTE __attribute__((weak))
#endif

typedef void (*ulmblas_cblas_xerbla_t)(int, const char *, const char *, ...);

static ulmblas_cblas_xerbla_t cblas_xerbla_replacement = nullptr;

//  the handler of the program, passed by the dispatching library (see
//  ../F77/xerbla.cc)
void
ulmblas_set_cblas_xerbla(ulmblas_cblas_xerbla_t handler)
{
    cblas_xerbla_replacement = handler;
}
#else
#define ULMBLAS_XERBLA_ATTRIBUTE
#endif

void
ULMBLAS(xerbla)(int info, const char *rout, const char *form, ...)
{
    fprintf(stderr, "Parameter %d to routine %s was incorrect\n", info, rout);
}

//
//  For row major calls the wrappers report the position of a parameter in
//  the call of the column major routine (as the reference CBLAS).  As
//  cblas_xerbla of the reference CBLAS, the position in the call of the
//  CBLAS function is printed.
//
ULMBLAS_XERBLA_ATTRIBUTE void
CBLAS(xerbla)(int info, const char *rout, const char *form, ...)
{
    extern int RowMajorStrg;

#if defined(ULMBLAS_REPLACEABLE_XERBLA)
    //  the interface passes no arguments after form
    ulmblas_cblas_xerbla_t handler = cblas_xerbla_replacement;
#if defined(_WIN32)
    if (!handler) {
	handler = reinterpret_cast<ulmblas_cblas_xerbla_t>(
	    reinterpret_cast<void *>(
	        GetProcAddress(GetModuleHandleW(nullptr), "cblas_xerbla")));
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
    fprintf(stderr, "Parameter %d to routine %s was incorrect\n", info, rout);
}

} // extern "C"
