#include <interfaces/blas/C/xerbla.h>
#include <cstdio>
#include <cstring>

extern "C" {

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
void
CBLAS(xerbla)(int info, const char *rout, const char *form, ...)
{
    extern int RowMajorStrg;

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
