/*
 *  LSAME of the reference BLAS: true if the characters CA and CB are the
 *  same letter regardless of case.  Not a BLAS routine of ulmBLAS, but
 *  part of every reference BLAS library: LAPACK built against an external
 *  BLAS calls it.  As the reference, only the first character counts; the
 *  hidden lengths of the CHARACTER arguments are not used.  LOGICAL is
 *  int (gfortran default).  ULMBLAS_LSAME and ULMBLAS_LOGICAL: name and
 *  result type, lsame_64_ and int64_t for the ILP64 interface (the
 *  reference compiles it with -fdefault-integer-8, which makes LOGICAL 8
 *  bytes wide as well).
 */
#include <stdint.h>

#ifndef ULMBLAS_LSAME
#define ULMBLAS_LSAME lsame_
#endif
#ifndef ULMBLAS_LOGICAL
#define ULMBLAS_LOGICAL int
#endif

ULMBLAS_LOGICAL
ULMBLAS_LSAME(const char *ca, const char *cb)
{
    int a = (unsigned char)*ca, b = (unsigned char)*cb;

    if (a >= 'a' && a <= 'z') {
	a -= 'a' - 'A';
    }
    if (b >= 'a' && b <= 'z') {
	b -= 'a' - 'A';
    }
    return a == b;
}
