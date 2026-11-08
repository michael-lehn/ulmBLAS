//
//  Must NOT compile: the LAPACK functions of ulmBLAS reject integer
//  element types (static_assert in ulmblas/lapack, see islapacktype.h).
//  run-tests.sh expects a compile error with the message of the
//  static_assert for getrf and for potrf.
//
#include <ulmblas/ulmblas.h>
#include <ulmblas/lapack/getrf.h>
#include <ulmblas/lapack/potrf.h>

int
main()
{
    int  A[4] = { 4, 2, 2, 3 };
    long piv[2];
    ulmBLAS::getrf(2L, 2L, A, 1L, 2L, piv, 1L);
    ulmBLAS::potrf(true, 2L, A, 1L, 2L);
    return 0;
}
