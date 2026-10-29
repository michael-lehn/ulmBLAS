#ifndef INTERFACE_BLAS_F77_XERBLA_H
#define INTERFACE_BLAS_F77_XERBLA_H 1

#include <cstddef>

extern "C" {

//
//  SUBROUTINE XERBLA( SRNAME, INFO ) with CHARACTER*(*) SRNAME: the length
//  of SRNAME follows the other arguments as a hidden argument (size_t with
//  gfortran 8 or newer and flang).  A XERBLA supplied by the user (e.g. the
//  one of the Reference BLAS) reads the name with this length.
//
void
F77BLAS(xerbla)(const char                *rout,
                const ULMBLAS_INDEX_TYPE  *info,
                std::size_t               len);

} // extern "C"

#endif // INTERFACE_BLAS_F77_XERBLA_H
