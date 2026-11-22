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
void F77BLAS(xerbla)(const char *rout, const ULMBLAS_INDEX_TYPE *info,
                     std::size_t len);

//
//  SUBROUTINE XERBLA_ARRAY( SRNAME_ARRAY, SRNAME_LEN, INFO ) of the
//  Reference BLAS, for callers that have the name as an array of
//  characters (e.g. C): calls XERBLA with the first min(SRNAME_LEN, 32)
//  characters.  The hidden length (of one element) follows.
//
void F77BLAS(xerbla_array)(const char *srname_array,
                           const ULMBLAS_INDEX_TYPE *srname_len,
                           const ULMBLAS_INDEX_TYPE *info, std::size_t len);

//
//  SET_BLAS_XERBLA( CB ), GET_BLAS_XERBLA( CB ) of the Netlib BLAS: the
//  handler that XERBLA calls (blas_fortran_xerbla_callback_t of the Netlib
//  cblas.h).  A procedure is passed by its address, a procedure pointer
//  (GET_BLAS_XERBLA) by the address of the pointer.
//
typedef void (*ulmblas_f77_xerbla_callback_t)(const char *srname,
                                              ULMBLAS_INDEX_TYPE *info,
                                              std::size_t lenSrname);

void F77BLAS(set_blas_xerbla)(ulmblas_f77_xerbla_callback_t handler);

void F77BLAS(get_blas_xerbla)(ulmblas_f77_xerbla_callback_t *handler);

} // extern "C"

#endif // INTERFACE_BLAS_F77_XERBLA_H
