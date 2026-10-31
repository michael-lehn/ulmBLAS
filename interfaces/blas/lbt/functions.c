/*
 *  The functions of the Fortran 77 interface (DDOT, DZNRM2, IDAMAX, ...)
 *  in C, for the shared library of this directory.  In ../F77 they are
 *  Fortran sources; here they are C, so that the library needs neither a
 *  Fortran compiler nor libgfortran, and the integer type is the one of
 *  the build (ULMBLAS_INDEX_TYPE).  Each function calls the subroutine
 *  with the suffix _sub, which returns the result in its last argument.
 *
 *  The return values follow gfortran: REAL as float, COMPLEX and DOUBLE
 *  COMPLEX as C complex types (in registers).
 *
 *  cdotc_sub and cdotu_sub declare their result as double *, but store
 *  two floats there (see ../F77/dot_sub.cc), hence the cast to void *.
 */
#include <complex.h>
#include BLAS_HEADER

#define REAL_FUNCTION(T, name, sub, args, call)                             \
T F77BLAS(name) args                                                        \
{                                                                           \
    T result;                                                               \
    F77BLAS(sub) call;                                                      \
    return result;                                                          \
}

#define COMPLEX_FUNCTION(C, T, name, sub)                                   \
C F77BLAS(name)(const ULMBLAS_INDEX_TYPE *n, const T *x,                    \
                const ULMBLAS_INDEX_TYPE *incX, const T *y,                 \
                const ULMBLAS_INDEX_TYPE *incY)                             \
{                                                                           \
    T result[2];                                                            \
    F77BLAS(sub)(n, x, incX, y, incY, (void *)result);                      \
    return result[0] + I*result[1];                                         \
}

#define NX      const ULMBLAS_INDEX_TYPE *n, const T_ *x,                   \
                const ULMBLAS_INDEX_TYPE *incX
#define NXY     NX, const T_ *y, const ULMBLAS_INDEX_TYPE *incY

/*  Level 1 functions of float and complex float */
#define T_ float
REAL_FUNCTION(float, sasum,  sasum_sub,  (NX),  (n, x, incX, &result))
REAL_FUNCTION(float, scasum, scasum_sub, (NX),  (n, x, incX, &result))
REAL_FUNCTION(float, snrm2,  snrm2_sub,  (NX),  (n, x, incX, &result))
REAL_FUNCTION(float, scnrm2, scnrm2_sub, (NX),  (n, x, incX, &result))
REAL_FUNCTION(float, sdot,   sdot_sub,   (NXY), (n, x, incX, y, incY, &result))
REAL_FUNCTION(double, dsdot, dsdot_sub,  (NXY), (n, x, incX, y, incY, &result))
REAL_FUNCTION(float, scabs1, scabs1_sub, (const float *z), (z, &result))
REAL_FUNCTION(ULMBLAS_INDEX_TYPE, isamax, isamax_sub, (NX),
              (n, x, incX, &result))
REAL_FUNCTION(ULMBLAS_INDEX_TYPE, icamax, icamax_sub, (NX),
              (n, x, incX, &result))

float
F77BLAS(sdsdot)(const ULMBLAS_INDEX_TYPE *n, const float *alpha,
                const float *x, const ULMBLAS_INDEX_TYPE *incX,
                const float *y, const ULMBLAS_INDEX_TYPE *incY)
{
    float result;
    F77BLAS(sdsdot_sub)(n, alpha, x, incX, y, incY, &result);
    return result;
}
#undef T_

/*  Level 1 functions of double and complex double */
#define T_ double
REAL_FUNCTION(double, dasum,  dasum_sub,  (NX),  (n, x, incX, &result))
REAL_FUNCTION(double, dzasum, dzasum_sub, (NX),  (n, x, incX, &result))
REAL_FUNCTION(double, dnrm2,  dnrm2_sub,  (NX),  (n, x, incX, &result))
REAL_FUNCTION(double, dznrm2, dznrm2_sub, (NX),  (n, x, incX, &result))
REAL_FUNCTION(double, ddot,   ddot_sub,   (NXY), (n, x, incX, y, incY, &result))
REAL_FUNCTION(double, dcabs1, dcabs1_sub, (const double *z), (z, &result))
REAL_FUNCTION(ULMBLAS_INDEX_TYPE, idamax, idamax_sub, (NX),
              (n, x, incX, &result))
REAL_FUNCTION(ULMBLAS_INDEX_TYPE, izamax, izamax_sub, (NX),
              (n, x, incX, &result))
#undef T_

COMPLEX_FUNCTION(float complex,  float,  cdotc, cdotc_sub)
COMPLEX_FUNCTION(float complex,  float,  cdotu, cdotu_sub)
COMPLEX_FUNCTION(double complex, double, zdotc, zdotc_sub)
COMPLEX_FUNCTION(double complex, double, zdotu, zdotu_sub)
