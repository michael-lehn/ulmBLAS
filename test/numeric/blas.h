#ifndef ULMBLAS_TEST_NUMERIC_BLAS_H
#define ULMBLAS_TEST_NUMERIC_BLAS_H 1

//
//  Overloaded C++ wrappers of the C interface of ulmBLAS, so that the tests
//  can be written once for float, double, complex<float> and
//  complex<double>.  Complex scalars are passed by pointer as required by
//  the C interface.  Some functions of the interface take non-const
//  pointers to input arguments; const_cast is used for these.
//

#include BLAS_HEADER
#include <test/numeric/testing.h>

namespace blas {

using test::cfloat;
using test::cdouble;

inline float        *rp(float *p)          { return p; }
inline double       *rp(double *p)         { return p; }
inline float        *rp(cfloat *p)         { return reinterpret_cast<float *>(p); }
inline double       *rp(cdouble *p)        { return reinterpret_cast<double *>(p); }
inline const float  *rp(const float *p)    { return p; }
inline const double *rp(const double *p)   { return p; }
inline const float  *rp(const cfloat *p)   { return reinterpret_cast<const float *>(p); }
inline const double *rp(const cdouble *p)  { return reinterpret_cast<const double *>(p); }

template <typename T>
typename test::Real<T>::type *
mp(const T *p)
{
    return const_cast<typename test::Real<T>::type *>(rp(p));
}

//
//  Level 1
//
inline void axpy(int n, float a, const float *x, int ix, float *y, int iy)
{ cblas_saxpy(n, a, x, ix, y, iy); }
inline void axpy(int n, double a, const double *x, int ix, double *y, int iy)
{ cblas_daxpy(n, a, x, ix, y, iy); }
inline void axpy(int n, cfloat a, const cfloat *x, int ix, cfloat *y, int iy)
{ cblas_caxpy(n, rp(&a), rp(x), ix, rp(y), iy); }
inline void axpy(int n, cdouble a, const cdouble *x, int ix, cdouble *y, int iy)
{ cblas_zaxpy(n, rp(&a), rp(x), ix, rp(y), iy); }

inline void copy(int n, const float *x, int ix, float *y, int iy)
{ cblas_scopy(n, x, ix, y, iy); }
inline void copy(int n, const double *x, int ix, double *y, int iy)
{ cblas_dcopy(n, x, ix, y, iy); }
inline void copy(int n, const cfloat *x, int ix, cfloat *y, int iy)
{ cblas_ccopy(n, rp(x), ix, rp(y), iy); }
inline void copy(int n, const cdouble *x, int ix, cdouble *y, int iy)
{ cblas_zcopy(n, rp(x), ix, rp(y), iy); }

inline void swap(int n, float *x, int ix, float *y, int iy)
{ cblas_sswap(n, x, ix, y, iy); }
inline void swap(int n, double *x, int ix, double *y, int iy)
{ cblas_dswap(n, x, ix, y, iy); }
inline void swap(int n, cfloat *x, int ix, cfloat *y, int iy)
{ cblas_cswap(n, rp(x), ix, rp(y), iy); }
inline void swap(int n, cdouble *x, int ix, cdouble *y, int iy)
{ cblas_zswap(n, rp(x), ix, rp(y), iy); }

inline void scal(int n, float a, float *x, int ix)
{ cblas_sscal(n, a, x, ix); }
inline void scal(int n, double a, double *x, int ix)
{ cblas_dscal(n, a, x, ix); }
inline void scal(int n, cfloat a, cfloat *x, int ix)
{ cblas_cscal(n, rp(&a), rp(x), ix); }
inline void scal(int n, cdouble a, cdouble *x, int ix)
{ cblas_zscal(n, rp(&a), rp(x), ix); }

// csscal, zdscal: complex vector scaled by a real scalar
inline void rscal(int n, float a, cfloat *x, int ix)
{ cblas_csscal(n, a, rp(x), ix); }
inline void rscal(int n, double a, cdouble *x, int ix)
{ cblas_zdscal(n, a, rp(x), ix); }
inline void rscal(int n, float a, float *x, int ix)    { scal(n, a, x, ix); }
inline void rscal(int n, double a, double *x, int ix)  { scal(n, a, x, ix); }

// dotu: x^T y, dotc: x^H y (identical for real types)
inline float dotu(int n, const float *x, int ix, const float *y, int iy)
{ return cblas_sdot(n, x, ix, y, iy); }
inline double dotu(int n, const double *x, int ix, const double *y, int iy)
{ return cblas_ddot(n, x, ix, y, iy); }
inline cfloat dotu(int n, const cfloat *x, int ix, const cfloat *y, int iy)
{ cfloat r; cblas_cdotu_sub(n, rp(x), ix, rp(y), iy, rp(&r)); return r; }
inline cdouble dotu(int n, const cdouble *x, int ix, const cdouble *y, int iy)
{ cdouble r; cblas_zdotu_sub(n, rp(x), ix, rp(y), iy, rp(&r)); return r; }

inline float dotc(int n, const float *x, int ix, const float *y, int iy)
{ return cblas_sdot(n, x, ix, y, iy); }
inline double dotc(int n, const double *x, int ix, const double *y, int iy)
{ return cblas_ddot(n, x, ix, y, iy); }
inline cfloat dotc(int n, const cfloat *x, int ix, const cfloat *y, int iy)
{ cfloat r; cblas_cdotc_sub(n, rp(x), ix, rp(y), iy, rp(&r)); return r; }
inline cdouble dotc(int n, const cdouble *x, int ix, const cdouble *y, int iy)
{ cdouble r; cblas_zdotc_sub(n, rp(x), ix, rp(y), iy, rp(&r)); return r; }

inline float  nrm2(int n, const float *x, int ix)   { return cblas_snrm2(n, x, ix); }
inline double nrm2(int n, const double *x, int ix)  { return cblas_dnrm2(n, x, ix); }
inline float  nrm2(int n, const cfloat *x, int ix)  { return cblas_scnrm2(n, rp(x), ix); }
inline double nrm2(int n, const cdouble *x, int ix) { return cblas_dznrm2(n, rp(x), ix); }

inline float  asum(int n, const float *x, int ix)   { return cblas_sasum(n, x, ix); }
inline double asum(int n, const double *x, int ix)  { return cblas_dasum(n, x, ix); }
inline float  asum(int n, const cfloat *x, int ix)  { return cblas_scasum(n, rp(x), ix); }
inline double asum(int n, const cdouble *x, int ix) { return cblas_dzasum(n, rp(x), ix); }

inline int iamax(int n, const float *x, int ix)   { return cblas_isamax(n, x, ix); }
inline int iamax(int n, const double *x, int ix)  { return cblas_idamax(n, x, ix); }
inline int iamax(int n, const cfloat *x, int ix)  { return cblas_icamax(n, rp(x), ix); }
inline int iamax(int n, const cdouble *x, int ix) { return cblas_izamax(n, rp(x), ix); }

inline void rot(int n, float *x, int ix, float *y, int iy, float c, float s)
{ cblas_srot(n, x, ix, y, iy, c, s); }
inline void rot(int n, double *x, int ix, double *y, int iy, double c, double s)
{ cblas_drot(n, x, ix, y, iy, c, s); }

inline void rotg(float *a, float *b, float *c, float *s)     { cblas_srotg(a, b, c, s); }
inline void rotg(double *a, double *b, double *c, double *s) { cblas_drotg(a, b, c, s); }
inline void rotg(cfloat *a, cfloat *b, float *c, cfloat *s)
{ cblas_crotg(rp(a), rp(b), c, rp(s)); }
inline void rotg(cdouble *a, cdouble *b, double *c, cdouble *s)
{ cblas_zrotg(rp(a), rp(b), c, rp(s)); }

// csrot, zdrot: complex vectors, real c and s
inline void rot(int n, cfloat *x, int ix, cfloat *y, int iy, float c, float s)
{ cblas_csrot(n, rp(x), ix, rp(y), iy, c, s); }
inline void rot(int n, cdouble *x, int ix, cdouble *y, int iy, double c,
                double s)
{ cblas_zdrot(n, rp(x), ix, rp(y), iy, c, s); }

inline void rotm(int n, float *x, int ix, float *y, int iy, const float *p)
{ cblas_srotm(n, x, ix, y, iy, p); }
inline void rotm(int n, double *x, int ix, double *y, int iy, const double *p)
{ cblas_drotm(n, x, ix, y, iy, p); }

inline void rotmg(float *d1, float *d2, float *x1, float y1, float *p)
{ cblas_srotmg(d1, d2, x1, y1, p); }
inline void rotmg(double *d1, double *d2, double *x1, double y1, double *p)
{ cblas_drotmg(d1, d2, x1, y1, p); }

inline float  abs1(const cfloat &z)  { return cblas_scabs1(rp(&z)); }
inline double abs1(const cdouble &z) { return cblas_dcabs1(rp(&z)); }

//
//  Level 2
//
inline void gemv(CBLAS_ORDER o, CBLAS_TRANSPOSE t, int m, int n, float a,
                 const float *A, int ldA, const float *x, int ix, float b,
                 float *y, int iy)
{ cblas_sgemv(o, t, m, n, a, A, ldA, x, ix, b, y, iy); }
inline void gemv(CBLAS_ORDER o, CBLAS_TRANSPOSE t, int m, int n, double a,
                 const double *A, int ldA, const double *x, int ix, double b,
                 double *y, int iy)
{ cblas_dgemv(o, t, m, n, a, A, ldA, x, ix, b, y, iy); }
inline void gemv(CBLAS_ORDER o, CBLAS_TRANSPOSE t, int m, int n, cfloat a,
                 const cfloat *A, int ldA, const cfloat *x, int ix, cfloat b,
                 cfloat *y, int iy)
{ cblas_cgemv(o, t, m, n, rp(&a), rp(A), ldA, rp(x), ix, rp(&b), rp(y), iy); }
inline void gemv(CBLAS_ORDER o, CBLAS_TRANSPOSE t, int m, int n, cdouble a,
                 const cdouble *A, int ldA, const cdouble *x, int ix, cdouble b,
                 cdouble *y, int iy)
{ cblas_zgemv(o, t, m, n, rp(&a), rp(A), ldA, rp(x), ix, rp(&b), rp(y), iy); }

// geru: A += alpha x y^T, gerc: A += alpha x y^H (ger for real types)
inline void geru(CBLAS_ORDER o, int m, int n, float a, const float *x, int ix,
                 const float *y, int iy, float *A, int ldA)
{ cblas_sger(o, m, n, a, x, ix, y, iy, A, ldA); }
inline void geru(CBLAS_ORDER o, int m, int n, double a, const double *x, int ix,
                 const double *y, int iy, double *A, int ldA)
{ cblas_dger(o, m, n, a, x, ix, y, iy, A, ldA); }
inline void geru(CBLAS_ORDER o, int m, int n, cfloat a, const cfloat *x, int ix,
                 const cfloat *y, int iy, cfloat *A, int ldA)
{ cblas_cgeru(o, m, n, rp(&a), rp(x), ix, rp(y), iy, rp(A), ldA); }
inline void geru(CBLAS_ORDER o, int m, int n, cdouble a, const cdouble *x, int ix,
                 const cdouble *y, int iy, cdouble *A, int ldA)
{ cblas_zgeru(o, m, n, rp(&a), rp(x), ix, rp(y), iy, rp(A), ldA); }

inline void gerc(CBLAS_ORDER o, int m, int n, float a, const float *x, int ix,
                 const float *y, int iy, float *A, int ldA)
{ cblas_sger(o, m, n, a, x, ix, y, iy, A, ldA); }
inline void gerc(CBLAS_ORDER o, int m, int n, double a, const double *x, int ix,
                 const double *y, int iy, double *A, int ldA)
{ cblas_dger(o, m, n, a, x, ix, y, iy, A, ldA); }
inline void gerc(CBLAS_ORDER o, int m, int n, cfloat a, const cfloat *x, int ix,
                 const cfloat *y, int iy, cfloat *A, int ldA)
{ cblas_cgerc(o, m, n, rp(&a), rp(x), ix, rp(y), iy, rp(A), ldA); }
inline void gerc(CBLAS_ORDER o, int m, int n, cdouble a, const cdouble *x, int ix,
                 const cdouble *y, int iy, cdouble *A, int ldA)
{ cblas_zgerc(o, m, n, rp(&a), rp(x), ix, rp(y), iy, rp(A), ldA); }

// symv for real, hemv for complex types
inline void hemv(CBLAS_ORDER o, CBLAS_UPLO u, int n, float a, const float *A,
                 int ldA, const float *x, int ix, float b, float *y, int iy)
{ cblas_ssymv(o, u, n, a, A, ldA, x, ix, b, y, iy); }
inline void hemv(CBLAS_ORDER o, CBLAS_UPLO u, int n, double a, const double *A,
                 int ldA, const double *x, int ix, double b, double *y, int iy)
{ cblas_dsymv(o, u, n, a, A, ldA, x, ix, b, y, iy); }
inline void hemv(CBLAS_ORDER o, CBLAS_UPLO u, int n, cfloat a, const cfloat *A,
                 int ldA, const cfloat *x, int ix, cfloat b, cfloat *y, int iy)
{ cblas_chemv(o, u, n, rp(&a), rp(A), ldA, rp(x), ix, rp(&b), rp(y), iy); }
inline void hemv(CBLAS_ORDER o, CBLAS_UPLO u, int n, cdouble a,
                 const cdouble *A, int ldA, const cdouble *x, int ix, cdouble b,
                 cdouble *y, int iy)
{ cblas_zhemv(o, u, n, rp(&a), rp(A), ldA, rp(x), ix, rp(&b), rp(y), iy); }

// syr for real, her for complex types (alpha is real)
inline void her(CBLAS_ORDER o, CBLAS_UPLO u, int n, float a, const float *x,
                int ix, float *A, int ldA)
{ cblas_ssyr(o, u, n, a, x, ix, A, ldA); }
inline void her(CBLAS_ORDER o, CBLAS_UPLO u, int n, double a, const double *x,
                int ix, double *A, int ldA)
{ cblas_dsyr(o, u, n, a, x, ix, A, ldA); }
inline void her(CBLAS_ORDER o, CBLAS_UPLO u, int n, float a, const cfloat *x,
                int ix, cfloat *A, int ldA)
{ cblas_cher(o, u, n, a, rp(x), ix, rp(A), ldA); }
inline void her(CBLAS_ORDER o, CBLAS_UPLO u, int n, double a, const cdouble *x,
                int ix, cdouble *A, int ldA)
{ cblas_zher(o, u, n, a, rp(x), ix, rp(A), ldA); }

// syr2 for real, her2 for complex types
inline void her2(CBLAS_ORDER o, CBLAS_UPLO u, int n, float a, const float *x,
                 int ix, const float *y, int iy, float *A, int ldA)
{ cblas_ssyr2(o, u, n, a, x, ix, y, iy, A, ldA); }
inline void her2(CBLAS_ORDER o, CBLAS_UPLO u, int n, double a, const double *x,
                 int ix, const double *y, int iy, double *A, int ldA)
{ cblas_dsyr2(o, u, n, a, x, ix, y, iy, A, ldA); }
inline void her2(CBLAS_ORDER o, CBLAS_UPLO u, int n, cfloat a, const cfloat *x,
                 int ix, const cfloat *y, int iy, cfloat *A, int ldA)
{ cblas_cher2(o, u, n, rp(&a), rp(x), ix, rp(y), iy, rp(A), ldA); }
inline void her2(CBLAS_ORDER o, CBLAS_UPLO u, int n, cdouble a,
                 const cdouble *x, int ix, const cdouble *y, int iy,
                 cdouble *A, int ldA)
{ cblas_zher2(o, u, n, rp(&a), rp(x), ix, rp(y), iy, rp(A), ldA); }

// spr for real, hpr for complex types (packed storage, alpha is real)
inline void hpr(CBLAS_ORDER o, CBLAS_UPLO u, int n, float a, const float *x,
                int ix, float *Ap)
{ cblas_sspr(o, u, n, a, x, ix, Ap); }
inline void hpr(CBLAS_ORDER o, CBLAS_UPLO u, int n, double a, const double *x,
                int ix, double *Ap)
{ cblas_dspr(o, u, n, a, x, ix, Ap); }
inline void hpr(CBLAS_ORDER o, CBLAS_UPLO u, int n, float a, const cfloat *x,
                int ix, cfloat *Ap)
{ cblas_chpr(o, u, n, a, rp(x), ix, rp(Ap)); }
inline void hpr(CBLAS_ORDER o, CBLAS_UPLO u, int n, double a, const cdouble *x,
                int ix, cdouble *Ap)
{ cblas_zhpr(o, u, n, a, rp(x), ix, rp(Ap)); }

// spr2 for real, hpr2 for complex types (packed storage)
inline void hpr2(CBLAS_ORDER o, CBLAS_UPLO u, int n, float a, const float *x,
                 int ix, const float *y, int iy, float *Ap)
{ cblas_sspr2(o, u, n, a, x, ix, y, iy, Ap); }
inline void hpr2(CBLAS_ORDER o, CBLAS_UPLO u, int n, double a, const double *x,
                 int ix, const double *y, int iy, double *Ap)
{ cblas_dspr2(o, u, n, a, x, ix, y, iy, Ap); }
inline void hpr2(CBLAS_ORDER o, CBLAS_UPLO u, int n, cfloat a, const cfloat *x,
                 int ix, const cfloat *y, int iy, cfloat *Ap)
{ cblas_chpr2(o, u, n, rp(&a), rp(x), ix, rp(y), iy, rp(Ap)); }
inline void hpr2(CBLAS_ORDER o, CBLAS_UPLO u, int n, cdouble a,
                 const cdouble *x, int ix, const cdouble *y, int iy,
                 cdouble *Ap)
{ cblas_zhpr2(o, u, n, rp(&a), rp(x), ix, rp(y), iy, rp(Ap)); }

inline void trmv(CBLAS_ORDER o, CBLAS_UPLO u, CBLAS_TRANSPOSE t, CBLAS_DIAG d,
                 int n, const float *A, int ldA, float *x, int ix)
{ cblas_strmv(o, u, t, d, n, A, ldA, x, ix); }
inline void trmv(CBLAS_ORDER o, CBLAS_UPLO u, CBLAS_TRANSPOSE t, CBLAS_DIAG d,
                 int n, const double *A, int ldA, double *x, int ix)
{ cblas_dtrmv(o, u, t, d, n, A, ldA, x, ix); }
inline void trmv(CBLAS_ORDER o, CBLAS_UPLO u, CBLAS_TRANSPOSE t, CBLAS_DIAG d,
                 int n, const cfloat *A, int ldA, cfloat *x, int ix)
{ cblas_ctrmv(o, u, t, d, n, rp(A), ldA, rp(x), ix); }
inline void trmv(CBLAS_ORDER o, CBLAS_UPLO u, CBLAS_TRANSPOSE t, CBLAS_DIAG d,
                 int n, const cdouble *A, int ldA, cdouble *x, int ix)
{ cblas_ztrmv(o, u, t, d, n, rp(A), ldA, rp(x), ix); }

inline void trsv(CBLAS_ORDER o, CBLAS_UPLO u, CBLAS_TRANSPOSE t, CBLAS_DIAG d,
                 int n, const float *A, int ldA, float *x, int ix)
{ cblas_strsv(o, u, t, d, n, A, ldA, x, ix); }
inline void trsv(CBLAS_ORDER o, CBLAS_UPLO u, CBLAS_TRANSPOSE t, CBLAS_DIAG d,
                 int n, const double *A, int ldA, double *x, int ix)
{ cblas_dtrsv(o, u, t, d, n, A, ldA, x, ix); }
inline void trsv(CBLAS_ORDER o, CBLAS_UPLO u, CBLAS_TRANSPOSE t, CBLAS_DIAG d,
                 int n, const cfloat *A, int ldA, cfloat *x, int ix)
{ cblas_ctrsv(o, u, t, d, n, rp(A), ldA, rp(x), ix); }
inline void trsv(CBLAS_ORDER o, CBLAS_UPLO u, CBLAS_TRANSPOSE t, CBLAS_DIAG d,
                 int n, const cdouble *A, int ldA, cdouble *x, int ix)
{ cblas_ztrsv(o, u, t, d, n, rp(A), ldA, rp(x), ix); }

// gbmv: y = alpha*op(A)*x + beta*y, A general band matrix (kl, ku)
inline void gbmv(CBLAS_ORDER o, CBLAS_TRANSPOSE t, int m, int n, int kl, int ku,
                 float a, const float *A, int ldA, const float *x, int ix,
                 float b, float *y, int iy)
{ cblas_sgbmv(o, t, m, n, kl, ku, a, A, ldA, x, ix, b, y, iy); }
inline void gbmv(CBLAS_ORDER o, CBLAS_TRANSPOSE t, int m, int n, int kl, int ku,
                 double a, const double *A, int ldA, const double *x, int ix,
                 double b, double *y, int iy)
{ cblas_dgbmv(o, t, m, n, kl, ku, a, A, ldA, x, ix, b, y, iy); }
inline void gbmv(CBLAS_ORDER o, CBLAS_TRANSPOSE t, int m, int n, int kl, int ku,
                 cfloat a, const cfloat *A, int ldA, const cfloat *x, int ix,
                 cfloat b, cfloat *y, int iy)
{ cblas_cgbmv(o, t, m, n, kl, ku, rp(&a), rp(A), ldA, rp(x), ix, rp(&b),
              rp(y), iy); }
inline void gbmv(CBLAS_ORDER o, CBLAS_TRANSPOSE t, int m, int n, int kl, int ku,
                 cdouble a, const cdouble *A, int ldA, const cdouble *x, int ix,
                 cdouble b, cdouble *y, int iy)
{ cblas_zgbmv(o, t, m, n, kl, ku, rp(&a), rp(A), ldA, rp(x), ix, rp(&b),
              rp(y), iy); }

// sbmv for real, hbmv for complex types (band, k super/sub diagonals)
inline void hbmv(CBLAS_ORDER o, CBLAS_UPLO u, int n, int k, float a,
                 const float *A, int ldA, const float *x, int ix, float b,
                 float *y, int iy)
{ cblas_ssbmv(o, u, n, k, a, A, ldA, x, ix, b, y, iy); }
inline void hbmv(CBLAS_ORDER o, CBLAS_UPLO u, int n, int k, double a,
                 const double *A, int ldA, const double *x, int ix, double b,
                 double *y, int iy)
{ cblas_dsbmv(o, u, n, k, a, A, ldA, x, ix, b, y, iy); }
inline void hbmv(CBLAS_ORDER o, CBLAS_UPLO u, int n, int k, cfloat a,
                 const cfloat *A, int ldA, const cfloat *x, int ix, cfloat b,
                 cfloat *y, int iy)
{ cblas_chbmv(o, u, n, k, rp(&a), rp(A), ldA, rp(x), ix, rp(&b), rp(y), iy); }
inline void hbmv(CBLAS_ORDER o, CBLAS_UPLO u, int n, int k, cdouble a,
                 const cdouble *A, int ldA, const cdouble *x, int ix, cdouble b,
                 cdouble *y, int iy)
{ cblas_zhbmv(o, u, n, k, rp(&a), rp(A), ldA, rp(x), ix, rp(&b), rp(y), iy); }

// spmv for real, hpmv for complex types (packed storage)
inline void hpmv(CBLAS_ORDER o, CBLAS_UPLO u, int n, float a, const float *Ap,
                 const float *x, int ix, float b, float *y, int iy)
{ cblas_sspmv(o, u, n, a, Ap, x, ix, b, y, iy); }
inline void hpmv(CBLAS_ORDER o, CBLAS_UPLO u, int n, double a, const double *Ap,
                 const double *x, int ix, double b, double *y, int iy)
{ cblas_dspmv(o, u, n, a, Ap, x, ix, b, y, iy); }
inline void hpmv(CBLAS_ORDER o, CBLAS_UPLO u, int n, cfloat a, const cfloat *Ap,
                 const cfloat *x, int ix, cfloat b, cfloat *y, int iy)
{ cblas_chpmv(o, u, n, rp(&a), rp(Ap), rp(x), ix, rp(&b), rp(y), iy); }
inline void hpmv(CBLAS_ORDER o, CBLAS_UPLO u, int n, cdouble a,
                 const cdouble *Ap, const cdouble *x, int ix, cdouble b,
                 cdouble *y, int iy)
{ cblas_zhpmv(o, u, n, rp(&a), rp(Ap), rp(x), ix, rp(&b), rp(y), iy); }

// tbmv, tbsv: triangular band matrix;  tpmv, tpsv: packed triangular matrix
#define ULMBLAS_TEST_TB(NAME, S, D, C, Z)                                    \
inline void NAME(CBLAS_ORDER o, CBLAS_UPLO u, CBLAS_TRANSPOSE t,            \
                 CBLAS_DIAG d, int n, int k, const float *A, int ldA,       \
                 float *x, int ix)                                          \
{ S(o, u, t, d, n, k, A, ldA, x, ix); }                                     \
inline void NAME(CBLAS_ORDER o, CBLAS_UPLO u, CBLAS_TRANSPOSE t,            \
                 CBLAS_DIAG d, int n, int k, const double *A, int ldA,      \
                 double *x, int ix)                                         \
{ D(o, u, t, d, n, k, A, ldA, x, ix); }                                     \
inline void NAME(CBLAS_ORDER o, CBLAS_UPLO u, CBLAS_TRANSPOSE t,            \
                 CBLAS_DIAG d, int n, int k, const cfloat *A, int ldA,      \
                 cfloat *x, int ix)                                         \
{ C(o, u, t, d, n, k, rp(A), ldA, rp(x), ix); }                             \
inline void NAME(CBLAS_ORDER o, CBLAS_UPLO u, CBLAS_TRANSPOSE t,            \
                 CBLAS_DIAG d, int n, int k, const cdouble *A, int ldA,     \
                 cdouble *x, int ix)                                        \
{ Z(o, u, t, d, n, k, rp(A), ldA, rp(x), ix); }

ULMBLAS_TEST_TB(tbmv, cblas_stbmv, cblas_dtbmv, cblas_ctbmv, cblas_ztbmv)
ULMBLAS_TEST_TB(tbsv, cblas_stbsv, cblas_dtbsv, cblas_ctbsv, cblas_ztbsv)
#undef ULMBLAS_TEST_TB

#define ULMBLAS_TEST_TP(NAME, S, D, C, Z)                                    \
inline void NAME(CBLAS_ORDER o, CBLAS_UPLO u, CBLAS_TRANSPOSE t,            \
                 CBLAS_DIAG d, int n, const float *Ap, float *x, int ix)    \
{ S(o, u, t, d, n, Ap, x, ix); }                                            \
inline void NAME(CBLAS_ORDER o, CBLAS_UPLO u, CBLAS_TRANSPOSE t,            \
                 CBLAS_DIAG d, int n, const double *Ap, double *x, int ix)  \
{ D(o, u, t, d, n, Ap, x, ix); }                                            \
inline void NAME(CBLAS_ORDER o, CBLAS_UPLO u, CBLAS_TRANSPOSE t,            \
                 CBLAS_DIAG d, int n, const cfloat *Ap, cfloat *x, int ix)  \
{ C(o, u, t, d, n, rp(Ap), rp(x), ix); }                                    \
inline void NAME(CBLAS_ORDER o, CBLAS_UPLO u, CBLAS_TRANSPOSE t,            \
                 CBLAS_DIAG d, int n, const cdouble *Ap, cdouble *x, int ix)\
{ Z(o, u, t, d, n, rp(Ap), rp(x), ix); }

ULMBLAS_TEST_TP(tpmv, cblas_stpmv, cblas_dtpmv, cblas_ctpmv, cblas_ztpmv)
ULMBLAS_TEST_TP(tpsv, cblas_stpsv, cblas_dtpsv, cblas_ctpsv, cblas_ztpsv)
#undef ULMBLAS_TEST_TP

//
//  Level 3
//
inline void gemm(CBLAS_ORDER o, CBLAS_TRANSPOSE ta, CBLAS_TRANSPOSE tb,
                 int m, int n, int k, float a, const float *A, int ldA,
                 const float *B, int ldB, float b, float *C, int ldC)
{ cblas_sgemm(o, ta, tb, m, n, k, a, A, ldA, B, ldB, b, C, ldC); }
inline void gemm(CBLAS_ORDER o, CBLAS_TRANSPOSE ta, CBLAS_TRANSPOSE tb,
                 int m, int n, int k, double a, const double *A, int ldA,
                 const double *B, int ldB, double b, double *C, int ldC)
{ cblas_dgemm(o, ta, tb, m, n, k, a, A, ldA, B, ldB, b, C, ldC); }
inline void gemm(CBLAS_ORDER o, CBLAS_TRANSPOSE ta, CBLAS_TRANSPOSE tb,
                 int m, int n, int k, cfloat a, const cfloat *A, int ldA,
                 const cfloat *B, int ldB, cfloat b, cfloat *C, int ldC)
{ cblas_cgemm(o, ta, tb, m, n, k, rp(&a), rp(A), ldA, rp(B), ldB, rp(&b),
              rp(C), ldC); }
inline void gemm(CBLAS_ORDER o, CBLAS_TRANSPOSE ta, CBLAS_TRANSPOSE tb,
                 int m, int n, int k, cdouble a, const cdouble *A, int ldA,
                 const cdouble *B, int ldB, cdouble b, cdouble *C, int ldC)
{ cblas_zgemm(o, ta, tb, m, n, k, rp(&a), rp(A), ldA, rp(B), ldB, rp(&b),
              rp(C), ldC); }

inline void symm(CBLAS_ORDER o, CBLAS_SIDE s, CBLAS_UPLO u, int m, int n,
                 float a, const float *A, int ldA, const float *B, int ldB,
                 float b, float *C, int ldC)
{ cblas_ssymm(o, s, u, m, n, a, A, ldA, mp(B), ldB, b, C, ldC); }
inline void symm(CBLAS_ORDER o, CBLAS_SIDE s, CBLAS_UPLO u, int m, int n,
                 double a, const double *A, int ldA, const double *B, int ldB,
                 double b, double *C, int ldC)
{ cblas_dsymm(o, s, u, m, n, a, A, ldA, mp(B), ldB, b, C, ldC); }
inline void symm(CBLAS_ORDER o, CBLAS_SIDE s, CBLAS_UPLO u, int m, int n,
                 cfloat a, const cfloat *A, int ldA, const cfloat *B, int ldB,
                 cfloat b, cfloat *C, int ldC)
{ cblas_csymm(o, s, u, m, n, rp(&a), rp(A), ldA, mp(B), ldB, rp(&b), rp(C),
              ldC); }
inline void symm(CBLAS_ORDER o, CBLAS_SIDE s, CBLAS_UPLO u, int m, int n,
                 cdouble a, const cdouble *A, int ldA, const cdouble *B,
                 int ldB, cdouble b, cdouble *C, int ldC)
{ cblas_zsymm(o, s, u, m, n, rp(&a), rp(A), ldA, mp(B), ldB, rp(&b), rp(C),
              ldC); }

inline void hemm(CBLAS_ORDER o, CBLAS_SIDE s, CBLAS_UPLO u, int m, int n,
                 cfloat a, const cfloat *A, int ldA, const cfloat *B, int ldB,
                 cfloat b, cfloat *C, int ldC)
{ cblas_chemm(o, s, u, m, n, mp(&a), rp(A), ldA, mp(B), ldB, mp(&b), rp(C),
              ldC); }
inline void hemm(CBLAS_ORDER o, CBLAS_SIDE s, CBLAS_UPLO u, int m, int n,
                 cdouble a, const cdouble *A, int ldA, const cdouble *B,
                 int ldB, cdouble b, cdouble *C, int ldC)
{ cblas_zhemm(o, s, u, m, n, mp(&a), rp(A), ldA, mp(B), ldB, mp(&b), rp(C),
              ldC); }

inline void syrk(CBLAS_ORDER o, CBLAS_UPLO u, CBLAS_TRANSPOSE t, int n, int k,
                 float a, const float *A, int ldA, float b, float *C, int ldC)
{ cblas_ssyrk(o, u, t, n, k, a, A, ldA, b, C, ldC); }
inline void syrk(CBLAS_ORDER o, CBLAS_UPLO u, CBLAS_TRANSPOSE t, int n, int k,
                 double a, const double *A, int ldA, double b, double *C,
                 int ldC)
{ cblas_dsyrk(o, u, t, n, k, a, A, ldA, b, C, ldC); }
inline void syrk(CBLAS_ORDER o, CBLAS_UPLO u, CBLAS_TRANSPOSE t, int n, int k,
                 cfloat a, const cfloat *A, int ldA, cfloat b, cfloat *C,
                 int ldC)
{ cblas_csyrk(o, u, t, n, k, rp(&a), rp(A), ldA, rp(&b), rp(C), ldC); }
inline void syrk(CBLAS_ORDER o, CBLAS_UPLO u, CBLAS_TRANSPOSE t, int n, int k,
                 cdouble a, const cdouble *A, int ldA, cdouble b, cdouble *C,
                 int ldC)
{ cblas_zsyrk(o, u, t, n, k, rp(&a), rp(A), ldA, rp(&b), rp(C), ldC); }

inline void herk(CBLAS_ORDER o, CBLAS_UPLO u, CBLAS_TRANSPOSE t, int n, int k,
                 float a, const cfloat *A, int ldA, float b, cfloat *C, int ldC)
{ cblas_cherk(o, u, t, n, k, a, rp(A), ldA, b, rp(C), ldC); }
inline void herk(CBLAS_ORDER o, CBLAS_UPLO u, CBLAS_TRANSPOSE t, int n, int k,
                 double a, const cdouble *A, int ldA, double b, cdouble *C,
                 int ldC)
{ cblas_zherk(o, u, t, n, k, a, rp(A), ldA, b, rp(C), ldC); }

inline void syr2k(CBLAS_ORDER o, CBLAS_UPLO u, CBLAS_TRANSPOSE t, int n, int k,
                  float a, const float *A, int ldA, const float *B, int ldB,
                  float b, float *C, int ldC)
{ cblas_ssyr2k(o, u, t, n, k, a, A, ldA, B, ldB, b, C, ldC); }
inline void syr2k(CBLAS_ORDER o, CBLAS_UPLO u, CBLAS_TRANSPOSE t, int n, int k,
                  double a, const double *A, int ldA, const double *B, int ldB,
                  double b, double *C, int ldC)
{ cblas_dsyr2k(o, u, t, n, k, a, A, ldA, B, ldB, b, C, ldC); }
inline void syr2k(CBLAS_ORDER o, CBLAS_UPLO u, CBLAS_TRANSPOSE t, int n, int k,
                  cfloat a, const cfloat *A, int ldA, const cfloat *B, int ldB,
                  cfloat b, cfloat *C, int ldC)
{ cblas_csyr2k(o, u, t, n, k, rp(&a), rp(A), ldA, rp(B), ldB, rp(&b), rp(C),
               ldC); }
inline void syr2k(CBLAS_ORDER o, CBLAS_UPLO u, CBLAS_TRANSPOSE t, int n, int k,
                  cdouble a, const cdouble *A, int ldA, const cdouble *B,
                  int ldB, cdouble b, cdouble *C, int ldC)
{ cblas_zsyr2k(o, u, t, n, k, rp(&a), rp(A), ldA, rp(B), ldB, rp(&b), rp(C),
               ldC); }

inline void her2k(CBLAS_ORDER o, CBLAS_UPLO u, CBLAS_TRANSPOSE t, int n, int k,
                  cfloat a, const cfloat *A, int ldA, const cfloat *B, int ldB,
                  float b, cfloat *C, int ldC)
{ cblas_cher2k(o, u, t, n, k, mp(&a), rp(A), ldA, rp(B), ldB, b, rp(C), ldC); }
inline void her2k(CBLAS_ORDER o, CBLAS_UPLO u, CBLAS_TRANSPOSE t, int n, int k,
                  cdouble a, const cdouble *A, int ldA, const cdouble *B,
                  int ldB, double b, cdouble *C, int ldC)
{ cblas_zher2k(o, u, t, n, k, mp(&a), rp(A), ldA, rp(B), ldB, b, rp(C), ldC); }

// For real types hermitian is symmetric: hemm, herk and her2k forward to
// symm, syrk and syr2k (only used to keep the tests generic).
template <typename T>
inline void hemm(CBLAS_ORDER o, CBLAS_SIDE s, CBLAS_UPLO u, int m, int n,
                 T a, const T *A, int ldA, const T *B, int ldB, T b, T *C,
                 int ldC)
{ symm(o, s, u, m, n, a, A, ldA, B, ldB, b, C, ldC); }
template <typename T>
inline void herk(CBLAS_ORDER o, CBLAS_UPLO u, CBLAS_TRANSPOSE t, int n, int k,
                 T a, const T *A, int ldA, T b, T *C, int ldC)
{ syrk(o, u, t, n, k, a, A, ldA, b, C, ldC); }
template <typename T>
inline void her2k(CBLAS_ORDER o, CBLAS_UPLO u, CBLAS_TRANSPOSE t, int n, int k,
                  T a, const T *A, int ldA, const T *B, int ldB, T b, T *C,
                  int ldC)
{ syr2k(o, u, t, n, k, a, A, ldA, B, ldB, b, C, ldC); }

inline void trmm(CBLAS_ORDER o, CBLAS_SIDE s, CBLAS_UPLO u, CBLAS_TRANSPOSE t,
                 CBLAS_DIAG d, int m, int n, float a, const float *A, int ldA,
                 float *B, int ldB)
{ cblas_strmm(o, s, u, t, d, m, n, a, A, ldA, B, ldB); }
inline void trmm(CBLAS_ORDER o, CBLAS_SIDE s, CBLAS_UPLO u, CBLAS_TRANSPOSE t,
                 CBLAS_DIAG d, int m, int n, double a, const double *A,
                 int ldA, double *B, int ldB)
{ cblas_dtrmm(o, s, u, t, d, m, n, a, A, ldA, B, ldB); }
inline void trmm(CBLAS_ORDER o, CBLAS_SIDE s, CBLAS_UPLO u, CBLAS_TRANSPOSE t,
                 CBLAS_DIAG d, int m, int n, cfloat a, const cfloat *A,
                 int ldA, cfloat *B, int ldB)
{ cblas_ctrmm(o, s, u, t, d, m, n, rp(&a), rp(A), ldA, rp(B), ldB); }
inline void trmm(CBLAS_ORDER o, CBLAS_SIDE s, CBLAS_UPLO u, CBLAS_TRANSPOSE t,
                 CBLAS_DIAG d, int m, int n, cdouble a, const cdouble *A,
                 int ldA, cdouble *B, int ldB)
{ cblas_ztrmm(o, s, u, t, d, m, n, rp(&a), rp(A), ldA, rp(B), ldB); }

inline void trsm(CBLAS_ORDER o, CBLAS_SIDE s, CBLAS_UPLO u, CBLAS_TRANSPOSE t,
                 CBLAS_DIAG d, int m, int n, float a, const float *A, int ldA,
                 float *B, int ldB)
{ cblas_strsm(o, s, u, t, d, m, n, a, A, ldA, B, ldB); }
inline void trsm(CBLAS_ORDER o, CBLAS_SIDE s, CBLAS_UPLO u, CBLAS_TRANSPOSE t,
                 CBLAS_DIAG d, int m, int n, double a, const double *A,
                 int ldA, double *B, int ldB)
{ cblas_dtrsm(o, s, u, t, d, m, n, a, A, ldA, B, ldB); }
inline void trsm(CBLAS_ORDER o, CBLAS_SIDE s, CBLAS_UPLO u, CBLAS_TRANSPOSE t,
                 CBLAS_DIAG d, int m, int n, cfloat a, const cfloat *A,
                 int ldA, cfloat *B, int ldB)
{ cblas_ctrsm(o, s, u, t, d, m, n, rp(&a), rp(A), ldA, rp(B), ldB); }
inline void trsm(CBLAS_ORDER o, CBLAS_SIDE s, CBLAS_UPLO u, CBLAS_TRANSPOSE t,
                 CBLAS_DIAG d, int m, int n, cdouble a, const cdouble *A,
                 int ldA, cdouble *B, int ldB)
{ cblas_ztrsm(o, s, u, t, d, m, n, rp(&a), rp(A), ldA, rp(B), ldB); }

} // namespace blas

#endif // ULMBLAS_TEST_NUMERIC_BLAS_H
