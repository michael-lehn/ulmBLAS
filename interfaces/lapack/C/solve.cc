#include <complex>
#include <cstddef>
#include LAPACK_HEADER
#include <ulmblas/lapack/gesv.h>
#include <ulmblas/lapack/getri.h>
#include <ulmblas/lapack/getrs.h>
#include <ulmblas/lapack/laswp.h>
#include <ulmblas/lapack/lauum.h>
#include <ulmblas/lapack/posv.h>
#include <ulmblas/lapack/potri.h>
#include <ulmblas/lapack/potrs.h>
#include <ulmblas/lapack/trtri.h>
#include <ulmblas/lapack/trtrs.h>

//
//  Solves and inverses (ATLAS' clapack_* where ATLAS has them; ?trtrs and
//  ?laswp in the same style).  Pivots are 0-based, as returned by ?getrf.
//  Row major: ?getrs, ?gesv and ?getri use the factorization of ?getrf for
//  row major storage (A^T = P L U in column major, as ATLAS), all other
//  functions work on the matrix as stored.  The return value is LAPACK's
//  INFO; invalid arguments: -i for argument i.
//

namespace {

typedef std::ptrdiff_t Idx;

bool
okOrder(CBLAS_ORDER o)
{
    return o == CblasColMajor || o == CblasRowMajor;
}
bool
okUpLo(CBLAS_UPLO u)
{
    return u == CblasLower || u == CblasUpper;
}
bool
okTrans(CBLAS_TRANSPOSE t)
{
    return t == CblasNoTrans || t == CblasTrans || t == CblasConjTrans;
}
bool
okDiag(CBLAS_DIAG d)
{
    return d == CblasUnit || d == CblasNonUnit;
}
ULMBLAS_INDEX_TYPE
max1(ULMBLAS_INDEX_TYPE n)
{
    return n > 1 ? n : 1;
}

//  strides of a matrix as stored
Idx
rowStride(CBLAS_ORDER o, ULMBLAS_INDEX_TYPE ld)
{
    return o == CblasColMajor ? 1 : ld;
}
Idx
colStride(CBLAS_ORDER o, ULMBLAS_INDEX_TYPE ld)
{
    return o == CblasColMajor ? ld : 1;
}

template <typename T>
ULMBLAS_INDEX_TYPE
getrs(CBLAS_ORDER o, CBLAS_TRANSPOSE t, ULMBLAS_INDEX_TYPE n,
      ULMBLAS_INDEX_TYPE nrhs, const T *A, ULMBLAS_INDEX_TYPE ldA,
      const ULMBLAS_INDEX_TYPE *piv, T *B, ULMBLAS_INDEX_TYPE ldB)
{
    if (!okOrder(o))
	return -1;
    if (!okTrans(t))
	return -2;
    if (n < 0)
	return -3;
    if (nrhs < 0)
	return -4;
    if (ldA < max1(n))
	return -6;
    if (ldB < max1(o == CblasColMajor ? n : nrhs))
	return -9;
    //  row major: the factorization is that of A^T in column major
    bool trans = t != CblasNoTrans;
    const bool conj = t == CblasConjTrans;
    if (o == CblasRowMajor) {
	trans = !trans;
    }
    ulmBLAS::getrs<Idx>(trans, conj, n, nrhs, A, 1, ldA, piv, 1, B,
                        rowStride(o, ldB), colStride(o, ldB));
    return 0;
}

template <typename T>
ULMBLAS_INDEX_TYPE
gesv(CBLAS_ORDER o, ULMBLAS_INDEX_TYPE n, ULMBLAS_INDEX_TYPE nrhs, T *A,
     ULMBLAS_INDEX_TYPE ldA, ULMBLAS_INDEX_TYPE *piv, T *B,
     ULMBLAS_INDEX_TYPE ldB)
{
    if (!okOrder(o))
	return -1;
    if (n < 0)
	return -2;
    if (nrhs < 0)
	return -3;
    if (ldA < max1(n))
	return -5;
    if (ldB < max1(o == CblasColMajor ? n : nrhs))
	return -8;
    const ULMBLAS_INDEX_TYPE info =
        ulmBLAS::getrf<Idx>(n, n, A, 1, ldA, piv, 1);
    if (info == 0) {
	ulmBLAS::getrs<Idx>(o == CblasRowMajor, false, n, nrhs, A, 1, ldA, piv,
	                    1, B, rowStride(o, ldB), colStride(o, ldB));
    }
    return info;
}

template <typename T>
ULMBLAS_INDEX_TYPE
getri(CBLAS_ORDER o, ULMBLAS_INDEX_TYPE n, T *A, ULMBLAS_INDEX_TYPE ldA,
      const ULMBLAS_INDEX_TYPE *piv)
{
    if (!okOrder(o))
	return -1;
    if (n < 0)
	return -2;
    if (ldA < max1(n))
	return -4;
    //  row major: inv(A^T) in column major is inv(A) in row major
    return ulmBLAS::getri<Idx>(n, A, 1, ldA, piv, 1);
}

template <typename T>
ULMBLAS_INDEX_TYPE
potrs(CBLAS_ORDER o, CBLAS_UPLO u, ULMBLAS_INDEX_TYPE n,
      ULMBLAS_INDEX_TYPE nrhs, const T *A, ULMBLAS_INDEX_TYPE ldA, T *B,
      ULMBLAS_INDEX_TYPE ldB)
{
    if (!okOrder(o))
	return -1;
    if (!okUpLo(u))
	return -2;
    if (n < 0)
	return -3;
    if (nrhs < 0)
	return -4;
    if (ldA < max1(n))
	return -6;
    if (ldB < max1(o == CblasColMajor ? n : nrhs))
	return -8;
    ulmBLAS::potrs<Idx>(u == CblasLower, n, nrhs, A, rowStride(o, ldA),
                        colStride(o, ldA), B, rowStride(o, ldB),
                        colStride(o, ldB));
    return 0;
}

template <typename T>
ULMBLAS_INDEX_TYPE
posv(CBLAS_ORDER o, CBLAS_UPLO u, ULMBLAS_INDEX_TYPE n, ULMBLAS_INDEX_TYPE nrhs,
     T *A, ULMBLAS_INDEX_TYPE ldA, T *B, ULMBLAS_INDEX_TYPE ldB)
{
    if (!okOrder(o))
	return -1;
    if (!okUpLo(u))
	return -2;
    if (n < 0)
	return -3;
    if (nrhs < 0)
	return -4;
    if (ldA < max1(n))
	return -6;
    if (ldB < max1(o == CblasColMajor ? n : nrhs))
	return -8;
    return ulmBLAS::posv<Idx>(u == CblasLower, n, nrhs, A, rowStride(o, ldA),
                              colStride(o, ldA), B, rowStride(o, ldB),
                              colStride(o, ldB));
}

template <typename T>
ULMBLAS_INDEX_TYPE
trtrs(CBLAS_ORDER o, CBLAS_UPLO u, CBLAS_TRANSPOSE t, CBLAS_DIAG d,
      ULMBLAS_INDEX_TYPE n, ULMBLAS_INDEX_TYPE nrhs, const T *A,
      ULMBLAS_INDEX_TYPE ldA, T *B, ULMBLAS_INDEX_TYPE ldB)
{
    if (!okOrder(o))
	return -1;
    if (!okUpLo(u))
	return -2;
    if (!okTrans(t))
	return -3;
    if (!okDiag(d))
	return -4;
    if (n < 0)
	return -5;
    if (nrhs < 0)
	return -6;
    if (ldA < max1(n))
	return -8;
    if (ldB < max1(o == CblasColMajor ? n : nrhs))
	return -10;
    return ulmBLAS::trtrs<Idx>(u == CblasLower, t != CblasNoTrans,
                               t == CblasConjTrans, d == CblasUnit, n, nrhs, A,
                               rowStride(o, ldA), colStride(o, ldA), B,
                               rowStride(o, ldB), colStride(o, ldB));
}

template <typename T>
ULMBLAS_INDEX_TYPE
trtri(CBLAS_ORDER o, CBLAS_UPLO u, CBLAS_DIAG d, ULMBLAS_INDEX_TYPE n, T *A,
      ULMBLAS_INDEX_TYPE ldA)
{
    if (!okOrder(o))
	return -1;
    if (!okUpLo(u))
	return -2;
    if (!okDiag(d))
	return -3;
    if (n < 0)
	return -4;
    if (ldA < max1(n))
	return -6;
    return ulmBLAS::trtri<Idx>(u == CblasLower, d == CblasUnit, n, A,
                               rowStride(o, ldA), colStride(o, ldA));
}

template <typename T>
ULMBLAS_INDEX_TYPE
lauum(CBLAS_ORDER o, CBLAS_UPLO u, ULMBLAS_INDEX_TYPE n, T *A,
      ULMBLAS_INDEX_TYPE ldA)
{
    if (!okOrder(o))
	return -1;
    if (!okUpLo(u))
	return -2;
    if (n < 0)
	return -3;
    if (ldA < max1(n))
	return -5;
    ulmBLAS::lauum<Idx>(u == CblasLower, n, A, rowStride(o, ldA),
                        colStride(o, ldA));
    return 0;
}

template <typename T>
ULMBLAS_INDEX_TYPE
potri(CBLAS_ORDER o, CBLAS_UPLO u, ULMBLAS_INDEX_TYPE n, T *A,
      ULMBLAS_INDEX_TYPE ldA)
{
    if (!okOrder(o))
	return -1;
    if (!okUpLo(u))
	return -2;
    if (n < 0)
	return -3;
    if (ldA < max1(n))
	return -5;
    return ulmBLAS::potri<Idx>(u == CblasLower, n, A, rowStride(o, ldA),
                               colStride(o, ldA));
}

} // namespace

extern "C" {

ULMBLAS_INDEX_TYPE
ULMLAPACK(sgetrs)(CBLAS_ORDER order, CBLAS_TRANSPOSE trans,
                  ULMBLAS_INDEX_TYPE n, ULMBLAS_INDEX_TYPE nrhs, const float *A,
                  ULMBLAS_INDEX_TYPE ldA, const ULMBLAS_INDEX_TYPE *piv,
                  float *B, ULMBLAS_INDEX_TYPE ldB)
{
    return getrs(order, trans, n, nrhs, A, ldA, piv, B, ldB);
}

ULMBLAS_INDEX_TYPE
ULMLAPACK(sgesv)(CBLAS_ORDER order, ULMBLAS_INDEX_TYPE n,
                 ULMBLAS_INDEX_TYPE nrhs, float *A, ULMBLAS_INDEX_TYPE ldA,
                 ULMBLAS_INDEX_TYPE *piv, float *B, ULMBLAS_INDEX_TYPE ldB)
{
    return gesv(order, n, nrhs, A, ldA, piv, B, ldB);
}

ULMBLAS_INDEX_TYPE
ULMLAPACK(sgetri)(CBLAS_ORDER order, ULMBLAS_INDEX_TYPE n, float *A,
                  ULMBLAS_INDEX_TYPE ldA, const ULMBLAS_INDEX_TYPE *piv)
{
    return getri(order, n, A, ldA, piv);
}

ULMBLAS_INDEX_TYPE
ULMLAPACK(spotrs)(CBLAS_ORDER order, CBLAS_UPLO upLo, ULMBLAS_INDEX_TYPE n,
                  ULMBLAS_INDEX_TYPE nrhs, const float *A,
                  ULMBLAS_INDEX_TYPE ldA, float *B, ULMBLAS_INDEX_TYPE ldB)
{
    return potrs(order, upLo, n, nrhs, A, ldA, B, ldB);
}

ULMBLAS_INDEX_TYPE
ULMLAPACK(sposv)(CBLAS_ORDER order, CBLAS_UPLO upLo, ULMBLAS_INDEX_TYPE n,
                 ULMBLAS_INDEX_TYPE nrhs, float *A, ULMBLAS_INDEX_TYPE ldA,
                 float *B, ULMBLAS_INDEX_TYPE ldB)
{
    return posv(order, upLo, n, nrhs, A, ldA, B, ldB);
}

ULMBLAS_INDEX_TYPE
ULMLAPACK(strtrs)(CBLAS_ORDER order, CBLAS_UPLO upLo, CBLAS_TRANSPOSE trans,
                  CBLAS_DIAG diag, ULMBLAS_INDEX_TYPE n,
                  ULMBLAS_INDEX_TYPE nrhs, const float *A,
                  ULMBLAS_INDEX_TYPE ldA, float *B, ULMBLAS_INDEX_TYPE ldB)
{
    return trtrs(order, upLo, trans, diag, n, nrhs, A, ldA, B, ldB);
}

ULMBLAS_INDEX_TYPE
ULMLAPACK(strtri)(CBLAS_ORDER order, CBLAS_UPLO upLo, CBLAS_DIAG diag,
                  ULMBLAS_INDEX_TYPE n, float *A, ULMBLAS_INDEX_TYPE ldA)
{
    return trtri(order, upLo, diag, n, A, ldA);
}

ULMBLAS_INDEX_TYPE
ULMLAPACK(slauum)(CBLAS_ORDER order, CBLAS_UPLO upLo, ULMBLAS_INDEX_TYPE n,
                  float *A, ULMBLAS_INDEX_TYPE ldA)
{
    return lauum(order, upLo, n, A, ldA);
}

ULMBLAS_INDEX_TYPE
ULMLAPACK(spotri)(CBLAS_ORDER order, CBLAS_UPLO upLo, ULMBLAS_INDEX_TYPE n,
                  float *A, ULMBLAS_INDEX_TYPE ldA)
{
    return potri(order, upLo, n, A, ldA);
}

void
ULMLAPACK(slaswp)(ULMBLAS_INDEX_TYPE n, float *A, ULMBLAS_INDEX_TYPE ldA,
                  ULMBLAS_INDEX_TYPE k1, ULMBLAS_INDEX_TYPE k2,
                  ULMBLAS_INDEX_TYPE *piv, ULMBLAS_INDEX_TYPE incPiv)
{
    ulmBLAS::laswp<Idx>(n, A, 1, ldA, k1, k2, piv, incPiv);
}

ULMBLAS_INDEX_TYPE
ULMLAPACK(dgetrs)(CBLAS_ORDER order, CBLAS_TRANSPOSE trans,
                  ULMBLAS_INDEX_TYPE n, ULMBLAS_INDEX_TYPE nrhs,
                  const double *A, ULMBLAS_INDEX_TYPE ldA,
                  const ULMBLAS_INDEX_TYPE *piv, double *B,
                  ULMBLAS_INDEX_TYPE ldB)
{
    return getrs(order, trans, n, nrhs, A, ldA, piv, B, ldB);
}

ULMBLAS_INDEX_TYPE
ULMLAPACK(dgesv)(CBLAS_ORDER order, ULMBLAS_INDEX_TYPE n,
                 ULMBLAS_INDEX_TYPE nrhs, double *A, ULMBLAS_INDEX_TYPE ldA,
                 ULMBLAS_INDEX_TYPE *piv, double *B, ULMBLAS_INDEX_TYPE ldB)
{
    return gesv(order, n, nrhs, A, ldA, piv, B, ldB);
}

ULMBLAS_INDEX_TYPE
ULMLAPACK(dgetri)(CBLAS_ORDER order, ULMBLAS_INDEX_TYPE n, double *A,
                  ULMBLAS_INDEX_TYPE ldA, const ULMBLAS_INDEX_TYPE *piv)
{
    return getri(order, n, A, ldA, piv);
}

ULMBLAS_INDEX_TYPE
ULMLAPACK(dpotrs)(CBLAS_ORDER order, CBLAS_UPLO upLo, ULMBLAS_INDEX_TYPE n,
                  ULMBLAS_INDEX_TYPE nrhs, const double *A,
                  ULMBLAS_INDEX_TYPE ldA, double *B, ULMBLAS_INDEX_TYPE ldB)
{
    return potrs(order, upLo, n, nrhs, A, ldA, B, ldB);
}

ULMBLAS_INDEX_TYPE
ULMLAPACK(dposv)(CBLAS_ORDER order, CBLAS_UPLO upLo, ULMBLAS_INDEX_TYPE n,
                 ULMBLAS_INDEX_TYPE nrhs, double *A, ULMBLAS_INDEX_TYPE ldA,
                 double *B, ULMBLAS_INDEX_TYPE ldB)
{
    return posv(order, upLo, n, nrhs, A, ldA, B, ldB);
}

ULMBLAS_INDEX_TYPE
ULMLAPACK(dtrtrs)(CBLAS_ORDER order, CBLAS_UPLO upLo, CBLAS_TRANSPOSE trans,
                  CBLAS_DIAG diag, ULMBLAS_INDEX_TYPE n,
                  ULMBLAS_INDEX_TYPE nrhs, const double *A,
                  ULMBLAS_INDEX_TYPE ldA, double *B, ULMBLAS_INDEX_TYPE ldB)
{
    return trtrs(order, upLo, trans, diag, n, nrhs, A, ldA, B, ldB);
}

ULMBLAS_INDEX_TYPE
ULMLAPACK(dtrtri)(CBLAS_ORDER order, CBLAS_UPLO upLo, CBLAS_DIAG diag,
                  ULMBLAS_INDEX_TYPE n, double *A, ULMBLAS_INDEX_TYPE ldA)
{
    return trtri(order, upLo, diag, n, A, ldA);
}

ULMBLAS_INDEX_TYPE
ULMLAPACK(dlauum)(CBLAS_ORDER order, CBLAS_UPLO upLo, ULMBLAS_INDEX_TYPE n,
                  double *A, ULMBLAS_INDEX_TYPE ldA)
{
    return lauum(order, upLo, n, A, ldA);
}

ULMBLAS_INDEX_TYPE
ULMLAPACK(dpotri)(CBLAS_ORDER order, CBLAS_UPLO upLo, ULMBLAS_INDEX_TYPE n,
                  double *A, ULMBLAS_INDEX_TYPE ldA)
{
    return potri(order, upLo, n, A, ldA);
}

ULMBLAS_INDEX_TYPE
ULMLAPACK(cgetrs)(CBLAS_ORDER order, CBLAS_TRANSPOSE trans,
                  ULMBLAS_INDEX_TYPE n, ULMBLAS_INDEX_TYPE nrhs, const void *A,
                  ULMBLAS_INDEX_TYPE ldA, const ULMBLAS_INDEX_TYPE *piv,
                  void *B, ULMBLAS_INDEX_TYPE ldB)
{
    return getrs(order, trans, n, nrhs,
                 static_cast<const std::complex<float> *>(A), ldA, piv,
                 static_cast<std::complex<float> *>(B), ldB);
}

ULMBLAS_INDEX_TYPE
ULMLAPACK(cgesv)(CBLAS_ORDER order, ULMBLAS_INDEX_TYPE n,
                 ULMBLAS_INDEX_TYPE nrhs, void *A, ULMBLAS_INDEX_TYPE ldA,
                 ULMBLAS_INDEX_TYPE *piv, void *B, ULMBLAS_INDEX_TYPE ldB)
{
    return gesv(order, n, nrhs, static_cast<std::complex<float> *>(A), ldA, piv,
                static_cast<std::complex<float> *>(B), ldB);
}

ULMBLAS_INDEX_TYPE
ULMLAPACK(cgetri)(CBLAS_ORDER order, ULMBLAS_INDEX_TYPE n, void *A,
                  ULMBLAS_INDEX_TYPE ldA, const ULMBLAS_INDEX_TYPE *piv)
{
    return getri(order, n, static_cast<std::complex<float> *>(A), ldA, piv);
}

ULMBLAS_INDEX_TYPE
ULMLAPACK(cpotrs)(CBLAS_ORDER order, CBLAS_UPLO upLo, ULMBLAS_INDEX_TYPE n,
                  ULMBLAS_INDEX_TYPE nrhs, const void *A,
                  ULMBLAS_INDEX_TYPE ldA, void *B, ULMBLAS_INDEX_TYPE ldB)
{
    return potrs(order, upLo, n, nrhs,
                 static_cast<const std::complex<float> *>(A), ldA,
                 static_cast<std::complex<float> *>(B), ldB);
}

ULMBLAS_INDEX_TYPE
ULMLAPACK(cposv)(CBLAS_ORDER order, CBLAS_UPLO upLo, ULMBLAS_INDEX_TYPE n,
                 ULMBLAS_INDEX_TYPE nrhs, void *A, ULMBLAS_INDEX_TYPE ldA,
                 void *B, ULMBLAS_INDEX_TYPE ldB)
{
    return posv(order, upLo, n, nrhs, static_cast<std::complex<float> *>(A),
                ldA, static_cast<std::complex<float> *>(B), ldB);
}

ULMBLAS_INDEX_TYPE
ULMLAPACK(ctrtrs)(CBLAS_ORDER order, CBLAS_UPLO upLo, CBLAS_TRANSPOSE trans,
                  CBLAS_DIAG diag, ULMBLAS_INDEX_TYPE n,
                  ULMBLAS_INDEX_TYPE nrhs, const void *A,
                  ULMBLAS_INDEX_TYPE ldA, void *B, ULMBLAS_INDEX_TYPE ldB)
{
    return trtrs(order, upLo, trans, diag, n, nrhs,
                 static_cast<const std::complex<float> *>(A), ldA,
                 static_cast<std::complex<float> *>(B), ldB);
}

ULMBLAS_INDEX_TYPE
ULMLAPACK(ctrtri)(CBLAS_ORDER order, CBLAS_UPLO upLo, CBLAS_DIAG diag,
                  ULMBLAS_INDEX_TYPE n, void *A, ULMBLAS_INDEX_TYPE ldA)
{
    return trtri(order, upLo, diag, n, static_cast<std::complex<float> *>(A),
                 ldA);
}

ULMBLAS_INDEX_TYPE
ULMLAPACK(clauum)(CBLAS_ORDER order, CBLAS_UPLO upLo, ULMBLAS_INDEX_TYPE n,
                  void *A, ULMBLAS_INDEX_TYPE ldA)
{
    return lauum(order, upLo, n, static_cast<std::complex<float> *>(A), ldA);
}

ULMBLAS_INDEX_TYPE
ULMLAPACK(cpotri)(CBLAS_ORDER order, CBLAS_UPLO upLo, ULMBLAS_INDEX_TYPE n,
                  void *A, ULMBLAS_INDEX_TYPE ldA)
{
    return potri(order, upLo, n, static_cast<std::complex<float> *>(A), ldA);
}

void
ULMLAPACK(claswp)(ULMBLAS_INDEX_TYPE n, void *A, ULMBLAS_INDEX_TYPE ldA,
                  ULMBLAS_INDEX_TYPE k1, ULMBLAS_INDEX_TYPE k2,
                  ULMBLAS_INDEX_TYPE *piv, ULMBLAS_INDEX_TYPE incPiv)
{
    ulmBLAS::laswp<Idx>(n, static_cast<std::complex<float> *>(A), 1, ldA, k1,
                        k2, piv, incPiv);
}

ULMBLAS_INDEX_TYPE
ULMLAPACK(zgetrs)(CBLAS_ORDER order, CBLAS_TRANSPOSE trans,
                  ULMBLAS_INDEX_TYPE n, ULMBLAS_INDEX_TYPE nrhs, const void *A,
                  ULMBLAS_INDEX_TYPE ldA, const ULMBLAS_INDEX_TYPE *piv,
                  void *B, ULMBLAS_INDEX_TYPE ldB)
{
    return getrs(order, trans, n, nrhs,
                 static_cast<const std::complex<double> *>(A), ldA, piv,
                 static_cast<std::complex<double> *>(B), ldB);
}

ULMBLAS_INDEX_TYPE
ULMLAPACK(zgesv)(CBLAS_ORDER order, ULMBLAS_INDEX_TYPE n,
                 ULMBLAS_INDEX_TYPE nrhs, void *A, ULMBLAS_INDEX_TYPE ldA,
                 ULMBLAS_INDEX_TYPE *piv, void *B, ULMBLAS_INDEX_TYPE ldB)
{
    return gesv(order, n, nrhs, static_cast<std::complex<double> *>(A), ldA,
                piv, static_cast<std::complex<double> *>(B), ldB);
}

ULMBLAS_INDEX_TYPE
ULMLAPACK(zgetri)(CBLAS_ORDER order, ULMBLAS_INDEX_TYPE n, void *A,
                  ULMBLAS_INDEX_TYPE ldA, const ULMBLAS_INDEX_TYPE *piv)
{
    return getri(order, n, static_cast<std::complex<double> *>(A), ldA, piv);
}

ULMBLAS_INDEX_TYPE
ULMLAPACK(zpotrs)(CBLAS_ORDER order, CBLAS_UPLO upLo, ULMBLAS_INDEX_TYPE n,
                  ULMBLAS_INDEX_TYPE nrhs, const void *A,
                  ULMBLAS_INDEX_TYPE ldA, void *B, ULMBLAS_INDEX_TYPE ldB)
{
    return potrs(order, upLo, n, nrhs,
                 static_cast<const std::complex<double> *>(A), ldA,
                 static_cast<std::complex<double> *>(B), ldB);
}

ULMBLAS_INDEX_TYPE
ULMLAPACK(zposv)(CBLAS_ORDER order, CBLAS_UPLO upLo, ULMBLAS_INDEX_TYPE n,
                 ULMBLAS_INDEX_TYPE nrhs, void *A, ULMBLAS_INDEX_TYPE ldA,
                 void *B, ULMBLAS_INDEX_TYPE ldB)
{
    return posv(order, upLo, n, nrhs, static_cast<std::complex<double> *>(A),
                ldA, static_cast<std::complex<double> *>(B), ldB);
}

ULMBLAS_INDEX_TYPE
ULMLAPACK(ztrtrs)(CBLAS_ORDER order, CBLAS_UPLO upLo, CBLAS_TRANSPOSE trans,
                  CBLAS_DIAG diag, ULMBLAS_INDEX_TYPE n,
                  ULMBLAS_INDEX_TYPE nrhs, const void *A,
                  ULMBLAS_INDEX_TYPE ldA, void *B, ULMBLAS_INDEX_TYPE ldB)
{
    return trtrs(order, upLo, trans, diag, n, nrhs,
                 static_cast<const std::complex<double> *>(A), ldA,
                 static_cast<std::complex<double> *>(B), ldB);
}

ULMBLAS_INDEX_TYPE
ULMLAPACK(ztrtri)(CBLAS_ORDER order, CBLAS_UPLO upLo, CBLAS_DIAG diag,
                  ULMBLAS_INDEX_TYPE n, void *A, ULMBLAS_INDEX_TYPE ldA)
{
    return trtri(order, upLo, diag, n, static_cast<std::complex<double> *>(A),
                 ldA);
}

ULMBLAS_INDEX_TYPE
ULMLAPACK(zlauum)(CBLAS_ORDER order, CBLAS_UPLO upLo, ULMBLAS_INDEX_TYPE n,
                  void *A, ULMBLAS_INDEX_TYPE ldA)
{
    return lauum(order, upLo, n, static_cast<std::complex<double> *>(A), ldA);
}

ULMBLAS_INDEX_TYPE
ULMLAPACK(zpotri)(CBLAS_ORDER order, CBLAS_UPLO upLo, ULMBLAS_INDEX_TYPE n,
                  void *A, ULMBLAS_INDEX_TYPE ldA)
{
    return potri(order, upLo, n, static_cast<std::complex<double> *>(A), ldA);
}

void
ULMLAPACK(zlaswp)(ULMBLAS_INDEX_TYPE n, void *A, ULMBLAS_INDEX_TYPE ldA,
                  ULMBLAS_INDEX_TYPE k1, ULMBLAS_INDEX_TYPE k2,
                  ULMBLAS_INDEX_TYPE *piv, ULMBLAS_INDEX_TYPE incPiv)
{
    ulmBLAS::laswp<Idx>(n, static_cast<std::complex<double> *>(A), 1, ldA, k1,
                        k2, piv, incPiv);
}

} // extern "C"
