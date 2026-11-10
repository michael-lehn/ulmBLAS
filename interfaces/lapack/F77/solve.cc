#include <cctype>
#include <complex>
#include <cstddef>
#include <utility>
#include LAPACK_HEADER
#include <interfaces/blas/F77/xerbla.h>
#include <ulmblas/lapack/buffer.h>
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
//  Solves and inverses with the arguments of LAPACK: ?GETRS, ?GESV,
//  ?GETRI, ?POTRS, ?POSV, ?TRTRS, ?TRTRI, ?LAUUM, ?POTRI, ?LASWP.  IPIV is
//  1-based, INFO = -i reports an invalid argument i (with XERBLA), the
//  length of CHARACTER arguments follows the other arguments.  ?GETRI
//  needs no work space (WORK, LWORK are checked as in LAPACK; a query,
//  LWORK = -1, returns N).
//

namespace {

typedef ULMBLAS_INDEX_TYPE F77Int;
typedef std::ptrdiff_t Idx;

F77Int
max1(F77Int n)
{
    return n > 1 ? n : 1;
}
char
up(const char *c)
{
    return char(std::toupper(*c));
}

bool
error(const char *name, F77Int info, F77Int *info_)
{
    if (info) {
	*info_ = -info;
	F77BLAS(xerbla)(name, &info, 6);
	return true;
    }
    *info_ = 0;
    return false;
}

//  1-based IPIV(1:n) to 0-based
struct Pivots
{
    Pivots(const F77Int *ipiv, F77Int n) : p(std::size_t(n > 0 ? n : 0))
    {
	for (F77Int i = 0; i < n; ++i) {
	    p[i] = Idx(ipiv[i] - 1);
	}
    }
    ulmBLAS::Buffer<Idx> p;
};

template <typename T>
void
getrs(const char *name, const char *trans_, const F77Int *n_,
      const F77Int *nrhs_, const T *A, const F77Int *ldA_, const F77Int *ipiv,
      T *B, const F77Int *ldB_, F77Int *info_)
{
    const char t = up(trans_);
    const F77Int n = *n_, nrhs = *nrhs_, ldA = *ldA_, ldB = *ldB_;
    F77Int info = 0;
    if (t != 'N' && t != 'T' && t != 'C')
	info = 1;
    else if (n < 0)
	info = 2;
    else if (nrhs < 0)
	info = 3;
    else if (ldA < max1(n))
	info = 5;
    else if (ldB < max1(n))
	info = 8;
    if (error(name, info, info_) || n == 0 || nrhs == 0) {
	return;
    }
    Pivots p(ipiv, n);
    ulmBLAS::getrs<Idx>(t != 'N', t == 'C', n, nrhs, A, 1, ldA, p.p.data(), 1,
                        B, 1, ldB);
}

template <typename T>
void
gesv(const char *name, const F77Int *n_, const F77Int *nrhs_, T *A,
     const F77Int *ldA_, F77Int *ipiv, T *B, const F77Int *ldB_, F77Int *info_)
{
    const F77Int n = *n_, nrhs = *nrhs_, ldA = *ldA_, ldB = *ldB_;
    F77Int info = 0;
    if (n < 0)
	info = 1;
    else if (nrhs < 0)
	info = 2;
    else if (ldA < max1(n))
	info = 4;
    else if (ldB < max1(n))
	info = 7;
    if (error(name, info, info_) || n == 0) {
	return;
    }
    *info_ = ulmBLAS::gesv<Idx>(n, nrhs, A, 1, ldA, ipiv, 1, B, 1, ldB);
    for (F77Int i = 0; i < n; ++i) {
	++ipiv[i];
    }
}

template <typename T>
void
getri(const char *name, const F77Int *n_, T *A, const F77Int *ldA_,
      const F77Int *ipiv, T *work, const F77Int *lwork_, F77Int *info_)
{
    const F77Int n = *n_, ldA = *ldA_, lwork = *lwork_;
    F77Int info = 0;
    if (n < 0)
	info = 1;
    else if (ldA < max1(n))
	info = 3;
    else if (lwork < max1(n) && lwork != -1)
	info = 6;
    if (error(name, info, info_)) {
	return;
    }
    work[0] = T(max1(n));
    if (lwork == -1 || n == 0) {
	return;
    }
    Pivots p(ipiv, n);
    *info_ = ulmBLAS::getri<Idx>(n, A, 1, ldA, p.p.data(), 1);
}

template <typename T>
void
potrs(const char *name, const char *upLo_, const F77Int *n_,
      const F77Int *nrhs_, const T *A, const F77Int *ldA_, T *B,
      const F77Int *ldB_, F77Int *info_)
{
    const char u = up(upLo_);
    const F77Int n = *n_, nrhs = *nrhs_, ldA = *ldA_, ldB = *ldB_;
    F77Int info = 0;
    if (u != 'U' && u != 'L')
	info = 1;
    else if (n < 0)
	info = 2;
    else if (nrhs < 0)
	info = 3;
    else if (ldA < max1(n))
	info = 5;
    else if (ldB < max1(n))
	info = 7;
    if (error(name, info, info_) || n == 0 || nrhs == 0) {
	return;
    }
    ulmBLAS::potrs<Idx>(u == 'L', n, nrhs, A, 1, ldA, B, 1, ldB);
}

template <typename T>
void
posv(const char *name, const char *upLo_, const F77Int *n_, const F77Int *nrhs_,
     T *A, const F77Int *ldA_, T *B, const F77Int *ldB_, F77Int *info_)
{
    const char u = up(upLo_);
    const F77Int n = *n_, nrhs = *nrhs_, ldA = *ldA_, ldB = *ldB_;
    F77Int info = 0;
    if (u != 'U' && u != 'L')
	info = 1;
    else if (n < 0)
	info = 2;
    else if (nrhs < 0)
	info = 3;
    else if (ldA < max1(n))
	info = 5;
    else if (ldB < max1(n))
	info = 7;
    if (error(name, info, info_) || n == 0) {
	return;
    }
    *info_ = ulmBLAS::posv<Idx>(u == 'L', n, nrhs, A, 1, ldA, B, 1, ldB);
}

template <typename T>
void
trtrs(const char *name, const char *upLo_, const char *trans_,
      const char *diag_, const F77Int *n_, const F77Int *nrhs_, const T *A,
      const F77Int *ldA_, T *B, const F77Int *ldB_, F77Int *info_)
{
    const char u = up(upLo_), t = up(trans_), d = up(diag_);
    const F77Int n = *n_, nrhs = *nrhs_, ldA = *ldA_, ldB = *ldB_;
    F77Int info = 0;
    if (u != 'U' && u != 'L')
	info = 1;
    else if (t != 'N' && t != 'T' && t != 'C')
	info = 2;
    else if (d != 'U' && d != 'N')
	info = 3;
    else if (n < 0)
	info = 4;
    else if (nrhs < 0)
	info = 5;
    else if (ldA < max1(n))
	info = 7;
    else if (ldB < max1(n))
	info = 9;
    if (error(name, info, info_) || n == 0) {
	return;
    }
    *info_ = ulmBLAS::trtrs<Idx>(u == 'L', t != 'N', t == 'C', d == 'U', n,
                                 nrhs, A, 1, ldA, B, 1, ldB);
}

template <typename T>
void
trtri(const char *name, const char *upLo_, const char *diag_, const F77Int *n_,
      T *A, const F77Int *ldA_, F77Int *info_)
{
    const char u = up(upLo_), d = up(diag_);
    const F77Int n = *n_, ldA = *ldA_;
    F77Int info = 0;
    if (u != 'U' && u != 'L')
	info = 1;
    else if (d != 'U' && d != 'N')
	info = 2;
    else if (n < 0)
	info = 3;
    else if (ldA < max1(n))
	info = 5;
    if (error(name, info, info_) || n == 0) {
	return;
    }
    *info_ = ulmBLAS::trtri<Idx>(u == 'L', d == 'U', n, A, 1, ldA);
}

template <typename T>
void
lauum(const char *name, const char *upLo_, const F77Int *n_, T *A,
      const F77Int *ldA_, F77Int *info_)
{
    const char u = up(upLo_);
    const F77Int n = *n_, ldA = *ldA_;
    F77Int info = 0;
    if (u != 'U' && u != 'L')
	info = 1;
    else if (n < 0)
	info = 2;
    else if (ldA < max1(n))
	info = 4;
    if (error(name, info, info_) || n == 0) {
	return;
    }
    ulmBLAS::lauum<Idx>(u == 'L', n, A, 1, ldA);
}

template <typename T>
void
potri(const char *name, const char *upLo_, const F77Int *n_, T *A,
      const F77Int *ldA_, F77Int *info_)
{
    const char u = up(upLo_);
    const F77Int n = *n_, ldA = *ldA_;
    F77Int info = 0;
    if (u != 'U' && u != 'L')
	info = 1;
    else if (n < 0)
	info = 2;
    else if (ldA < max1(n))
	info = 4;
    if (error(name, info, info_) || n == 0) {
	return;
    }
    *info_ = ulmBLAS::potri<Idx>(u == 'L', n, A, 1, ldA);
}

//  ?LASWP( N, A, LDA, K1, K2, IPIV, INCX ): rows I and IPIV(K1 + (I-K1)*
//  |INCX|) are interchanged for I = K1, ..., K2 (INCX > 0) or I = K2, ...,
//  K1 (INCX < 0), as in LAPACK; INCX = 0: nothing
template <typename T>
void
laswp(const F77Int *n_, T *A, const F77Int *ldA_, const F77Int *k1_,
      const F77Int *k2_, const F77Int *ipiv, const F77Int *incx_)
{
    const F77Int n = *n_, ldA = *ldA_, k1 = *k1_, k2 = *k2_, incx = *incx_;
    if (incx == 0 || n <= 0) {
	return;
    }
    const F77Int ai = incx > 0 ? incx : -incx;
    for (F77Int j = 0; j < n; ++j) {
	T *a = &A[j * ldA];
	for (F77Int c = 0; c <= k2 - k1; ++c) {
	    const F77Int i = incx > 0 ? k1 + c : k2 - c; // 1-based row
	    const F77Int ip = ipiv[(i - k1) * ai];
	    if (ip != i) {
		std::swap(a[i - 1], a[ip - 1]);
	    }
	}
    }
}

} // namespace

extern "C" {

void
F77BLAS(sgetrs)(const char *trans, const ULMBLAS_INDEX_TYPE *n,
                const ULMBLAS_INDEX_TYPE *nrhs, const float *A,
                const ULMBLAS_INDEX_TYPE *ldA, const ULMBLAS_INDEX_TYPE *ipiv,
                float *B, const ULMBLAS_INDEX_TYPE *ldB,
                ULMBLAS_INDEX_TYPE *info, std::size_t)
{
    getrs("SGETRS ", trans, n, nrhs, A, ldA, ipiv, B, ldB, info);
}

void
F77BLAS(sgesv)(const ULMBLAS_INDEX_TYPE *n, const ULMBLAS_INDEX_TYPE *nrhs,
               float *A, const ULMBLAS_INDEX_TYPE *ldA,
               ULMBLAS_INDEX_TYPE *ipiv, float *B,
               const ULMBLAS_INDEX_TYPE *ldB, ULMBLAS_INDEX_TYPE *info)
{
    gesv("SGESV  ", n, nrhs, A, ldA, ipiv, B, ldB, info);
}

void
F77BLAS(sgetri)(const ULMBLAS_INDEX_TYPE *n, float *A,
                const ULMBLAS_INDEX_TYPE *ldA, const ULMBLAS_INDEX_TYPE *ipiv,
                float *work, const ULMBLAS_INDEX_TYPE *lwork,
                ULMBLAS_INDEX_TYPE *info)
{
    getri("SGETRI ", n, A, ldA, ipiv, work, lwork, info);
}

void
F77BLAS(spotrs)(const char *upLo, const ULMBLAS_INDEX_TYPE *n,
                const ULMBLAS_INDEX_TYPE *nrhs, const float *A,
                const ULMBLAS_INDEX_TYPE *ldA, float *B,
                const ULMBLAS_INDEX_TYPE *ldB, ULMBLAS_INDEX_TYPE *info,
                std::size_t)
{
    potrs("SPOTRS ", upLo, n, nrhs, A, ldA, B, ldB, info);
}

void
F77BLAS(sposv)(const char *upLo, const ULMBLAS_INDEX_TYPE *n,
               const ULMBLAS_INDEX_TYPE *nrhs, float *A,
               const ULMBLAS_INDEX_TYPE *ldA, float *B,
               const ULMBLAS_INDEX_TYPE *ldB, ULMBLAS_INDEX_TYPE *info,
               std::size_t)
{
    posv("SPOSV  ", upLo, n, nrhs, A, ldA, B, ldB, info);
}

void
F77BLAS(strtrs)(const char *upLo, const char *trans, const char *diag,
                const ULMBLAS_INDEX_TYPE *n, const ULMBLAS_INDEX_TYPE *nrhs,
                const float *A, const ULMBLAS_INDEX_TYPE *ldA, float *B,
                const ULMBLAS_INDEX_TYPE *ldB, ULMBLAS_INDEX_TYPE *info,
                std::size_t, std::size_t, std::size_t)
{
    trtrs("STRTRS ", upLo, trans, diag, n, nrhs, A, ldA, B, ldB, info);
}

void
F77BLAS(strtri)(const char *upLo, const char *diag, const ULMBLAS_INDEX_TYPE *n,
                float *A, const ULMBLAS_INDEX_TYPE *ldA,
                ULMBLAS_INDEX_TYPE *info, std::size_t, std::size_t)
{
    trtri("STRTRI ", upLo, diag, n, A, ldA, info);
}

void
F77BLAS(slauum)(const char *upLo, const ULMBLAS_INDEX_TYPE *n, float *A,
                const ULMBLAS_INDEX_TYPE *ldA, ULMBLAS_INDEX_TYPE *info,
                std::size_t)
{
    lauum("SLAUUM ", upLo, n, A, ldA, info);
}

void
F77BLAS(spotri)(const char *upLo, const ULMBLAS_INDEX_TYPE *n, float *A,
                const ULMBLAS_INDEX_TYPE *ldA, ULMBLAS_INDEX_TYPE *info,
                std::size_t)
{
    potri("SPOTRI ", upLo, n, A, ldA, info);
}

void
F77BLAS(slaswp)(const ULMBLAS_INDEX_TYPE *n, float *A,
                const ULMBLAS_INDEX_TYPE *ldA, const ULMBLAS_INDEX_TYPE *k1,
                const ULMBLAS_INDEX_TYPE *k2, const ULMBLAS_INDEX_TYPE *ipiv,
                const ULMBLAS_INDEX_TYPE *incx)
{
    laswp(n, A, ldA, k1, k2, ipiv, incx);
}

void
F77BLAS(dgetrs)(const char *trans, const ULMBLAS_INDEX_TYPE *n,
                const ULMBLAS_INDEX_TYPE *nrhs, const double *A,
                const ULMBLAS_INDEX_TYPE *ldA, const ULMBLAS_INDEX_TYPE *ipiv,
                double *B, const ULMBLAS_INDEX_TYPE *ldB,
                ULMBLAS_INDEX_TYPE *info, std::size_t)
{
    getrs("DGETRS ", trans, n, nrhs, A, ldA, ipiv, B, ldB, info);
}

void
F77BLAS(dgesv)(const ULMBLAS_INDEX_TYPE *n, const ULMBLAS_INDEX_TYPE *nrhs,
               double *A, const ULMBLAS_INDEX_TYPE *ldA,
               ULMBLAS_INDEX_TYPE *ipiv, double *B,
               const ULMBLAS_INDEX_TYPE *ldB, ULMBLAS_INDEX_TYPE *info)
{
    gesv("DGESV  ", n, nrhs, A, ldA, ipiv, B, ldB, info);
}

void
F77BLAS(dgetri)(const ULMBLAS_INDEX_TYPE *n, double *A,
                const ULMBLAS_INDEX_TYPE *ldA, const ULMBLAS_INDEX_TYPE *ipiv,
                double *work, const ULMBLAS_INDEX_TYPE *lwork,
                ULMBLAS_INDEX_TYPE *info)
{
    getri("DGETRI ", n, A, ldA, ipiv, work, lwork, info);
}

void
F77BLAS(dpotrs)(const char *upLo, const ULMBLAS_INDEX_TYPE *n,
                const ULMBLAS_INDEX_TYPE *nrhs, const double *A,
                const ULMBLAS_INDEX_TYPE *ldA, double *B,
                const ULMBLAS_INDEX_TYPE *ldB, ULMBLAS_INDEX_TYPE *info,
                std::size_t)
{
    potrs("DPOTRS ", upLo, n, nrhs, A, ldA, B, ldB, info);
}

void
F77BLAS(dposv)(const char *upLo, const ULMBLAS_INDEX_TYPE *n,
               const ULMBLAS_INDEX_TYPE *nrhs, double *A,
               const ULMBLAS_INDEX_TYPE *ldA, double *B,
               const ULMBLAS_INDEX_TYPE *ldB, ULMBLAS_INDEX_TYPE *info,
               std::size_t)
{
    posv("DPOSV  ", upLo, n, nrhs, A, ldA, B, ldB, info);
}

void
F77BLAS(dtrtrs)(const char *upLo, const char *trans, const char *diag,
                const ULMBLAS_INDEX_TYPE *n, const ULMBLAS_INDEX_TYPE *nrhs,
                const double *A, const ULMBLAS_INDEX_TYPE *ldA, double *B,
                const ULMBLAS_INDEX_TYPE *ldB, ULMBLAS_INDEX_TYPE *info,
                std::size_t, std::size_t, std::size_t)
{
    trtrs("DTRTRS ", upLo, trans, diag, n, nrhs, A, ldA, B, ldB, info);
}

void
F77BLAS(dtrtri)(const char *upLo, const char *diag, const ULMBLAS_INDEX_TYPE *n,
                double *A, const ULMBLAS_INDEX_TYPE *ldA,
                ULMBLAS_INDEX_TYPE *info, std::size_t, std::size_t)
{
    trtri("DTRTRI ", upLo, diag, n, A, ldA, info);
}

void
F77BLAS(dlauum)(const char *upLo, const ULMBLAS_INDEX_TYPE *n, double *A,
                const ULMBLAS_INDEX_TYPE *ldA, ULMBLAS_INDEX_TYPE *info,
                std::size_t)
{
    lauum("DLAUUM ", upLo, n, A, ldA, info);
}

void
F77BLAS(dpotri)(const char *upLo, const ULMBLAS_INDEX_TYPE *n, double *A,
                const ULMBLAS_INDEX_TYPE *ldA, ULMBLAS_INDEX_TYPE *info,
                std::size_t)
{
    potri("DPOTRI ", upLo, n, A, ldA, info);
}

void
F77BLAS(dlaswp)(const ULMBLAS_INDEX_TYPE *n, double *A,
                const ULMBLAS_INDEX_TYPE *ldA, const ULMBLAS_INDEX_TYPE *k1,
                const ULMBLAS_INDEX_TYPE *k2, const ULMBLAS_INDEX_TYPE *ipiv,
                const ULMBLAS_INDEX_TYPE *incx)
{
    laswp(n, A, ldA, k1, k2, ipiv, incx);
}

void
F77BLAS(cgetrs)(const char *trans, const ULMBLAS_INDEX_TYPE *n,
                const ULMBLAS_INDEX_TYPE *nrhs, const float *A,
                const ULMBLAS_INDEX_TYPE *ldA, const ULMBLAS_INDEX_TYPE *ipiv,
                float *B, const ULMBLAS_INDEX_TYPE *ldB,
                ULMBLAS_INDEX_TYPE *info, std::size_t)
{
    getrs("CGETRS ", trans, n, nrhs,
          reinterpret_cast<const std::complex<float> *>(A), ldA, ipiv,
          reinterpret_cast<std::complex<float> *>(B), ldB, info);
}

void
F77BLAS(cgesv)(const ULMBLAS_INDEX_TYPE *n, const ULMBLAS_INDEX_TYPE *nrhs,
               float *A, const ULMBLAS_INDEX_TYPE *ldA,
               ULMBLAS_INDEX_TYPE *ipiv, float *B,
               const ULMBLAS_INDEX_TYPE *ldB, ULMBLAS_INDEX_TYPE *info)
{
    gesv("CGESV  ", n, nrhs, reinterpret_cast<std::complex<float> *>(A), ldA,
         ipiv, reinterpret_cast<std::complex<float> *>(B), ldB, info);
}

void
F77BLAS(cgetri)(const ULMBLAS_INDEX_TYPE *n, float *A,
                const ULMBLAS_INDEX_TYPE *ldA, const ULMBLAS_INDEX_TYPE *ipiv,
                float *work, const ULMBLAS_INDEX_TYPE *lwork,
                ULMBLAS_INDEX_TYPE *info)
{
    getri("CGETRI ", n, reinterpret_cast<std::complex<float> *>(A), ldA, ipiv,
          reinterpret_cast<std::complex<float> *>(work), lwork, info);
}

void
F77BLAS(cpotrs)(const char *upLo, const ULMBLAS_INDEX_TYPE *n,
                const ULMBLAS_INDEX_TYPE *nrhs, const float *A,
                const ULMBLAS_INDEX_TYPE *ldA, float *B,
                const ULMBLAS_INDEX_TYPE *ldB, ULMBLAS_INDEX_TYPE *info,
                std::size_t)
{
    potrs("CPOTRS ", upLo, n, nrhs,
          reinterpret_cast<const std::complex<float> *>(A), ldA,
          reinterpret_cast<std::complex<float> *>(B), ldB, info);
}

void
F77BLAS(cposv)(const char *upLo, const ULMBLAS_INDEX_TYPE *n,
               const ULMBLAS_INDEX_TYPE *nrhs, float *A,
               const ULMBLAS_INDEX_TYPE *ldA, float *B,
               const ULMBLAS_INDEX_TYPE *ldB, ULMBLAS_INDEX_TYPE *info,
               std::size_t)
{
    posv("CPOSV  ", upLo, n, nrhs, reinterpret_cast<std::complex<float> *>(A),
         ldA, reinterpret_cast<std::complex<float> *>(B), ldB, info);
}

void
F77BLAS(ctrtrs)(const char *upLo, const char *trans, const char *diag,
                const ULMBLAS_INDEX_TYPE *n, const ULMBLAS_INDEX_TYPE *nrhs,
                const float *A, const ULMBLAS_INDEX_TYPE *ldA, float *B,
                const ULMBLAS_INDEX_TYPE *ldB, ULMBLAS_INDEX_TYPE *info,
                std::size_t, std::size_t, std::size_t)
{
    trtrs("CTRTRS ", upLo, trans, diag, n, nrhs,
          reinterpret_cast<const std::complex<float> *>(A), ldA,
          reinterpret_cast<std::complex<float> *>(B), ldB, info);
}

void
F77BLAS(ctrtri)(const char *upLo, const char *diag, const ULMBLAS_INDEX_TYPE *n,
                float *A, const ULMBLAS_INDEX_TYPE *ldA,
                ULMBLAS_INDEX_TYPE *info, std::size_t, std::size_t)
{
    trtri("CTRTRI ", upLo, diag, n, reinterpret_cast<std::complex<float> *>(A),
          ldA, info);
}

void
F77BLAS(clauum)(const char *upLo, const ULMBLAS_INDEX_TYPE *n, float *A,
                const ULMBLAS_INDEX_TYPE *ldA, ULMBLAS_INDEX_TYPE *info,
                std::size_t)
{
    lauum("CLAUUM ", upLo, n, reinterpret_cast<std::complex<float> *>(A), ldA,
          info);
}

void
F77BLAS(cpotri)(const char *upLo, const ULMBLAS_INDEX_TYPE *n, float *A,
                const ULMBLAS_INDEX_TYPE *ldA, ULMBLAS_INDEX_TYPE *info,
                std::size_t)
{
    potri("CPOTRI ", upLo, n, reinterpret_cast<std::complex<float> *>(A), ldA,
          info);
}

void
F77BLAS(claswp)(const ULMBLAS_INDEX_TYPE *n, float *A,
                const ULMBLAS_INDEX_TYPE *ldA, const ULMBLAS_INDEX_TYPE *k1,
                const ULMBLAS_INDEX_TYPE *k2, const ULMBLAS_INDEX_TYPE *ipiv,
                const ULMBLAS_INDEX_TYPE *incx)
{
    laswp(n, reinterpret_cast<std::complex<float> *>(A), ldA, k1, k2, ipiv,
          incx);
}

void
F77BLAS(zgetrs)(const char *trans, const ULMBLAS_INDEX_TYPE *n,
                const ULMBLAS_INDEX_TYPE *nrhs, const double *A,
                const ULMBLAS_INDEX_TYPE *ldA, const ULMBLAS_INDEX_TYPE *ipiv,
                double *B, const ULMBLAS_INDEX_TYPE *ldB,
                ULMBLAS_INDEX_TYPE *info, std::size_t)
{
    getrs("ZGETRS ", trans, n, nrhs,
          reinterpret_cast<const std::complex<double> *>(A), ldA, ipiv,
          reinterpret_cast<std::complex<double> *>(B), ldB, info);
}

void
F77BLAS(zgesv)(const ULMBLAS_INDEX_TYPE *n, const ULMBLAS_INDEX_TYPE *nrhs,
               double *A, const ULMBLAS_INDEX_TYPE *ldA,
               ULMBLAS_INDEX_TYPE *ipiv, double *B,
               const ULMBLAS_INDEX_TYPE *ldB, ULMBLAS_INDEX_TYPE *info)
{
    gesv("ZGESV  ", n, nrhs, reinterpret_cast<std::complex<double> *>(A), ldA,
         ipiv, reinterpret_cast<std::complex<double> *>(B), ldB, info);
}

void
F77BLAS(zgetri)(const ULMBLAS_INDEX_TYPE *n, double *A,
                const ULMBLAS_INDEX_TYPE *ldA, const ULMBLAS_INDEX_TYPE *ipiv,
                double *work, const ULMBLAS_INDEX_TYPE *lwork,
                ULMBLAS_INDEX_TYPE *info)
{
    getri("ZGETRI ", n, reinterpret_cast<std::complex<double> *>(A), ldA, ipiv,
          reinterpret_cast<std::complex<double> *>(work), lwork, info);
}

void
F77BLAS(zpotrs)(const char *upLo, const ULMBLAS_INDEX_TYPE *n,
                const ULMBLAS_INDEX_TYPE *nrhs, const double *A,
                const ULMBLAS_INDEX_TYPE *ldA, double *B,
                const ULMBLAS_INDEX_TYPE *ldB, ULMBLAS_INDEX_TYPE *info,
                std::size_t)
{
    potrs("ZPOTRS ", upLo, n, nrhs,
          reinterpret_cast<const std::complex<double> *>(A), ldA,
          reinterpret_cast<std::complex<double> *>(B), ldB, info);
}

void
F77BLAS(zposv)(const char *upLo, const ULMBLAS_INDEX_TYPE *n,
               const ULMBLAS_INDEX_TYPE *nrhs, double *A,
               const ULMBLAS_INDEX_TYPE *ldA, double *B,
               const ULMBLAS_INDEX_TYPE *ldB, ULMBLAS_INDEX_TYPE *info,
               std::size_t)
{
    posv("ZPOSV  ", upLo, n, nrhs, reinterpret_cast<std::complex<double> *>(A),
         ldA, reinterpret_cast<std::complex<double> *>(B), ldB, info);
}

void
F77BLAS(ztrtrs)(const char *upLo, const char *trans, const char *diag,
                const ULMBLAS_INDEX_TYPE *n, const ULMBLAS_INDEX_TYPE *nrhs,
                const double *A, const ULMBLAS_INDEX_TYPE *ldA, double *B,
                const ULMBLAS_INDEX_TYPE *ldB, ULMBLAS_INDEX_TYPE *info,
                std::size_t, std::size_t, std::size_t)
{
    trtrs("ZTRTRS ", upLo, trans, diag, n, nrhs,
          reinterpret_cast<const std::complex<double> *>(A), ldA,
          reinterpret_cast<std::complex<double> *>(B), ldB, info);
}

void
F77BLAS(ztrtri)(const char *upLo, const char *diag, const ULMBLAS_INDEX_TYPE *n,
                double *A, const ULMBLAS_INDEX_TYPE *ldA,
                ULMBLAS_INDEX_TYPE *info, std::size_t, std::size_t)
{
    trtri("ZTRTRI ", upLo, diag, n, reinterpret_cast<std::complex<double> *>(A),
          ldA, info);
}

void
F77BLAS(zlauum)(const char *upLo, const ULMBLAS_INDEX_TYPE *n, double *A,
                const ULMBLAS_INDEX_TYPE *ldA, ULMBLAS_INDEX_TYPE *info,
                std::size_t)
{
    lauum("ZLAUUM ", upLo, n, reinterpret_cast<std::complex<double> *>(A), ldA,
          info);
}

void
F77BLAS(zpotri)(const char *upLo, const ULMBLAS_INDEX_TYPE *n, double *A,
                const ULMBLAS_INDEX_TYPE *ldA, ULMBLAS_INDEX_TYPE *info,
                std::size_t)
{
    potri("ZPOTRI ", upLo, n, reinterpret_cast<std::complex<double> *>(A), ldA,
          info);
}

void
F77BLAS(zlaswp)(const ULMBLAS_INDEX_TYPE *n, double *A,
                const ULMBLAS_INDEX_TYPE *ldA, const ULMBLAS_INDEX_TYPE *k1,
                const ULMBLAS_INDEX_TYPE *k2, const ULMBLAS_INDEX_TYPE *ipiv,
                const ULMBLAS_INDEX_TYPE *incx)
{
    laswp(n, reinterpret_cast<std::complex<double> *>(A), ldA, k1, k2, ipiv,
          incx);
}

} // extern "C"
