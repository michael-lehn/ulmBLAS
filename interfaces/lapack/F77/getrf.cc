#include <complex>
#include <cstddef>
#include LAPACK_HEADER
#include <interfaces/blas/F77/xerbla.h>
#include <ulmblas/lapack/getrf.h>

//
//  ?GETRF( M, N, A, LDA, IPIV, INFO ) of LAPACK: LU factorization with
//  partial pivoting, P A = L U.  IPIV is 1-based.  INFO = -i: argument i is
//  invalid (reported by XERBLA); INFO = i > 0: U(i,i) is exactly zero, the
//  factorization has been completed.  A complex matrix is passed as pairs
//  of real numbers (real and imaginary part).
//

namespace {

template <typename T>
void
getrf(const char *name, const ULMBLAS_INDEX_TYPE *m_,
      const ULMBLAS_INDEX_TYPE *n_, T *A, const ULMBLAS_INDEX_TYPE *ldA_,
      ULMBLAS_INDEX_TYPE *ipiv, ULMBLAS_INDEX_TYPE *info_)
{
    const ULMBLAS_INDEX_TYPE m   = *m_;
    const ULMBLAS_INDEX_TYPE n   = *n_;
    const ULMBLAS_INDEX_TYPE ldA = *ldA_;

    ULMBLAS_INDEX_TYPE info = 0;
    if (m<0) {
        info = 1;
    } else if (n<0) {
        info = 2;
    } else if (ldA<(m>1 ? m : 1)) {
        info = 4;
    }
    if (info) {
        *info_ = -info;
        F77BLAS(xerbla)(name, &info, 6);
        return;
    }
    *info_ = 0;
    if (m==0 || n==0) {
        return;
    }
    *info_ = ulmBLAS::getrf<std::ptrdiff_t>(m, n, A, 1, ldA, ipiv, 1);
    const ULMBLAS_INDEX_TYPE k = (m<n) ? m : n;
    for (ULMBLAS_INDEX_TYPE i=0; i<k; ++i) {
        ++ipiv[i];
    }
}

} // namespace

extern "C" {

void
F77BLAS(sgetrf)(const ULMBLAS_INDEX_TYPE *m,
                const ULMBLAS_INDEX_TYPE *n,
                float                    *A,
                const ULMBLAS_INDEX_TYPE *ldA,
                ULMBLAS_INDEX_TYPE       *ipiv,
                ULMBLAS_INDEX_TYPE       *info)
{
    getrf("SGETRF ", m, n, A, ldA, ipiv, info);
}

void
F77BLAS(dgetrf)(const ULMBLAS_INDEX_TYPE *m,
                const ULMBLAS_INDEX_TYPE *n,
                double                   *A,
                const ULMBLAS_INDEX_TYPE *ldA,
                ULMBLAS_INDEX_TYPE       *ipiv,
                ULMBLAS_INDEX_TYPE       *info)
{
    getrf("DGETRF ", m, n, A, ldA, ipiv, info);
}

void
F77BLAS(cgetrf)(const ULMBLAS_INDEX_TYPE *m,
                const ULMBLAS_INDEX_TYPE *n,
                float                    *A,
                const ULMBLAS_INDEX_TYPE *ldA,
                ULMBLAS_INDEX_TYPE       *ipiv,
                ULMBLAS_INDEX_TYPE       *info)
{
    getrf("CGETRF ", m, n, reinterpret_cast<std::complex<float> *>(A), ldA, ipiv, info);
}

void
F77BLAS(zgetrf)(const ULMBLAS_INDEX_TYPE *m,
                const ULMBLAS_INDEX_TYPE *n,
                double                   *A,
                const ULMBLAS_INDEX_TYPE *ldA,
                ULMBLAS_INDEX_TYPE       *ipiv,
                ULMBLAS_INDEX_TYPE       *info)
{
    getrf("ZGETRF ", m, n, reinterpret_cast<std::complex<double> *>(A), ldA, ipiv, info);
}

} // extern "C"
