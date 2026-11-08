#include <cctype>
#include <complex>
#include <cstddef>
#include LAPACK_HEADER
#include <interfaces/blas/F77/xerbla.h>
#include <ulmblas/lapack/potrf.h>

//
//  ?POTRF( UPLO, N, A, LDA, INFO ) of LAPACK: Cholesky factorization of a
//  Hermitian (symmetric) positive definite matrix, A = U^H U (UPLO = 'U')
//  or A = L L^H (UPLO = 'L').  INFO = -i: argument i is invalid (reported
//  by XERBLA); INFO = i > 0: the leading minor of order i is not positive
//  definite.  A complex matrix is passed as pairs of real numbers.
//

namespace {

template <typename T>
void
potrf(const char *name, const char *upLo_, const ULMBLAS_INDEX_TYPE *n_,
      T *A, const ULMBLAS_INDEX_TYPE *ldA_, ULMBLAS_INDEX_TYPE *info_)
{
    const char               upLo = char(std::toupper(*upLo_));
    const ULMBLAS_INDEX_TYPE n    = *n_;
    const ULMBLAS_INDEX_TYPE ldA  = *ldA_;

    ULMBLAS_INDEX_TYPE info = 0;
    if (upLo!='U' && upLo!='L') {
        info = 1;
    } else if (n<0) {
        info = 2;
    } else if (ldA<(n>1 ? n : 1)) {
        info = 4;
    }
    if (info) {
        *info_ = -info;
        F77BLAS(xerbla)(name, &info, 6);
        return;
    }
    *info_ = 0;
    if (n==0) {
        return;
    }
    *info_ = ulmBLAS::potrf<std::ptrdiff_t>(upLo=='L', n, A, 1, ldA);
}

} // namespace

extern "C" {

void
F77BLAS(spotrf)(const char               *upLo,
                const ULMBLAS_INDEX_TYPE *n,
                float                    *A,
                const ULMBLAS_INDEX_TYPE *ldA,
                ULMBLAS_INDEX_TYPE       *info,
                std::size_t)
{
    potrf("SPOTRF ", upLo, n, A, ldA, info);
}

void
F77BLAS(dpotrf)(const char               *upLo,
                const ULMBLAS_INDEX_TYPE *n,
                double                   *A,
                const ULMBLAS_INDEX_TYPE *ldA,
                ULMBLAS_INDEX_TYPE       *info,
                std::size_t)
{
    potrf("DPOTRF ", upLo, n, A, ldA, info);
}

void
F77BLAS(cpotrf)(const char               *upLo,
                const ULMBLAS_INDEX_TYPE *n,
                float                    *A,
                const ULMBLAS_INDEX_TYPE *ldA,
                ULMBLAS_INDEX_TYPE       *info,
                std::size_t)
{
    potrf("CPOTRF ", upLo, n, reinterpret_cast<std::complex<float> *>(A), ldA, info);
}

void
F77BLAS(zpotrf)(const char               *upLo,
                const ULMBLAS_INDEX_TYPE *n,
                double                   *A,
                const ULMBLAS_INDEX_TYPE *ldA,
                ULMBLAS_INDEX_TYPE       *info,
                std::size_t)
{
    potrf("ZPOTRF ", upLo, n, reinterpret_cast<std::complex<double> *>(A), ldA, info);
}

} // extern "C"
