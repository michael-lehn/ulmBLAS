#ifndef ULMBLAS_LEVEL1_SCAL_TCC
#define ULMBLAS_LEVEL1_SCAL_TCC 1

#include <ulmblas/level1/scal.h>

namespace ulmBLAS {

//
//  x = alpha*x; for alpha = 0 the elements are set to zero (also NaN and
//  Inf), as required for beta = 0 in level 2 and 3.
//
template <typename IndexType, typename Alpha, typename VX>
void
scal(IndexType      n,
     const Alpha    &alpha,
     VX             *x,
     IndexType      incX)
{
    if (alpha!=Alpha(1) && alpha!=Alpha(0)) {
        for (IndexType i=0; i<n; ++i) {
            x[i*incX] *= alpha;
        }
    } else if (alpha==Alpha(0)) {
        for (IndexType i=0; i<n; ++i) {
            x[i*incX] = Alpha(0);
        }
    }
}

//
//  x = alpha*x as ?SCAL of the reference BLAS: x is unchanged for alpha = 1,
//  otherwise every element is multiplied, also for alpha = 0 (NaN and Inf
//  give NaN).  Complex products are computed as in Fortran, without the
//  special treatment of infinities of std::complex.
//
template <typename Alpha, typename VX>
VX
scalBlasProduct(const Alpha &alpha, const VX &x)
{
    return alpha*x;
}

template <typename T>
std::complex<T>
scalBlasProduct(const std::complex<T> &alpha, const std::complex<T> &x)
{
    return std::complex<T>(alpha.real()*x.real() - alpha.imag()*x.imag(),
                           alpha.real()*x.imag() + alpha.imag()*x.real());
}

template <typename IndexType, typename Alpha, typename VX>
void
scalBlas(IndexType      n,
         const Alpha    &alpha,
         VX             *x,
         IndexType      incX)
{
    if (alpha==Alpha(1)) {
        return;
    }
    for (IndexType i=0; i<n; ++i) {
        x[i*incX] = scalBlasProduct(alpha, x[i*incX]);
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL1_SCAL_TCC 1
