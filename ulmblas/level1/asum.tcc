#ifndef ULMBLAS_LEVEL1_ASUM_TCC
#define ULMBLAS_LEVEL1_ASUM_TCC 1

#include <cmath>
#include <ulmblas/level1/asum.h>

namespace ulmBLAS {

template <typename IndexType, typename VX>
VX
asum(IndexType  n,
     const VX   *x,
     IndexType  incX)
{
    VX result;

    asum(n, x, incX, result);
    return result;
}

template <typename IndexType, typename VX>
VX
asum(IndexType                n,
     const std::complex<VX>   *x,
     IndexType                incX)
{
    VX result;

    asum(n, x, incX, result);
    return result;
}

template <typename IndexType, typename VX, typename Result>
void
asum(IndexType  n,
     const VX   *x,
     IndexType  incX,
     Result     &result)
{
    result = Result(0);

    for (IndexType i=0; i<n; ++i) {
        result += std::abs(x[i*incX]);
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL1_ASUM_TCC 1

