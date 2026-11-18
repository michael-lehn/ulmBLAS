#ifndef ULMBLAS_LEVEL1_ASUM_TCC
#define ULMBLAS_LEVEL1_ASUM_TCC 1

#include <cmath>
#include <complex>
#include <type_traits>
#include <ulmblas/level1/kernel/reduce.h>
#include <ulmblas/level1/asum.h>

namespace ulmBLAS {

template <typename IndexType, typename VX>
VX
asum(IndexType n, const VX *x, IndexType incX)
{
    VX result;

    asum(n, x, incX, result);
    return result;
}

//  complex: |re| + |im| summed with partial sums (ulmblas/level1/kernel/
//  reduce.h); contiguous: the 2n real numbers
template <typename IndexType, typename VX>
VX
asum(IndexType n, const std::complex<VX> *x, IndexType incX)
{
    if (incX == IndexType(1)) {
	return reduce::sumAbs<VX>(2 * n, reinterpret_cast<const VX *>(x),
	                          IndexType(1));
    }
    return reduce::sumAbs<VX>(n, x, incX);
}

//  x of a floating-point type: partial sums (ulmblas/level1/kernel/
//  reduce.h)
template <typename IndexType, typename VX, typename Result>
void
asumImpl(IndexType n, const VX *x, IndexType incX, Result &result,
         std::true_type simd)
{
    result = reduce::sumAbs<Result>(n, x, incX);
}

template <typename IndexType, typename VX, typename Result>
void
asumImpl(IndexType n, const VX *x, IndexType incX, Result &result,
         std::false_type simd)
{
    result = Result(0);

    for (IndexType i = 0; i < n; ++i) {
	result += std::abs(x[i * incX]);
    }
}

template <typename IndexType, typename VX, typename Result>
void
asum(IndexType n, const VX *x, IndexType incX, Result &result)
{
    typedef std::integral_constant<bool,
                                   std::is_floating_point<VX>::value &&
                                       std::is_floating_point<Result>::value>
        Simd;
    asumImpl(n, x, incX, result, Simd());
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL1_ASUM_TCC 1
