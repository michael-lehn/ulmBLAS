#ifndef ULMBLAS_LEVEL1_SDSDOT_TCC
#define ULMBLAS_LEVEL1_SDSDOT_TCC 1

#include <ulmblas/level1/sdsdot.h>

#include <type_traits>
#include <ulmblas/level1/kernel/reduce.h>

namespace ulmBLAS {

//
//  result += x^T y with the products and the sum computed in the precision
//  of Result (SDSDOT, DSDOT: single precision vectors, double precision
//  accumulation).  Vectors of the same floating-point type: partial sums
//  (ulmblas/level1/kernel/reduce.h), otherwise in the order of the
//  elements as in the Reference BLAS.
//
template <typename IndexType, typename T, typename Result>
void
sdsdotImpl(IndexType n, const T *x, IndexType incX, const T *y, IndexType incY,
           Result &result, std::true_type simd)
{
    result += reduce::dot<Result>(n, x, incX, y, incY);
}

template <typename IndexType, typename VX, typename VY, typename Result>
void
sdsdotImpl(IndexType n, const VX *x, IndexType incX, const VY *y,
           IndexType incY, Result &result, std::false_type simd)
{
    for (IndexType i = 0; i < n; ++i) {
	result += Result(x[i * incX]) * Result(y[i * incY]);
    }
}

template <typename IndexType, typename VX, typename VY, typename Result>
void
sdsdot(IndexType n, const VX *x, IndexType incX, const VY *y, IndexType incY,
       Result &result)
{
    typedef std::integral_constant<bool,
                                   std::is_same<VX, VY>::value &&
                                       std::is_floating_point<VX>::value &&
                                       std::is_floating_point<Result>::value>
        Simd;
    sdsdotImpl(n, x, incX, y, incY, result, Simd());
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL1_SDSDOT_TCC 1
