#ifndef ULMBLAS_LEVEL1_KERNEL_REF_DOT_TCC
#define ULMBLAS_LEVEL1_KERNEL_REF_DOT_TCC 1

#include <ulmblas/auxiliary/conjugate.h>
#include <ulmblas/level1/kernel/ref/dot.h>

#include <complex>
#include <type_traits>
#include <ulmblas/level1/kernel/reduce.h>

namespace ulmBLAS {
namespace ref {

//
//  Vectors of the type of the result (float, double and their complex
//  types): partial sums (ulmblas/level1/kernel/reduce.h).
//
template <typename T> struct DotSimd : std::is_floating_point<T>
{
};

template <typename T>
struct DotSimd<std::complex<T>> : std::is_floating_point<T>
{
};

template <typename IndexType, typename T>
bool
dotLanes(IndexType n, bool conjX, const T *x, IndexType incX, const T *y,
         IndexType incY, T &result, std::true_type simd)
{
    result = reduce::dot<T>(n, x, incX, y, incY);
    return true;
}

template <typename IndexType, typename T>
bool
dotLanes(IndexType n, bool conjX, const std::complex<T> *x, IndexType incX,
         const std::complex<T> *y, IndexType incY, std::complex<T> &result,
         std::true_type simd)
{
    result = reduce::dot(n, conjX, x, incX, y, incY);
    return true;
}

template <typename IndexType, typename VX, typename VY, typename Result>
bool
dotLanes(IndexType n, bool conjX, const VX *x, IndexType incX, const VY *y,
         IndexType incY, Result &result, std::false_type simd)
{
    return false;
}

template <typename IndexType, typename VX, typename VY, typename Result>
bool
dotLanes(IndexType n, bool conjX, const VX *x, IndexType incX, const VY *y,
         IndexType incY, Result &result)
{
    typedef std::integral_constant<bool, std::is_same<VX, VY>::value &&
                                             std::is_same<VX, Result>::value &&
                                             DotSimd<Result>::value>
        Simd;
    return dotLanes(n, conjX, x, incX, y, incY, result, Simd());
}

template <typename IndexType, typename VX, typename VY, typename Result>
void
dotu(IndexType n, const VX *x, IndexType incX, const VY *y, IndexType incY,
     Result &result)
{
    if (dotLanes(n, false, x, incX, y, incY, result)) {
	return;
    }
    result = Result(0);

    for (IndexType i = 0; i < n; ++i) {
	result += x[i * incX] * y[i * incY];
    }
}

template <typename IndexType, typename VX, typename VY, typename Result>
void
dotc(IndexType n, const VX *x, IndexType incX, const VY *y, IndexType incY,
     Result &result)
{
    if (dotLanes(n, true, x, incX, y, incY, result)) {
	return;
    }
    result = Result(0);

    for (IndexType i = 0; i < n; ++i) {
	result += conjugate(x[i * incX]) * y[i * incY];
    }
}

} // namespace ref
} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL1_KERNEL_REF_DOT_TCC 1
