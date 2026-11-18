#ifndef ULMBLAS_LAPACK_UNITSTRIDE_TCC
#define ULMBLAS_LAPACK_UNITSTRIDE_TCC 1

#include <cstddef>
#include <ulmblas/config/blocksize.h>
#include <ulmblas/lapack/buffer.h>
#include <ulmblas/lapack/unitstride.h>

namespace ulmBLAS {

template <typename T>
constexpr std::size_t
lapackCopyMax()
{
#if defined(ULMBLAS_LAPACK_COPY_MAX)
    return std::size_t(ULMBLAS_LAPACK_COPY_MAX);
#else
    return std::size_t(BlockSize<T>::KC) * std::size_t(BlockSize<T>::NC);
#endif
}

template <typename T, typename IndexType>
bool
lapackCopyFits(IndexType m, IndexType n)
{
    return n == 0 || std::size_t(m) <= lapackCopyMax<T>() / std::size_t(n);
}

template <typename IndexType, typename T, typename F>
IndexType
withUnitStride(IndexType m, IndexType n, T *A, IndexType incRowA,
               IndexType incColA, bool lowerOnly, F f)
{
    if (incRowA == 1 || incColA == 1 || !lapackCopyFits<T>(m, n)) {
	return f(A, incRowA, incColA);
    }
    Buffer<T> B(std::size_t(m) * std::size_t(n));
    for (IndexType j = 0; j < n; ++j) {
	for (IndexType i = lowerOnly ? j : 0; i < m; ++i) {
	    B[i + j * m] = A[i * incRowA + j * incColA];
	}
    }
    const IndexType info = f(B.data(), IndexType(1), m);
    for (IndexType j = 0; j < n; ++j) {
	for (IndexType i = lowerOnly ? j : 0; i < m; ++i) {
	    A[i * incRowA + j * incColA] = B[i + j * m];
	}
    }
    return info;
}

} // namespace ulmBLAS

#endif // ULMBLAS_LAPACK_UNITSTRIDE_TCC
