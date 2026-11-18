#ifndef ULMBLAS_LEVEL2_GEMVFUSED_H
#define ULMBLAS_LEVEL2_GEMVFUSED_H 1

#include <complex>
#include <type_traits>
#include <ulmblas/auxiliary/memorypool.h>
#include <ulmblas/config/level2.h>
#include <ulmblas/level2/kernel/gemvf.h>

//
//  gemv with the fused kernels of ulmblas/level2/kernel/gemvf.h for
//  operands of one type (float, double, complex<float>, complex<double>):
//
//      columns:  y += alpha A x, A column major (BLAS N): blocks of BlockN
//                rows of y (L1 cache, as OpenBLAS's NBMAX), FuseN columns
//                per pass over a block, then 4, 2, 1
//      rows:     y += alpha A x, A row major (BLAS T): blocks of BlockT
//                elements of x, FuseT rows per pass over a block, then 4,
//                2, 1
//
//  Strided x (columns) and y (rows) are fine; strided y (columns) and x
//  (rows) go through a buffer.  Complex x (rows) always: the buffer holds
//  x and x with the real and imaginary parts swapped.
//
//  ger, float and double: A += alpha x y^T with the kernel rank1, FuseGer
//  columns (A column major, x contiguous) or rows (A row major, y
//  contiguous) at a time.
//

namespace ulmBLAS {
namespace level2 {

//  the types of the fused ger kernel (real numbers)
template <typename T> struct FusedGer
{
    static constexpr bool value =
        std::is_same<T, float>::value || std::is_same<T, double>::value;
};

template <typename T> struct FusedGemv
{
    static constexpr bool value = std::is_same<T, float>::value ||
                                  std::is_same<T, double>::value ||
                                  std::is_same<T, std::complex<float>>::value ||
                                  std::is_same<T, std::complex<double>>::value;
};

template <typename T> struct RealType
{
    typedef T type;
};

template <typename T> struct RealType<std::complex<T>>
{
    typedef T type;
};

//
//  Fewer than F columns (rows) left: kernels for 4, 2, 1 (1 for 1: the
//  loop with one column leaves nothing)
//
template <int F> struct Smaller
{
    static constexpr int value = F > 4 ? 4 : (F > 1 ? F / 2 : 1);
};

//
//  One block of rows of y (contiguous): the columns j = 0, ..., n-1 in
//  groups of F, then of 4, 2, 1
//
template <int F, bool Conj, typename IndexType, typename T>
void
columnsBlock(IndexType m, IndexType n, IndexType j, const T &alpha, const T *A,
             IndexType ldA, const T *x, IndexType incX, T *y)
{
    for (; j + F <= n; j += F) {
	T a[F];
	for (int f = 0; f < F; ++f) {
	    a[f] = alpha * x[(j + f) * incX];
	}
	columns<F>(m, a, &A[j * ldA], ldA, y);
    }
    if (j < n) {
	columnsBlock<Smaller<F>::value, Conj>(m, n, j, alpha, A, ldA, x, incX,
	                                      y);
    }
}

template <int F, bool Conj, typename IndexType, typename T>
void
columnsBlock(IndexType m, IndexType n, IndexType j,
             const std::complex<T> &alpha, const std::complex<T> *A,
             IndexType ldA, const std::complex<T> *x, IndexType incX,
             std::complex<T> *y)
{
    for (; j + F <= n; j += F) {
	std::complex<T> a[F];
	for (int f = 0; f < F; ++f) {
	    a[f] = alpha * x[(j + f) * incX];
	}
	columnsComplex<F, Conj>(m, a, reinterpret_cast<const T *>(&A[j * ldA]),
	                        ldA, reinterpret_cast<T *>(y));
    }
    if (j < n) {
	columnsBlock<Smaller<F>::value, Conj>(m, n, j, alpha, A, ldA, x, incX,
	                                      y);
    }
}

//
//  y += alpha A x (Conj: alpha conj(A) x), A column major
//
template <bool Conj, typename IndexType, typename T>
void
gemvColumns(IndexType m, IndexType n, const T &alpha, const T *A, IndexType ldA,
            const T *x, IndexType incX, T *y, IndexType incY)
{
    constexpr int F = Level2Param<T>::FuseN;
    const IndexType MB = IndexType(Level2Param<T>::BlockN);

    static MemoryPool<T> memoryPool;
    T *buffer = incY != IndexType(1) ? memoryPool.allocate(MB) : nullptr;

    for (IndexType i = 0; i < m; i += MB) {
	const IndexType mb = m - i < MB ? m - i : MB;
	T *y_ = &y[i];
	if (buffer) {
	    for (IndexType k = 0; k < mb; ++k) {
		buffer[k] = T(0);
	    }
	    y_ = buffer;
	}
	columnsBlock<F, Conj>(mb, n, IndexType(0), alpha, &A[i], ldA, x, incX,
	                      y_);
	if (buffer) {
	    for (IndexType k = 0; k < mb; ++k) {
		y[(i + k) * incY] += buffer[k];
	    }
	}
    }
    if (buffer) {
	memoryPool.release(buffer);
    }
}

//
//  One block of x (contiguous, xs: complex only): the rows i = 0, ..., m-1
//  in groups of F, then of 4, 2, 1
//
template <int F, bool Conj, typename IndexType, typename T>
void
rowsBlock(IndexType m, IndexType n, IndexType i, const T &alpha, const T *A,
          IndexType ldA, const T *x, const T *xs, T *y, IndexType incY)
{
    constexpr int L = Level2Param<T>::LanesT;
    for (; i + F <= m; i += F) {
	T r[F];
	rows<F, L>(n, &A[i * ldA], ldA, x, r);
	for (int f = 0; f < F; ++f) {
	    y[(i + f) * incY] += alpha * r[f];
	}
    }
    if (i < m) {
	rowsBlock<Smaller<F>::value, Conj>(m, n, i, alpha, A, ldA, x, xs, y,
	                                   incY);
    }
}

template <int F, bool Conj, typename IndexType, typename T>
void
rowsBlock(IndexType m, IndexType n, IndexType i, const std::complex<T> &alpha,
          const std::complex<T> *A, IndexType ldA, const T *x, const T *xs,
          std::complex<T> *y, IndexType incY)
{
    constexpr int L = Level2Param<std::complex<T>>::LanesT;
    for (; i + F <= m; i += F) {
	std::complex<T> r[F];
	rowsComplex<F, L, Conj>(n, reinterpret_cast<const T *>(&A[i * ldA]),
	                        ldA, x, xs, r);
	for (int f = 0; f < F; ++f) {
	    y[(i + f) * incY] += alpha * r[f];
	}
    }
    if (i < m) {
	rowsBlock<Smaller<F>::value, Conj>(m, n, i, alpha, A, ldA, x, xs, y,
	                                   incY);
    }
}

//  the block of x for rowsBlock: real, contiguous: x itself
template <typename IndexType, typename T>
const T *
rowsX(IndexType n, const T *x, IndexType incX, T *buffer, T *&xs)
{
    xs = nullptr;
    if (incX == IndexType(1)) {
	return x;
    }
    for (IndexType j = 0; j < n; ++j) {
	buffer[j] = x[j * incX];
    }
    return buffer;
}

//  complex: x and the swapped x as real numbers
template <typename IndexType, typename T>
const T *
rowsX(IndexType n, const std::complex<T> *x, IndexType incX,
      std::complex<T> *buffer, T *&xs)
{
    T *x_ = reinterpret_cast<T *>(buffer);
    xs = x_ + 2 * n;
    for (IndexType j = 0; j < n; ++j) {
	const T re = x[j * incX].real(), im = x[j * incX].imag();
	x_[2 * j] = re;
	x_[2 * j + 1] = im;
	xs[2 * j] = im;
	xs[2 * j + 1] = re;
    }
    return x_;
}

template <typename T> struct IsComplex
{
    static constexpr bool value = false;
};

template <typename T> struct IsComplex<std::complex<T>>
{
    static constexpr bool value = true;
};

//
//  y += alpha A x (Conj: alpha conj(A) x), A row major
//
template <bool Conj, typename IndexType, typename T>
void
gemvRows(IndexType m, IndexType n, const T &alpha, const T *A, IndexType ldA,
         const T *x, IndexType incX, T *y, IndexType incY)
{
    typedef typename RealType<T>::type R;
    constexpr int F = Level2Param<T>::FuseT;
    const IndexType NB = IndexType(Level2Param<T>::BlockT);
    const IndexType nb0 = n < NB ? n : NB;

    static MemoryPool<T> memoryPool;
    const bool pack = IsComplex<T>::value || incX != IndexType(1);
    T *buffer = pack ? memoryPool.allocate(2 * nb0) : nullptr;

    for (IndexType j = 0; j < n; j += NB) {
	const IndexType nb = n - j < NB ? n - j : NB;
	R *xs;
	const R *x_ = reinterpret_cast<const R *>(
	    rowsX(nb, &x[j * incX], incX, buffer, xs));
	rowsBlock<F, Conj>(m, nb, IndexType(0), alpha, &A[j], ldA, x_, xs, y,
	                   incY);
    }
    if (buffer) {
	memoryPool.release(buffer);
    }
}

//
//  gemvFused<true>::run: y += alpha A x (conjA: alpha conj(A) x) with the
//  fused kernels if A is column or row major; false otherwise (nothing
//  done).  gemvFused<false>: types the kernels do not handle.
//
template <bool Fused> struct GemvFused
{
    template <typename IndexType, typename Alpha, typename TA, typename TX,
              typename TY>
    static bool
    run(IndexType m, IndexType n, const Alpha &alpha, bool conjA, const TA *A,
        IndexType incRowA, IndexType incColA, const TX *x, IndexType incX,
        TY *y, IndexType incY)
    {
	return false;
    }
};

template <> struct GemvFused<true>
{
    template <typename IndexType, typename T>
    static bool
    run(IndexType m, IndexType n, const T &alpha, bool conjA, const T *A,
        IndexType incRowA, IndexType incColA, const T *x, IndexType incX, T *y,
        IndexType incY)
    {
	if (incRowA == IndexType(1)) {
	    if (conjA) {
		gemvColumns<true>(m, n, alpha, A, incColA, x, incX, y, incY);
	    } else {
		gemvColumns<false>(m, n, alpha, A, incColA, x, incX, y, incY);
	    }
	    return true;
	}
	if (incColA == IndexType(1)) {
	    if (conjA) {
		gemvRows<true>(m, n, alpha, A, incRowA, x, incX, y, incY);
	    } else {
		gemvRows<false>(m, n, alpha, A, incRowA, x, incX, y, incY);
	    }
	    return true;
	}
	return false;
    }
};

//
//  GerFused<true>::run: A += alpha x y^T with rank1 (float, double), F =
//  FuseGer columns (A column major, x contiguous) or rows (A row major, y
//  contiguous) at a time, then 4, 2, 1; false otherwise (nothing done)
//
template <int F, typename IndexType, typename T>
void
rank1Block(IndexType m, IndexType n, IndexType j, const T &alpha, const T *x,
           const T *y, IndexType incY, T *A, IndexType ldA)
{
    for (; j + F <= n; j += F) {
	T a[F];
	for (int f = 0; f < F; ++f) {
	    a[f] = alpha * y[(j + f) * incY];
	}
	rank1<F>(m, a, x, &A[j * ldA], ldA);
    }
    if (j < n) {
	rank1Block<Smaller<F>::value>(m, n, j, alpha, x, y, incY, A, ldA);
    }
}

template <bool Fused> struct GerFused
{
    template <typename IndexType, typename Alpha, typename TX, typename TY,
              typename TA>
    static bool
    run(IndexType m, IndexType n, const Alpha &alpha, const TX *x,
        IndexType incX, const TY *y, IndexType incY, TA *A, IndexType incRowA,
        IndexType incColA)
    {
	return false;
    }
};

template <> struct GerFused<true>
{
    template <typename IndexType, typename T>
    static bool
    run(IndexType m, IndexType n, const T &alpha, const T *x, IndexType incX,
        const T *y, IndexType incY, T *A, IndexType incRowA, IndexType incColA)
    {
	constexpr int F = Level2Param<T>::FuseGer;
	if (incRowA == IndexType(1) && incX == IndexType(1)) {
	    rank1Block<F>(m, n, IndexType(0), alpha, x, y, incY, A, incColA);
	    return true;
	}
	if (incColA == IndexType(1) && incY == IndexType(1)) {
	    rank1Block<F>(n, m, IndexType(0), alpha, y, x, incX, A, incRowA);
	    return true;
	}
	return false;
    }
};

} // namespace level2
} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_GEMVFUSED_H
