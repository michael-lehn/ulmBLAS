#ifndef ULMBLAS_LEVEL2_GER_TCC
#define ULMBLAS_LEVEL2_GER_TCC 1

#include <ulmblas/auxiliary/conjugate.h>
#include <ulmblas/auxiliary/memorypool.h>
#include <ulmblas/config/blocksize.h>
#include <ulmblas/level1/axpy.h>
#include <ulmblas/level1/copy.h>
#include <ulmblas/level1extensions/gecopy.h>
#include <ulmblas/level2/gemvfused.h>
#include <ulmblas/level2/ger.h>
#include <ulmblas/level2/threads.h>

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename TX, typename TY,
          typename TA>
void
gerSerial(IndexType m, IndexType n, const Alpha &alpha, const TX *x,
          IndexType incX, const TY *y, IndexType incY, TA *A, IndexType incRowA,
          IndexType incColA)
{
    typedef decltype(Alpha(0) * TX(0) * TY(0) + TA(0)) T;

    if (m == 0 || n == 0 || alpha == Alpha(0)) {
	return;
    }

    constexpr IndexType UnitStride(1);
    static constexpr bool homogeneousTypes =
        std::is_same<T, Alpha>::value && std::is_same<T, TX>::value &&
        std::is_same<T, TY>::value && std::is_same<T, TA>::value;

    //
    //  float, double: the fused kernel (ulmblas/level2/kernel/gemvf.h)
    //
    static constexpr bool fused =
        homogeneousTypes && level2::FusedGer<T>::value;
    if (level2::GerFused<fused>::run(m, n, alpha, x, incX, y, incY, A, incRowA,
                                     incColA)) {
	return;
    }

    //
    //  If all operands have the same element type and if vectors x and y have
    //  unit stride and matrix A is row or col major the called axpy can use a
    //  fast kernel.
    //
    if (homogeneousTypes && incX == UnitStride && incY == UnitStride) {
	if (incColA == UnitStride) {
	    for (IndexType i = 0; i < m; ++i) {
		axpy(n, alpha * x[i * incX], y, UnitStride, &A[i * incRowA],
		     UnitStride);
	    }
	    return;
	}

	if (incRowA == UnitStride) {
	    for (IndexType j = 0; j < n; ++j) {
		axpy(m, alpha * y[j * incY], x, UnitStride, &A[j * incColA],
		     UnitStride);
	    }
	    return;
	}
    }

    //
    //  Otherwise we pack corresponding operands
    //
    static MemoryPool<T> memoryPool;

    const bool packX = !(incX == UnitStride && std::is_same<T, TX>::value);
    const bool packY = !(incY == UnitStride && std::is_same<T, TY>::value);
    const bool packA = !((incRowA == UnitStride || incColA == UnitStride) &&
                         std::is_same<T, TA>::value);

    // printf("packing: packX=%d, packY=%d, packA=%d\n", packX, packY, packA);
    // printf("packing: incX=%d, incY=%d\n", incX, incY);

    constexpr IndexType MC = BlockSize<T>::MC;
    constexpr IndexType NC = BlockSize<T>::NC;

    const T &alpha_ = alpha;
    T *buffer_x = packX ? memoryPool.allocate(MC) : 0;
    T *buffer_y = packY ? memoryPool.allocate(NC) : 0;
    T *buffer_A = packA ? memoryPool.allocate(MC * NC) : 0;

    const T *x_ = packX ? buffer_x : 0;
    const T *y_ = packY ? buffer_y : 0;

    const IndexType mb = (m + MC - 1) / MC;
    const IndexType nb = (n + NC - 1) / NC;

    const IndexType mc_ = m % MC;
    const IndexType nc_ = n % NC;

    for (IndexType j = 0; j < nb; ++j) {
	IndexType nc = (j != nb - 1 || nc_ == 0) ? NC : nc_;

	if (packY) {
	    copy(nc, false, &y[j * NC * incY], incY, buffer_y, UnitStride);
	} else {
	    y_ = &y[j * NC];
	}

	for (IndexType i = 0; i < mb; ++i) {
	    IndexType mc = (i != mb - 1 || mc_ == 0) ? MC : mc_;

	    if (packX) {
		copy(mc, false, &x[i * MC * incX], incX, buffer_x, UnitStride);
	    } else {
		x_ = &x[i * MC];
	    }

	    if (packA) {
		//  A = A + alpha x y^T: the buffer needs the block of A
		gecopy(mc, nc, false, &A[i * MC * incRowA + j * NC * incColA],
		       incRowA, incColA, buffer_A, UnitStride, mc);
		gerSerial(mc, nc, alpha_, x_, UnitStride, y_, UnitStride,
		          buffer_A, UnitStride, mc);
		gecopy(mc, nc, false, buffer_A, UnitStride, mc,
		       &A[i * MC * incRowA + j * NC * incColA], incRowA,
		       incColA);
	    } else {
		gerSerial(mc, nc, alpha_, x_, UnitStride, y_, UnitStride,
		          &A[i * MC * incRowA + j * NC * incColA], incRowA,
		          incColA);
	    }
	}
    }
    memoryPool.release(buffer_x);
    memoryPool.release(buffer_y);
    memoryPool.release(buffer_A);
}

template <typename IndexType, typename Alpha, typename TX, typename TY,
          typename TA>
void
gercSerial(IndexType m, IndexType n, const Alpha &alpha, bool conj, const TX *x,
           IndexType incX, const TY *y, IndexType incY, TA *A,
           IndexType incRowA, IndexType incColA)
{
    typedef decltype(Alpha(0) * TX(0) * TY(0) + TA(0)) T;

    if (m == 0 || n == 0 || alpha == Alpha(0)) {
	return;
    }

    constexpr IndexType UnitStride(1);
    static constexpr bool homogeneousTypes =
        std::is_same<T, Alpha>::value && std::is_same<T, TX>::value &&
        std::is_same<T, TY>::value && std::is_same<T, TA>::value;

    //
    //  If all operands have the same element type and if vectors x and y have
    //  unit stride and matrix A is row or col major the called axpy can use a
    //  fast kernel.
    //
    if (homogeneousTypes && incX == UnitStride && incY == UnitStride) {
	if (incColA == UnitStride) {
	    if (!conj) {
		for (IndexType i = 0; i < m; ++i) {
		    acxpy(n, alpha * x[i * incX], y, UnitStride,
		          &A[i * incRowA], UnitStride);
		}
		return;
	    } else {
		for (IndexType i = 0; i < m; ++i) {
		    axpy(n, alpha * conjugate(x[i * incX]), y, UnitStride,
		         &A[i * incRowA], UnitStride);
		}
		return;
	    }
	}

	if (incRowA == UnitStride) {
	    if (!conj) {
		for (IndexType j = 0; j < n; ++j) {
		    axpy(m, alpha * conjugate(y[j * incY]), x, UnitStride,
		         &A[j * incColA], UnitStride);
		}
		return;
	    } else {
		for (IndexType j = 0; j < n; ++j) {
		    acxpy(m, alpha * y[j * incY], x, UnitStride,
		          &A[j * incColA], UnitStride);
		}
		return;
	    }
	}
    }

    //
    //  General case
    //
    if (!conj) {
	for (IndexType j = 0; j < n; ++j) {
	    axpy(m, alpha * conjugate(y[j * incY]), x, incX, &A[j * incColA],
	         incRowA);
	}
    } else {
	for (IndexType j = 0; j < n; ++j) {
	    acxpy(m, alpha * y[j * incY], x, incX, &A[j * incColA], incRowA);
	}
    }
}

//
//  The split of ger on several threads (ulmblas/level2/threads.h): the
//  columns of A if A is column major (parts of 4 columns, as OpenBLAS), the
//  rows if A is row major; the other dimension (parts of 64 bytes) if that
//  gives more parts and the first fewer than the threads (tall and skinny
//  A, e.g. in getf2).  nt = 1: one thread.
//
template <typename T, typename IndexType>
std::ptrdiff_t
gerSplit(IndexType m, IndexType n, IndexType incRowA, bool &splitColumns,
         IndexType &u)
{
    const bool colMajor = incRowA == IndexType(1);
    const IndexType line = IndexType(64 / sizeof(T) > 0 ? 64 / sizeof(T) : 1);
    const std::int64_t outer = colMajor ? n : m;
    const std::int64_t inner = colMajor ? m : n;
    const std::int64_t outerUnits = (outer + 3) / 4;
    const std::int64_t innerUnits = (inner + line - 1) / line;
    const bool splitOuter = outerUnits >= std::int64_t(Level2Threads::pool()) ||
                            outerUnits >= innerUnits;
    splitColumns = colMajor == splitOuter;
    u = splitOuter ? IndexType(4) : line;
    return Level2Threads::ger<T>(m, n, splitOuter ? outerUnits : innerUnits);
}

template <typename IndexType, typename Alpha, typename TX, typename TY,
          typename TA>
void
ger(IndexType m, IndexType n, const Alpha &alpha, const TX *x, IndexType incX,
    const TY *y, IndexType incY, TA *A, IndexType incRowA, IndexType incColA)
{
    typedef decltype(Alpha(0) * TX(0) * TY(0) + TA(0)) T;

    if (m <= 0 || n <= 0 || alpha == Alpha(0)) {
	return;
    }
    bool splitColumns;
    IndexType u;
    const std::ptrdiff_t nt = gerSplit<T>(m, n, incRowA, splitColumns, u);
    if (nt <= 1) {
	gerSerial(m, n, alpha, x, incX, y, incY, A, incRowA, incColA);
	return;
    }
    auto body = [&](std::ptrdiff_t tid, std::ptrdiff_t threads) {
	IndexType k0, k1;
	Level2Threads::range(tid, threads, splitColumns ? n : m, u, k0, k1);
	if (k1 <= k0) {
	    return;
	}
	if (splitColumns) {
	    gerSerial(m, k1 - k0, alpha, x, incX, &y[k0 * incY], incY,
	              &A[k0 * incColA], incRowA, incColA);
	} else {
	    gerSerial(k1 - k0, n, alpha, &x[k0 * incX], incX, y, incY,
	              &A[k0 * incRowA], incRowA, incColA);
	}
    };
    ThreadPool::instance().run(nt, body, ULMBLAS_L2_PIN_CALLER != 0);
}

template <typename IndexType, typename Alpha, typename TX, typename TY,
          typename TA>
void
gerc(IndexType m, IndexType n, const Alpha &alpha, bool conj, const TX *x,
     IndexType incX, const TY *y, IndexType incY, TA *A, IndexType incRowA,
     IndexType incColA)
{
    typedef decltype(Alpha(0) * TX(0) * TY(0) + TA(0)) T;

    if (m <= 0 || n <= 0 || alpha == Alpha(0)) {
	return;
    }
    bool splitColumns;
    IndexType u;
    const std::ptrdiff_t nt = gerSplit<T>(m, n, incRowA, splitColumns, u);
    if (nt <= 1) {
	gercSerial(m, n, alpha, conj, x, incX, y, incY, A, incRowA, incColA);
	return;
    }
    auto body = [&](std::ptrdiff_t tid, std::ptrdiff_t threads) {
	IndexType k0, k1;
	Level2Threads::range(tid, threads, splitColumns ? n : m, u, k0, k1);
	if (k1 <= k0) {
	    return;
	}
	if (splitColumns) {
	    gercSerial(m, k1 - k0, alpha, conj, x, incX, &y[k0 * incY], incY,
	               &A[k0 * incColA], incRowA, incColA);
	} else {
	    gercSerial(k1 - k0, n, alpha, conj, &x[k0 * incX], incX, y, incY,
	               &A[k0 * incRowA], incRowA, incColA);
	}
    };
    ThreadPool::instance().run(nt, body, ULMBLAS_L2_PIN_CALLER != 0);
}

template <typename IndexType, typename Alpha, typename TX, typename TY,
          typename TA>
void
gerc(IndexType m, IndexType n, const Alpha &alpha, const TX *x, IndexType incX,
     const TY *y, IndexType incY, TA *A, IndexType incRowA, IndexType incColA)
{
    gerc(m, n, alpha, false, x, incX, y, incY, A, incRowA, incColA);
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_GER_TCC
