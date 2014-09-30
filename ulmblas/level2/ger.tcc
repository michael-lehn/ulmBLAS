#ifndef ULMBLAS_LEVEL2_GER_TCC
#define ULMBLAS_LEVEL2_GER_TCC 1

#include <ulmblas/auxiliary/printmatrix.h>

#include <ulmblas/auxiliary/memorypool.h>
#include <ulmblas/config/blocksize.h>
#include <ulmblas/level1/axpy.h>
#include <ulmblas/level1/copy.h>
#include <ulmblas/level1extensions/gecopy.h>
#include <ulmblas/level2/ger.h>

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename TX, typename TY,
          typename TA>
void
ger(IndexType    m,
    IndexType    n,
    const Alpha  &alpha,
    const TX     *x,
    IndexType    incX,
    const TY     *y,
    IndexType    incY,
    TA           *A,
    IndexType    incRowA,
    IndexType    incColA)
{
    typedef decltype(Alpha(0)*TX(0)*TY(0)+TA(0))  T;

    const IndexType    UnitStride(1);
    static const bool  homogeneousTypes = std::is_same<T,Alpha>::value
                                       && std::is_same<T,TX>::value
                                       && std::is_same<T,TY>::value
                                       && std::is_same<T,TA>::value;

//
//  If all operands have the same element type and if vectors x and y have unit
//  stride and matrix A is row or col major the called axpy can use a fast
//  kernel.
//
    if (homogeneousTypes && incX==UnitStride && incY==UnitStride) {
        if (incColA==UnitStride) {
            for (IndexType i=0; i<m; ++i) {
                axpy(n, alpha*x[i*incX],
                     y, UnitStride,
                     &A[i*incRowA], UnitStride);
            }
            return;
        }

        if (incRowA==UnitStride) {
            for (IndexType j=0; j<n; ++j) {
                axpy(m, alpha*y[j*incY],
                     x, UnitStride,
                     &A[j*incColA], UnitStride);
            }
            return;
        }
    }

//
//  Otherwise we pack corresponding operands
//
    static MemoryPool<T> memoryPool;

    const bool packX    = !(incX==UnitStride && std::is_same<T,TX>::value);
    const bool packY    = !(incY==UnitStride && std::is_same<T,TY>::value);
    const bool packA    = !((incRowA==UnitStride || incColA==UnitStride) &&
                            std::is_same<T,TA>::value);

    //printf("packing: packX=%d, packY=%d, packA=%d\n", packX, packY, packA);
    //printf("packing: incX=%d, incY=%d\n", incX, incY);

    const IndexType MC  = BlockSize<T>::MC;
    const IndexType NC  = BlockSize<T>::NC;

    const T &_alpha     = alpha;
    T *buffer_x         = packX ? memoryPool.allocate(MC)    : 0;
    T *buffer_y         = packY ? memoryPool.allocate(NC)    : 0;
    T *buffer_A         = packA ? memoryPool.allocate(MC*NC) : 0;

    const T *_x         = packX ? buffer_x : 0;
    const T *_y         = packY ? buffer_y : 0;

    const IndexType mb  = (m+MC-1) / MC;
    const IndexType nb  = (n+NC-1) / NC;

    const IndexType _mc = m % MC;
    const IndexType _nc = n % NC;

    for (IndexType j=0; j<nb; ++j) {
        IndexType nc = (j!=nb-1 || _nc==0) ? NC : _nc;

        if (packY) {
            copy(nc, &y[j*NC*incY], incY, buffer_y, UnitStride);
        } else {
            _y = &y[j*NC];
        }

        for (IndexType i=0; i<mb; ++i) {
            IndexType mc = (i!=mb-1 || _mc==0) ? MC : _mc;

            if (packX) {
                copy(mc, &x[i*MC*incX], incX, buffer_x, UnitStride);
            } else {
                _x = &x[i*MC];
            }

            if (packA) {
                ger(mc, nc, _alpha,
                    _x, UnitStride,
                    _y, UnitStride,
                    buffer_A, UnitStride, mc);
                gecopy(mc, nc,
                       buffer_A, UnitStride, mc,
                       &A[i*MC*incRowA+j*NC*incColA], incRowA, incColA);
            } else {
                ger(mc, nc, _alpha,
                    _x, UnitStride,
                    _y, UnitStride,
                    &A[i*MC*incRowA+j*NC*incColA], incRowA, incColA);
            }
        }
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_GER_TCC
