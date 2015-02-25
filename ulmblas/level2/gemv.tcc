#ifndef ULMBLAS_LEVEL2_GEMV_TCC
#define ULMBLAS_LEVEL2_GEMV_TCC 1

#include <ulmblas/auxiliary/memorypool.h>
#include <ulmblas/config/blocksize.h>
#include <ulmblas/config/fusefactor.h>
#include <ulmblas/level1/axpy.h>
#include <ulmblas/level1/copy.h>
#include <ulmblas/level1/dot.h>
#include <ulmblas/level1/scal.h>
#include <ulmblas/level1extensions/axpy2v.h>
#include <ulmblas/level1extensions/axpyf.h>
#include <ulmblas/level1extensions/dotxf.h>
#include <ulmblas/level1extensions/gecopy.h>
#include <ulmblas/level2/gemv.h>

#include <iostream>

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename TA, typename TX,
          typename Beta, typename TY>
void
gemv(IndexType    m,
     IndexType    n,
     const Alpha  &alpha,
     bool         conjA,
     const TA     *A,
     IndexType    incRowA,
     IndexType    incColA,
     const TX     *x,
     IndexType    incX,
     const Beta   &beta,
     TY           *y,
     IndexType    incY)
{
    typedef decltype(Alpha(0)*TA(0)*TX(0)+Beta(0)*TY(0))  T;

    const IndexType    UnitStride(1);
    static const bool  homogeneousTypes = std::is_same<T,Alpha>::value
                                       && std::is_same<T,TA>::value
                                       && std::is_same<T,TX>::value
                                       && std::is_same<T,TY>::value;

    if (m<=0 || n<=0 || (alpha==Alpha(0) && beta==Beta(1))) {
        return;
    }

    scal(m, beta, y, incY);

    if (alpha==Alpha(0)) {
        return;
    }

//
//  If all operands have the same element type and matrix A is col major we use
//  fused axpy/acxpy operations.
//
    if (homogeneousTypes && incRowA==UnitStride) {
        if (!conjA) {
            const IndexType bf = FuseFactor<T>::axpyf;
            const IndexType nb = (n/bf)*bf;

            for (IndexType j=0; j<nb; j+=bf) {
                axpyf(m, alpha, &x[j*incX], incX,
                      &A[j*incColA], UnitStride, incColA,
                      y, incY);
            }
            for (IndexType j=nb; j<n; ++j) {
                axpy(m, alpha*x[j*incX], &A[j*incColA], UnitStride, y, incY);
            }
        } else {
            const IndexType bf = FuseFactor<T>::acxpyf;
            const IndexType nb = (n/bf)*bf;

            for (IndexType j=0; j<nb; j+=bf) {
                acxpyf(m, alpha, &x[j*incX], incX,
                       &A[j*incColA], UnitStride, incColA,
                       y, incY);
            }
            for (IndexType j=nb; j<n; ++j) {
                acxpy(m, alpha*x[j*incX], &A[j*incColA], UnitStride, y, incY);
            }
         }
//
//  If all operands have the same element type and matrix A is row major we use
//  fused dotu/dotc operations.
//
    } else if (homogeneousTypes && incColA==UnitStride) {
        if (!conjA) {
            const IndexType bf = FuseFactor<T>::dotuxf;
            const IndexType mb = (m/bf)*bf;

            TY tmp[bf];

            for (IndexType i=0; i<mb; i+=bf) {
                ref::dotuxf(n, &A[i*incRowA], incRowA, UnitStride,
                            x, incX,
                            tmp, UnitStride);
                for (IndexType l=0; l<bf; ++l) {
                    y[(i+l)*incY] += alpha*tmp[l];
                }
            }
            for (IndexType i=mb; i<m; ++i) {
                dotu(n, &A[i*incRowA], UnitStride, x, incX, tmp[0]);
                y[i*incY] += alpha*tmp[0];
            }
        } else {
            const IndexType bf = FuseFactor<T>::dotuxf;
            const IndexType mb = (m/bf)*bf;

            TY tmp[bf];

            for (IndexType i=0; i<mb; i+=bf) {
                ref::dotcxf(n, &A[i*incRowA], incRowA, UnitStride,
                            x, incX,
                            tmp, UnitStride);
                for (IndexType l=0; l<bf; ++l) {
                    y[(i+l)*incY] += alpha*tmp[l];
                }
            }
            for (IndexType i=mb; i<m; ++i) {
                dotc(n, &A[i*incRowA], UnitStride, x, incX, tmp[0]);
                y[i*incY] += alpha*tmp[0];
            }
        }
    } else {
//
//  Otherwise we pack operands.
//
        static MemoryPool<T> memoryPool;
        const bool packA    = !((incRowA==UnitStride || incColA==UnitStride)
                                && std::is_same<T,TA>::value
                                && !conjA);
        const bool packX    = !(incX==UnitStride && std::is_same<T,TX>::value);
        const bool packY    = !(incY==UnitStride && std::is_same<T,TY>::value);

        const IndexType MC  = BlockSize<T>::MC;
        const IndexType NC  = BlockSize<T>::NC;

        const T &alpha_     = alpha;
        T *buffer_A         = packA ? memoryPool.allocate(MC*NC) : 0;
        T *buffer_x         = packX ? memoryPool.allocate(NC)    : 0;
        T *buffer_y         = packY ? memoryPool.allocate(MC)    : 0;

        const T *A_         = packA ? buffer_A : 0;
        const T *x_         = packX ? buffer_x : 0;

        const IndexType mb  = (m+MC-1) / MC;
        const IndexType nb  = (n+NC-1) / NC;

        const IndexType mc_ = m % MC;
        const IndexType nc_ = n % NC;

        for (IndexType j=0; j<nb; ++j) {
            IndexType nc = (j!=nb-1 || nc_==0) ? NC : nc_;

            if (packX) {
                copy(nc, &x[j*NC*incX], incX, buffer_x, UnitStride);
            } else {
                x_ = &x[j*NC];
            }

            for (IndexType i=0; i<mb; ++i) {
                IndexType mc = (i!=mb-1 || mc_==0) ? MC : mc_;
                IndexType incRow_A, incCol_A;

                if (packA) {
                    incRow_A = UnitStride;
                    incCol_A = mc;
                    gecopy(mc, nc, conjA,
                           &A[i*MC*incRowA+j*NC*incColA], incRowA, incColA,
                           buffer_A, incRow_A, incCol_A);
                } else {
                    incRow_A = incRowA;
                    incCol_A = incColA;
                    A_ = &A[i*MC*incRowA+j*NC*incColA];
                }

                if (packY) {
                    gemv(mc, nc, alpha_,
                         A_, incRow_A, incCol_A,
                         x_, UnitStride,
                         T(0),
                         buffer_y, UnitStride);

                    axpy(mc, T(1), buffer_y, UnitStride, &y[i*MC*incY], incY);
                } else {
                    gemv(mc, nc, alpha_,
                         A_, incRow_A, incCol_A,
                         x_, UnitStride,
                         T(1),
                         &y[i*MC*incY], UnitStride);
                }
            }
        }

        memoryPool.release(buffer_x);
        memoryPool.release(buffer_y);
        memoryPool.release(buffer_A);
    }
}

template <typename IndexType, typename Alpha, typename TA, typename TX,
          typename Beta, typename TY>
void
gemv(IndexType    m,
     IndexType    n,
     const Alpha  &alpha,
     const TA     *A,
     IndexType    incRowA,
     IndexType    incColA,
     const TX     *x,
     IndexType    incX,
     const Beta   &beta,
     TY           *y,
     IndexType    incY)
{
    gemv(m, n, alpha, false, A, incRowA, incColA, x, incX, beta, y, incY);
}


} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_GEMV_TCC
