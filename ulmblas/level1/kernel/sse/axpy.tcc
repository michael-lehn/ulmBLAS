#ifndef ULMBLAS_LEVEL1_KERNEL_SSE_AXPY_TCC
#define ULMBLAS_LEVEL1_KERNEL_SSE_AXPY_TCC 1

#include <immintrin.h>

#include <ulmblas/auxiliary/isaligned.h>
#include <ulmblas/level1/kernel/ref/axpy.h>
#include <ulmblas/level1/kernel/sse/axpy.h>

namespace ulmBLAS { namespace sse {

//
// ----------------
// Double Precision
// ----------------
//

template <typename IndexType>
void
axpy(IndexType      n,
     const double   &alpha,
     const double   *x,
     IndexType      incX,
     double         *y,
     IndexType      incY)
{
    if (n<=0 || alpha==double(0)) {
        return;
    }

    if (incX!=1 || incY!=1) {
        ref::axpy(n, alpha, x, incX, y, incY);
        return;
    }

    bool xAligned = isAligned(x, 16);
    bool yAligned = isAligned(y, 16);

    if (!xAligned && !yAligned) {
        y[0] += alpha*x[0];
        ++x;
        ++y;
        --n;
        xAligned = yAligned = true;
    }
    if (xAligned && yAligned) {
        IndexType nb = n / 6;
        IndexType nl = n % 6;

        __m128d alpha11, x12, x34, x56, y12, y34, y56;

        alpha11 = _mm_loaddup_pd(&alpha);

        for (IndexType i=0; i<nb; ++i) {
            x12 = _mm_load_pd(x);
            y12 = _mm_load_pd(y);

            x12 = x12 * alpha11;
            y12 = y12 + x12;
            _mm_store_pd(y, y12);

            x34 = _mm_load_pd(x+2);
            y34 = _mm_load_pd(y+2);

            x34 = x34 * alpha11;
            y34 = y34 + x34;
            _mm_store_pd(y+2, y34);

            x56 = _mm_load_pd(x+4);
            y56 = _mm_load_pd(y+4);

            x56 = x56 * alpha11;
            y56 = y56 + x56;
            _mm_store_pd(y+4, y56);

            x += 6;
            y += 6;
        }
        for (IndexType i=0; i<nl; ++i) {
            y[i] += alpha*x[i];
        }
    } else {
        ref::axpy(n, alpha, x, IndexType(1), y, IndexType(1));
    }
}

} } // namespace sse, ulmBLAS

#endif // ULMBLAS_LEVEL1_KERNEL_SSE_AXPY_TCC 1
