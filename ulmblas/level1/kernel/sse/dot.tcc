#ifndef ULMBLAS_LEVEL1_KERNEL_SSE_DOT_TCC
#define ULMBLAS_LEVEL1_KERNEL_SSE_DOT_TCC 1

#include <immintrin.h>

#include <ulmblas/auxiliary/isaligned.h>
#include <ulmblas/level1/kernel/ref/dot.h>
#include <ulmblas/level1/kernel/sse/dot.h>

namespace ulmBLAS { namespace sse {

//
// ----------------
// Double Precision
// ----------------
//

template <typename IndexType>
void
dotu(IndexType      n,
     const double   *x,
     IndexType      incX,
     const double   *y,
     IndexType      incY,
     double         &result)
{
    if (n<=0) {
        result = 0;
    }

    if (incX!=1 || incY!=1) {
        ref::dotu(n, x, incX, y, incY, result);
        return;
    }

    bool xAligned = isAligned(x, 16);
    bool yAligned = isAligned(y, 16);

    double _result = 0;

    if (!xAligned && !yAligned) {
        _result = y[0]*x[0];
        ++x;
        ++y;
        --n;
        xAligned = yAligned = true;
    }
    if (xAligned && yAligned) {
        IndexType nb = n / 2;
        IndexType nl = n % 2;

        __m128d x12, y12, result12;
        double  _result12[2];

        result12 = _mm_setzero_pd();
        for (IndexType i=0; i<nb; ++i) {
            x12 = _mm_load_pd(x);
            y12 = _mm_load_pd(y);

            x12      = x12*y12;
            result12 = result12 + x12;

            x += 2;
            y += 2;
        }
        _mm_store_pd(_result12, result12);

        result = _result + _result12[0] + _result12[1];

        for (IndexType i=0; i<nl; ++i) {
            result += x[i]*y[i];
        }

    } else {
        result = 0;
        ref::dotu(n, x, incX, y, incY, result);
        result += _result;
    }
}

} } // namespace ref, ulmBLAS

#endif // ULMBLAS_LEVEL1_KERNEL_SSE_DOT_TCC 1
