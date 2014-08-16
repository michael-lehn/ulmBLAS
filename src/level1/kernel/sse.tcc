#ifndef ULMBLAS_SRC_LEVEL1_KERNEL_SSE_TCC
#define ULMBLAS_SRC_LEVEL1_KERNEL_SSE_TCC 1

#include <emmintrin.h>
#include <pmmintrin.h>

#include <src/auxiliary/isaligned.h>
#include <src/level1/kernel/sse.h>
#include <src/level1/ref/axpy.h>
#include <src/level1/ref/dot.h>

namespace ulmBLAS {

template <typename IndexType>
void
axpy(IndexType      n,
     const double   &alpha,
     const double   *x,
     IndexType      incX,
     double         *y,
     IndexType      incY)
{
    if (incX!=1 || incY!=1) {
        axpy_ref(n, alpha, x, incX, y, incY);
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
        axpy_ref(n, alpha, x, IndexType(1), y, IndexType(1));
    }
}

template <typename IndexType, typename Result>
static typename std::enable_if<std::is_convertible<double,Result>::value,
void>::type
dotu(IndexType      n,
     const double   *x,
     IndexType      incX,
     const double   *y,
     IndexType      incY,
     Result         &result)
{
        dotu_ref(n, x, incX, y, incY, result);
        return;
    if (incX!=1 || incY!=1) {
        dotu_ref(n, x, incX, y, incY, result);
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
    }
    if (isAligned(x, 16) && isAligned(y, 16)) {
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
        dotu_ref(n, x, incX, y, incY, result);
        result += _result;
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_SRC_LEVEL1_KERNEL_SSE_TCC
