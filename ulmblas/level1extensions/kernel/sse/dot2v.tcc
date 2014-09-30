#ifndef ULMBLAS_LEVEL1EXTENSIONS_KERNEL_SSE_DOT2V_TCC
#define ULMBLAS_LEVEL1EXTENSIONS_KERNEL_SSE_DOT2V_TCC 1

#include <ulmblas/level1extensions/kernel/ref/dot2v.h>

namespace ulmBLAS { namespace sse {

//
// ----------------
// Double Precision
// ----------------
//
template <typename IndexType>
void
dotu2v(IndexType      n,
       const double   *x0,
       IndexType      incX0,
       const double   *x1,
       IndexType      incX1,
       double         *y,
       IndexType      incY,
       double         *result,
       IndexType      resultInc)
{
    if (n<=0) {
        return;
    }

    if (incX0!=1 || incX1!=1 || incY!=1) {
        ref::dotu2v(n, x0, incX0, x1, incX1, y, incY, result, resultInc);
        return;
    }

    bool x0Aligned = isAligned(x0, 16);
    bool x1Aligned = isAligned(x1, 16);
    bool yAligned  = isAligned(y, 16);

    double &result0 = result[0*resultInc];
    double &result1 = result[1*resultInc];

    result0 = result1 = double(0);

    if (!x0Aligned && !x1Aligned && !yAligned) {
        result0 += x0[0]*y[0];
        result1 += x1[0]*y[0];
        ++x0;
        ++x1;
        ++y;
        --n;
        x0Aligned = x1Aligned = yAligned = true;
    }
    if (x0Aligned && x1Aligned && yAligned) {
        IndexType nb = n / 8;
        IndexType nl = n % 8;

        __m128d rho0, rho1;
        __m128d y12, y34, y56, y78;
        __m128d x0_12, x0_34, x0_56, x0_78;
        __m128d x1_12, x1_34, x1_56, x1_78;

        rho0 = _mm_setzero_pd();
        rho1 = _mm_setzero_pd();

        for (IndexType i=0; i<nb; ++i) {
            y12   = _mm_load_pd(y);
            y34   = _mm_load_pd(y+2);
            y56   = _mm_load_pd(y+4);
            y78   = _mm_load_pd(y+6);

            x0_12 = _mm_load_pd(x0);
            x0_34 = _mm_load_pd(x0+2);
            x0_56 = _mm_load_pd(x0+4);
            x0_78 = _mm_load_pd(x0+6);

            x1_12 = _mm_load_pd(x1);
            x1_34 = _mm_load_pd(x1+2);
            x1_56 = _mm_load_pd(x1+4);
            x1_78 = _mm_load_pd(x1+6);

            x0_12 = x0_12 * y12;
            rho0 += x0_12;
            x1_12 = x1_12 * y12;
            rho1 += x1_12;

            x0_34 = x0_34 * y34;
            rho0 += x0_34;
            x1_34 = x1_34 * y34;
            rho1 += x1_34;

            x0_56 = x0_56 * y56;
            rho0 += x0_56;
            x1_56 = x1_56 * y56;
            rho1 += x1_56;

            x0_78 = x0_78 * y78;
            rho0 += x0_78;
            x1_78 = x1_78 * y78;
            rho1 += x1_78;

            x0 += 8;
            x1 += 8;
            y  += 8;
        }

        double rho0_[2];
        double rho1_[2];

        _mm_store_pd(rho0_, rho0);
        _mm_store_pd(rho1_, rho1);

        for (IndexType i=0; i<nl; ++i) {
            result0 += x0[i]*y[i];
            result1 += x1[i]*y[i];
        }

        result0 += rho0_[0] + rho0_[1];
        result1 += rho1_[0] + rho1_[1];
    } else {
        ref::dotu2v(n, x0, incX0, x1, incX1, y, incY, result, resultInc);
    }
}

} } // namespace sse, ulmBLAS

#endif // ULMBLAS_LEVEL1EXTENSIONS_KERNEL_SSE_DOT2V_TCC 1
