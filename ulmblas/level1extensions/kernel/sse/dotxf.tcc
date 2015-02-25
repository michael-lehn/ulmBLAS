#ifndef ULMBLAS_LEVEL1EXTENSIONS_KERNEL_SSE_DOTXF_TCC
#define ULMBLAS_LEVEL1EXTENSIONS_KERNEL_SSE_DOTXF_TCC 1

#include <immintrin.h>
#include <ulmblas/level1extensions/kernel/ref/dotxf.h>
#include <ulmblas/level1extensions/kernel/sse/dotxf.h>
#include <ulmblas/level1extensions/kernel/sse/dot2v.h>

namespace ulmBLAS { namespace sse {

//
// ----------------
// Double Precision
// ----------------
//

template <typename IndexType>
typename std::enable_if<std::is_integral<IndexType>::value
                     && FuseFactor<double>::dotuxf==2,
void>::type
dotuxf(IndexType      n,
       const double   *X,
       IndexType      incRowX,
       IndexType      incColX,
       const double   *y,
       IndexType      incY,
       double         *result,
       IndexType      resultInc)
{
    dotu2v(n,
           &X[0*incRowX], incColX,
           &X[1*incRowX], incColX,
           y, incY,
           result, resultInc);
}

template <typename IndexType>
typename std::enable_if<std::is_integral<IndexType>::value
                     && FuseFactor<double>::dotuxf==4,
void>::type
dotuxf(IndexType      n,
       const double   *X,
       IndexType      incRowX,
       IndexType      incColX,
       const double   *y,
       IndexType      incY,
       double         *result,
       IndexType      resultInc)
{
    if (n<=0) {
        return;
    }

    if (incColX!=1 || incY!=1) {
        ref::dotuxf(n, X, incRowX, incColX, y, incY, result, resultInc);
        return;
    }

    double &result1  = result[0*resultInc];
    double &result2  = result[1*resultInc];
    double &result3  = result[2*resultInc];
    double &result4  = result[3*resultInc];

    const double *x0 = &X[0*incRowX];
    const double *x1 = &X[1*incRowX];
    const double *x2 = &X[2*incRowX];
    const double *x3 = &X[3*incRowX];

    bool x0Aligned   = isAligned(x0, 16);
    bool x1Aligned   = isAligned(x1, 16);
    bool yAligned    = isAligned(y, 16);

    result1 = result2 = result3 = result4 = double(0);

    if (!x0Aligned && !x1Aligned && !yAligned) {
        result1 += x0[0]*y[0];
        result2 += x1[0]*y[0];
        result3 += x2[0]*y[0];
        result4 += x3[0]*y[0];

        ++x0;
        ++x1;
        ++x2;
        ++x3;
        ++y;
        --n;
        x0Aligned = x1Aligned = yAligned = true;
    }
    if (x0Aligned && x1Aligned && yAligned) {
        IndexType nb = n / 4;
        IndexType nl = n % 4;

        __m128d rho0, rho1, rho2, rho3;
        __m128d y12, y34;
        __m128d x0_12, x0_34;
        __m128d x1_12, x1_34;
        __m128d x2_12, x2_34;
        __m128d x3_12, x3_34;

        rho0 = _mm_setzero_pd();
        rho1 = _mm_setzero_pd();
        rho2 = _mm_setzero_pd();
        rho3 = _mm_setzero_pd();

        for (IndexType i=0; i<nb; ++i) {
            y12 = _mm_load_pd(y);
            y34 = _mm_load_pd(y+2);

            x0_12 = _mm_load_pd(x0);
            x0_34 = _mm_load_pd(x0+2);

            x1_12 = _mm_load_pd(x1);
            x1_34 = _mm_load_pd(x1+2);

            x2_12 = _mm_load_pd(x2);
            x2_34 = _mm_load_pd(x2+2);

            x3_12 = _mm_load_pd(x3);
            x3_34 = _mm_load_pd(x3+2);

            x0_12 *= y12;
            rho0  += x0_12;
            x0_34 *= y34;
            rho0  += x0_34;

            x1_12 *= y12;
            rho1  += x1_12;
            x1_34 *= y34;
            rho1  += x1_34;

            x2_12 *= y12;
            rho2  += x2_12;
            x2_34 *= y34;
            rho2  += x2_34;

            x3_12 *= y12;
            rho3  += x3_12;
            x3_34 *= y34;
            rho3  += x3_34;

            x0 += 4;
            x1 += 4;
            x2 += 4;
            x3 += 4;
            y  += 4;
        }

        double rho0_[2];
        double rho1_[2];
        double rho2_[2];
        double rho3_[2];

        _mm_store_pd(rho0_, rho0);
        _mm_store_pd(rho1_, rho1);
        _mm_store_pd(rho2_, rho2);
        _mm_store_pd(rho3_, rho3);

        for (IndexType i=0; i<nl; ++i) {
            result1 += x0[i]*y[i];
            result2 += x1[i]*y[i];
            result3 += x2[i]*y[i];
            result4 += x3[i]*y[i];
        }

        result1 += rho0_[0] + rho0_[1];
        result2 += rho1_[0] + rho1_[1];
        result3 += rho2_[0] + rho2_[1];
        result4 += rho3_[0] + rho3_[1];

    } else {
        ref::dotuxf(n, X, incRowX, incColX, y, incY, result, resultInc);
    }
}

} } // namespace sse, ulmBLAS

#endif // ULMBLAS_LEVEL1EXTENSIONS_KERNEL_SSE_DOTXF_TCC
