#ifndef ULMBLAS_LEVEL1EXTENSIONS_KERNEL_SSE_AXPYF_TCC
#define ULMBLAS_LEVEL1EXTENSIONS_KERNEL_SSE_AXPYF_TCC

#include <ulmblas/config/fusefactor.h>
#include <ulmblas/level1extensions/kernel/ref/axpyf.h>
#include <ulmblas/level1extensions/kernel/sse/axpyf.h>
#include <ulmblas/level1extensions/kernel/sse/axpy2v.h>
#include <type_traits>


namespace ulmBLAS { namespace sse {

//
// ----------------
// Double Precision
// ----------------
//

//
// Implementation for fuse factor 2
//
template <typename IndexType>
typename std::enable_if<std::is_integral<IndexType>::value
                     && FuseFactor<double>::axpyf==2,
void>::type
axpyf(IndexType      n,
      const double   &alpha,
      const double   *a,
      IndexType      incA,
      const double   *X,
      IndexType      incRowX,
      IndexType      incColX,
      double         *y,
      IndexType      incY)
{
    axpy2v(n,
           alpha*a[0*incA],
           alpha*a[1*incA],
           &X[0*incColX], incRowX,
           &X[1*incColX], incRowX,
           y, incY);
}

//
// Implementation for fuse factor 4
//
template <typename IndexType>
typename std::enable_if<std::is_integral<IndexType>::value
                     && FuseFactor<double>::axpyf==4,
void>::type
axpyf(IndexType      n,
      const double   &alpha,
      const double   *a,
      IndexType      incA,
      const double   *X,
      IndexType      incRowX,
      IndexType      incColX,
      double         *y,
      IndexType      incY)
{
    if (n<=0) {
        return;
    }

    if (incRowX!=1 || incY!=1) {
        ref::axpyf(n, alpha, a, incA, X, incRowX, incColX, y, incY);
        return;
    }

    const double *x0 = &X[0*incColX];
    const double *x1 = &X[1*incColX];
    const double *x2 = &X[2*incColX];
    const double *x3 = &X[3*incColX];

    bool x0Aligned = isAligned(x0, 16);
    bool x1Aligned = isAligned(x1, 16);
    bool yAligned  = isAligned(y, 16);

    double alpha0 = alpha*a[0*incA];
    double alpha1 = alpha*a[1*incA];
    double alpha2 = alpha*a[2*incA];
    double alpha3 = alpha*a[3*incA];

    if (!x0Aligned && !x1Aligned && !yAligned) {
        y[0] += alpha0*x0[0] + alpha1*x1[0] + alpha2*x2[0] + alpha3*x3[0];
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

        __m128d y12, y34;
        __m128d alpha0_11, alpha1_11, alpha2_11, alpha3_11;
        __m128d x0_12, x0_34;
        __m128d x1_12, x1_34;
        __m128d x2_12, x2_34;
        __m128d x3_12, x3_34;

        alpha0_11 = _mm_loaddup_pd(&alpha0);
        alpha1_11 = _mm_loaddup_pd(&alpha1);
        alpha2_11 = _mm_loaddup_pd(&alpha2);
        alpha3_11 = _mm_loaddup_pd(&alpha3);

        for (IndexType i=0; i<nb; ++i) {
            x0_12 = _mm_load_pd(x0);
            x1_12 = _mm_load_pd(x1);
            x2_12 = _mm_load_pd(x2);
            x3_12 = _mm_load_pd(x3);
            y12   = _mm_load_pd(y);

            x0_12 = x0_12 * alpha0_11;
            y12  += x0_12;
            x1_12 = x1_12 * alpha1_11;
            y12  += x1_12;
            x2_12 = x2_12 * alpha2_11;
            y12  += x2_12;
            x3_12 = x3_12 * alpha3_11;
            y12  += x3_12;
            _mm_store_pd(y, y12);

            x0_34 = _mm_load_pd(x0+2);
            x1_34 = _mm_load_pd(x1+2);
            x2_34 = _mm_load_pd(x2+2);
            x3_34 = _mm_load_pd(x3+2);
            y34   = _mm_load_pd(y+2);

            x0_34 = x0_34 * alpha0_11;
            y34  += x0_34;
            x1_34 = x1_34 * alpha1_11;
            y34  += x1_34;
            x2_34 = x2_34 * alpha2_11;
            y34  += x2_34;
            x3_34 = x3_34 * alpha3_11;
            y34  += x3_34;
            _mm_store_pd(y+2, y34);

            x0 += 4;
            x1 += 4;
            x2 += 4;
            x3 += 4;
            y  += 4;
         }
        for (IndexType i=0; i<nl; ++i) {
            y[i] += alpha0*x0[i] + alpha1*x1[i] + alpha2*x2[i] + alpha3*x3[i];
        }
    } else {
        ref::axpyf(n, alpha, a, incA, X, incRowX, incColX, y, incY);
    }
}

} } // namespace sse, ulmBLAS

#endif // ULMBLAS_LEVEL1EXTENSIONS_KERNEL_SSE_AXPYF_TCC
