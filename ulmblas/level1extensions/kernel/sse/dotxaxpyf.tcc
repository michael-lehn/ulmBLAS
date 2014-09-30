#ifndef ULMBLAS_LEVEL1EXTENSIONS_KERNEL_SSE_DOTXAXPYF_TCC
#define ULMBLAS_LEVEL1EXTENSIONS_KERNEL_SSE_DOTXAXPYF_TCC 1

#include <ulmblas/level1extensions/kernel/sse/dotxaxpyf.h>
#include <ulmblas/level1extensions/kernel/ref/dotxaxpyf.h>

#ifdef DDOTXAXPYF_FUSEFACTOR
#undef DDOTXAXPYF_FUSEFACTOR
#endif

#define DDOTXAXPYF_FUSEFACTOR  2

namespace ulmBLAS { namespace sse {

template <typename T>
int
dotxaxpyf_fusefactor()
{
    if (std::is_same<T,double>::value) {
        return DDOTXAXPYF_FUSEFACTOR;
    }
    return ref::dotxaxpyf_fusefactor<T>();
}

//
// ----------------
// Double Precision
// ----------------
//

template <typename IndexType>
void
dotxaxpyf(IndexType      n,
          bool           conjX,
          bool           conjXt,
          bool           conjY,
          const double   &alpha,
          const double   *a,
          IndexType      incA,
          const double   *X,
          IndexType      incRowX,
          IndexType      incColX,
          const double   *y,
          IndexType      incY,
          double         *z,
          IndexType      incZ,
          double         *rho,
          IndexType      incRho)
{
    const IndexType bf = dotxaxpyf_fusefactor<double>();

    for (IndexType l=0; l<bf; ++l) {
        rho[l*incRho] = 0;
    }

    if (n<=0) {
        return;
    }

    if (incRowX!=1 || incY!=1 || incZ!=1 || conjX || conjXt || conjY) {
        ref::dotxaxpyf(n, conjX, conjXt, conjY, alpha, a, incA,
                       X, incRowX, incColX, y, incY,
                       z, incZ, rho, incRho);
        return;
    }

    const double alpha0 = alpha*a[0*incA];
    const double alpha1 = alpha*a[1*incA];

    const double *x0    = &X[0*incColX];
    const double *x1    = &X[1*incColX];

    double &rho0        = rho[0*incRho];
    double &rho1        = rho[1*incRho];

    bool x0Aligned      = isAligned(x0, 16);
    bool x1Aligned      = isAligned(x1, 16);
    bool yAligned       = isAligned(y, 16);
    bool zAligned       = isAligned(z, 16);

    rho0 = rho1 = 0;

    if (!x0Aligned && !x1Aligned && !yAligned && !zAligned) {
        z[0] += alpha0*x0[0] + alpha1*x1[0];
        rho0 += x0[0]*y[0];
        rho1 += x1[0]*y[0];

        ++x0;
        ++x1;
        ++y;
        ++z;
        --n;
        x0Aligned = x1Aligned = yAligned = zAligned = true;
    }
    if (x0Aligned && x1Aligned && yAligned && zAligned) {
        IndexType nb = n / 4;
        IndexType nl = n % 4;

        __m128d rho00, rho11;
        __m128d alpha0_11, alpha1_11;
        __m128d z12, z34;
        __m128d y12, y34;
        __m128d tmp0, tmp1;
        __m128d x0_12, x0_34;
        __m128d x1_12, x1_34;

        alpha0_11 = _mm_loaddup_pd(&alpha0);
        alpha1_11 = _mm_loaddup_pd(&alpha1);

        rho00     = _mm_setzero_pd();
        rho11     = _mm_setzero_pd();

        double _rho00[2], _rho11[2];

        for (IndexType i=0; i<nb; ++i) {
            x0_12 = _mm_load_pd(x0);
            x0_34 = _mm_load_pd(x0+2);

            x1_12 = _mm_load_pd(x1);
            x1_34 = _mm_load_pd(x1+2);

            y12   = _mm_load_pd(y);
            y34   = _mm_load_pd(y+2);

            z12   = _mm_load_pd(z);
            z34   = _mm_load_pd(z+2);

            tmp0  = y12;
            tmp1  = y12;

            tmp0  = tmp0 * x0_12;
            tmp1  = tmp1 * x1_12;
            x0_12 = alpha0_11 * x0_12;
            x1_12 = alpha1_11 * x1_12;
            z12   = z12 + x0_12;
            z12   = z12 + x1_12;
            rho00 = rho00 + tmp0;
            rho11 = rho11 + tmp1;
            _mm_store_pd(z, z12);

            tmp0  = y34;
            tmp1  = y34;

            tmp0  = tmp0 * x0_34;
            tmp1  = tmp1 * x1_34;
            x0_34 = alpha0_11 * x0_34;
            x1_34 = alpha1_11 * x1_34;
            z34   = z34 + x0_34;
            z34   = z34 + x1_34;
            rho00 = rho00 + tmp0;
            rho11 = rho11 + tmp1;
            _mm_store_pd(z+2, z34);

            x0 += 4;
            x1 += 4;
            y  += 4;
            z  += 4;
        }
        _mm_store_pd(_rho00, rho00);
        _mm_store_pd(_rho11, rho11);

        rho0 += _rho00[0] + _rho00[1];
        rho1 += _rho11[0] + _rho11[1];

        for (IndexType i=0; i<nl; ++i) {
            z[i] += alpha0*x0[i] + alpha1*x1[i];
            rho0 += x0[i]*y[i];
            rho1 += x1[i]*y[i];
        }

    } else {
        ref::dotxaxpyf(n, conjX, conjXt, conjY, alpha, a, incA,
                       X, incRowX, incColX, y, incY,
                       z, incZ, rho, incRho);
    }
}

} } // namespace sse, ulmBLAS

#endif // ULMBLAS_LEVEL1EXTENSIONS_KERNEL_SSE_DOTXAXPYF_TCC 1
