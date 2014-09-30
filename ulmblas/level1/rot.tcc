#ifndef ULMBLAS_LEVEL1_ROT_TCC
#define ULMBLAS_LEVEL1_ROT_TCC 1

#include <cmath>
#include <complex>
#include <ulmblas/level1/rot.h>

namespace ulmBLAS {

template <typename IndexType, typename VX, typename VY, typename T>
void
rot(IndexType   n,
    VX          *x,
    IndexType   incX,
    VY          *y,
    IndexType   incY,
    T           c,
    T           s)
{
    for (IndexType i=0; i<n; ++i) {
        VX _x =  c*x[i*incX] + s*y[i*incY];
        VY _y = -s*x[i*incX] + c*y[i*incY];
        x[i*incX] = _x;
        y[i*incY] = _y;
    }
}

template <typename IndexType, typename X, typename Y, typename T>
void
rot(IndexType              n,
    std::complex<X>        *x,
    IndexType              incX,
    std::complex<Y>        *y,
    IndexType              incY,
    T                      c,
    const std::complex<T>  &s)
{
    typedef std::complex<T>   CT;

    if (incX != IndexType(1) || incY != IndexType(1)) {
        for (IndexType i=0; i<n; ++i) {
            const CT tmp = c*x[i*incX] + s*y[i*incY];
            y[i*incY]  = c*y[i*incY] - std::conj(s)*x[i*incX];
            x[i*incX]  = tmp;
        }
    } else {
        for (IndexType i=0; i<n; ++i) {
            const CT tmp = c*x[i] + s*y[i];
            y[i] = c*y[i] - std::conj(s)*x[i];
            x[i] = tmp;
        }
    }
}

template <typename A, typename B, typename T>
void
rotg(A &a,
     B &b,
     T &c,
     T &s)
{
    A absA = std::abs(a);
    B absB = std::abs(b);

    T scale = absA + absB;
    T roe = (absA > absB) ? a : b;
    if (scale==0) {
        c = 1;
        s = 0;
        a = 0;
        b = 0;
        return;
    }
    A aScaled = absA / scale;
    B bScaled = absB / scale;
    T r = scale*std::sqrt(aScaled*aScaled + bScaled*bScaled);
    if (roe<0) {
        r = -r;
    }
    c = a / r;
    s = b / r;

    B z = 1;
    if (absA > absB) {
        z = s;
    }
    if ((absA < absB) && (c != 0)) {
        z = T(1)/c;
    }
    a = r;
    b = z;
}

template <typename TA, typename TB, typename T>
void
rotg(std::complex<TA>   &a,
     std::complex<TB>   &b,
     T                  &c,
     std::complex<T>    &s)
{
    std::complex<T>  alpha;
    T                norm, scale;

    if (std::abs(a)==TA(0)) {
        c = 0;
        s = std::complex<T>(1,0);
        a = b;
    } else {
        scale = std::abs(a) + std::abs(b);
        norm  = scale*std::sqrt(std::pow(std::abs(a/scale),2)
                               +std::pow(std::abs(b/scale),2));
        alpha = a / std::abs(a);
        c     = std::abs(a) / norm;
        s     = alpha*std::conj(b)/norm;
        a     = alpha*norm;
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL1_ROT_TCC 1
