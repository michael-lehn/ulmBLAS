#ifndef ULMBLAS_LEVEL1_NRM2_TCC
#define ULMBLAS_LEVEL1_NRM2_TCC 1

#include <complex>
#include <cmath>
#include <ulmblas/level1/nrm2.h>

namespace ulmBLAS {

template <typename IndexType, typename VX>
VX
nrm2(IndexType  n,
     const VX   *x,
     IndexType  incX)
{
    VX result;

    nrm2(n, x, incX, result);
    return result;
}

template <typename IndexType, typename VX>
VX
nrm2(IndexType                n,
     const std::complex<VX>   *x,
     IndexType                incX)
{
    VX result;

    nrm2(n, x, incX, result);
    return result;
}

template <typename IndexType, typename VX, typename Result>
void
nrm2(IndexType  n,
     const VX   *x,
     IndexType  incX,
     Result     &result)
{
    const Result  Zero(0), One(1);

    if (n<1) {
        result = Zero;
    } else if (n==1) {
        result = std::abs(*x);
    } else {
        Result scale = 0;
        Result ssq   = 1;

        for (IndexType i=0; i<n; ++i) {
            if (x[i*incX]!=Zero) {
                Result absXi = std::abs(x[i*incX]);
                if (scale<absXi) {
                    ssq = One + ssq * std::pow(scale/absXi, 2);
                    scale = absXi;
                } else {
                    ssq += std::pow(absXi/scale, 2);
                }
            }
        }
        result = scale*sqrt(ssq);
    }
}

template <typename IndexType, typename VX, typename Result>
void
nrm2(IndexType                n,
     const std::complex<VX>   *x,
     IndexType                incX,
     Result                   &result)
{
    const Result  Zero(0), One(1);

    if (n<1) {
        result = Zero;
    } else if (n==1) {
        result = std::abs(*x);
    } else {
        Result scale = 0;
        Result ssq = 1;

        for (IndexType i=0; i<n; ++i) {
            if (std::real(x[i*incX]) != Zero) {
                Result absXi = std::abs(std::real(x[i*incX]));
                if (scale<absXi) {
                    ssq = One + ssq * std::pow(scale/absXi, 2);
                    scale = absXi;
                } else {
                    ssq += std::pow(absXi/scale, 2);
                }
            }
            if (imag(x[i*incX]) != Zero) {
                Result absXi = std::abs(std::imag(x[i*incX]));
                if (scale<absXi) {
                    ssq = One + ssq * std::pow(scale/absXi, 2);
                    scale = absXi;
                } else {
                    ssq += std::pow(absXi/scale, 2);
                }
            }
        }
        result = scale*sqrt(ssq);
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL1_NRM2_TCC 1
