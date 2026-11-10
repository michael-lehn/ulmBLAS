#include BLAS_HEADER
#include <complex>
#include <ulmblas/ulmblas.h>

extern "C" {

float
ULMBLAS(scabs1)(const float *z_)
{
    typedef std::complex<float> fcomplex;
    const fcomplex *z = reinterpret_cast<const fcomplex *>(z_);

    return ulmBLAS::abs1(*z);
}

double
ULMBLAS(dcabs1)(const double *z_)
{
    typedef std::complex<double> dcomplex;
    const dcomplex *z = reinterpret_cast<const dcomplex *>(z_);

    return ulmBLAS::abs1(*z);
}

float
CBLAS(scabs1)(const void *z_v)
{
    const float *z = static_cast<const float *>(z_v);

    return ULMBLAS(scabs1)(z);
}

double
CBLAS(dcabs1)(const void *z_v)
{
    const double *z = static_cast<const double *>(z_v);

    return ULMBLAS(dcabs1)(z);
}

} // extern "C"
