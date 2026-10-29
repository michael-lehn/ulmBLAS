#include BLAS_HEADER
#include <complex>
#include <ulmblas/ulmblas.h>

extern "C" {

void
F77BLAS(scabs1_sub)(const float   *z_,
                    float         *result_)
{
    typedef std::complex<float> fcomplex;
    const fcomplex *z = reinterpret_cast<const fcomplex *>(z_);

    *result_ = ulmBLAS::abs1(*z);
}

void
F77BLAS(dcabs1_sub)(const double  *z_,
                    double        *result_)
{
    typedef std::complex<double> dcomplex;
    const dcomplex *z = reinterpret_cast<const dcomplex *>(z_);

    *result_ = ulmBLAS::abs1(*z);
}

} // extern "C"
