#include BLAS_HEADER
#include <complex>
#include <ulmblas/ulmblas.h>

extern "C" {

void
ULMBLAS(srotg)(float *a, float *b, float *c, float *s)
{
    ulmBLAS::rotg(*a, *b, *c, *s);
}

void
ULMBLAS(drotg)(double *a, double *b, double *c, double *s)
{
    ulmBLAS::rotg(*a, *b, *c, *s);
}

void
ULMBLAS(crotg)(float *a_, float *b_, float *c, float *s_)
{
    typedef std::complex<float> fcomplex;
    fcomplex *a = reinterpret_cast<fcomplex *>(a_);
    fcomplex *b = reinterpret_cast<fcomplex *>(b_);
    fcomplex *s = reinterpret_cast<fcomplex *>(s_);

    ulmBLAS::rotg(*a, *b, *c, *s);
}

void
ULMBLAS(zrotg)(double *a_, double *b_, double *c, double *s_)
{
    typedef std::complex<double> dcomplex;
    dcomplex *a = reinterpret_cast<dcomplex *>(a_);
    dcomplex *b = reinterpret_cast<dcomplex *>(b_);
    dcomplex *s = reinterpret_cast<dcomplex *>(s_);

    ulmBLAS::rotg(*a, *b, *c, *s);
}

void
CBLAS(srotg)(float *a, float *b, float *c, float *s)
{
    ulmBLAS::rotg(*a, *b, *c, *s);
}

void
CBLAS(drotg)(double *a, double *b, double *c, double *s)
{
    ulmBLAS::rotg(*a, *b, *c, *s);
}

void
CBLAS(crotg)(void *a_v, void *b_v, float *c, void *s_v)
{
    float *a = static_cast<float *>(a_v);
    float *b = static_cast<float *>(b_v);
    float *s = static_cast<float *>(s_v);

    ULMBLAS(crotg)(a, b, c, s);
}

void
CBLAS(zrotg)(void *a_v, void *b_v, double *c, void *s_v)
{
    double *a = static_cast<double *>(a_v);
    double *b = static_cast<double *>(b_v);
    double *s = static_cast<double *>(s_v);

    ULMBLAS(zrotg)(a, b, c, s);
}

} // extern "C"
