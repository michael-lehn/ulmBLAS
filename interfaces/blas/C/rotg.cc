#include BLAS_HEADER
#include <complex>
#include <ulmblas/ulmblas.h>

extern "C" {

void
ULMBLAS(srotg)(float *a,
               float *b,
               float *c,
               float *s)
{
    ulmBLAS::rotg(*a, *b, *c, *s);
}

void
ULMBLAS(drotg)(double *a,
               double *b,
               double *c,
               double *s)
{
    ulmBLAS::rotg(*a, *b, *c, *s);
}

void
ULMBLAS(crotg)(float *a_,
               float *b_,
               float *c,
               float *s_)
{
    typedef std::complex<float> fcomplex;
    fcomplex *a = reinterpret_cast<fcomplex *>(a_);
    fcomplex *b = reinterpret_cast<fcomplex *>(b_);
    fcomplex *s = reinterpret_cast<fcomplex *>(s_);

    ulmBLAS::rotg(*a, *b, *c, *s);
}

void
ULMBLAS(zrotg)(double *a_,
               double *b_,
               double *c,
               double *s_)
{
    typedef std::complex<double> dcomplex;
    dcomplex *a = reinterpret_cast<dcomplex *>(a_);
    dcomplex *b = reinterpret_cast<dcomplex *>(b_);
    dcomplex *s = reinterpret_cast<dcomplex *>(s_);

    ulmBLAS::rotg(*a, *b, *c, *s);
}

void
CBLAS(srotg)(float *a,
             float *b,
             float *c,
             float *s)
{
    ulmBLAS::rotg(*a, *b, *c, *s);
}

void
CBLAS(drotg)(double *a,
             double *b,
             double *c,
             double *s)
{
    ulmBLAS::rotg(*a, *b, *c, *s);
}

void
CBLAS(crotg)(float *a,
             float *b,
             float *c,
             float *s)
{
    ULMBLAS(crotg)(a, b, c, s);
}

void
CBLAS(zrotg)(double *a,
             double *b,
             double *c,
             double *s)
{
    ULMBLAS(zrotg)(a, b, c, s);
}

} // extern "C"
