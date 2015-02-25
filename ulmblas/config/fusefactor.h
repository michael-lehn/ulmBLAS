#ifndef ULMBLAS_CONFIG_FUSEFACTOR_H
#define ULMBLAS_CONFIG_FUSEFACTOR_H 1

#include <complex>
#include <type_traits>

#include <ulmblas/config/simd.h>

namespace ulmBLAS {

#if defined(USE_TESTPARAM)

#define DAXPYF_FUSEFACTOR   4
#define ZAXPYF_FUSEFACTOR   4
#define DDOTUXF_FUSEFACTOR  4
#define ZDOTUXF_FUSEFACTOR  4

template <typename T>
struct FuseFactor
{
    typedef std::complex<double>  dcomplex;

    static const int axpyf = std::is_same<T,double>::value   ? DAXPYF_FUSEFACTOR
                           : std::is_same<T,dcomplex>::value ? ZAXPYF_FUSEFACTOR
                           : 1;

    static const int acxpyf = axpyf;

    static const int dotuxf = std::is_same<T,double>::value ? DDOTUXF_FUSEFACTOR
                           : std::is_same<T,dcomplex>::value? ZDOTUXF_FUSEFACTOR
                           : 1;

    static const int dotxaxpyf = axpyf;
};


#elif defined(USE_SSE)

template <typename T>
struct FuseFactor
{
    typedef std::complex<double>  dcomplex;

    static const int axpyf = std::is_same<T,double>::value   ? 2
                           : std::is_same<T,dcomplex>::value ? 4
                           : 1;

    static const int acxpyf = axpyf;

    static const int dotuxf = std::is_same<T,double>::value ? 4
                           : std::is_same<T,dcomplex>::value? 4
                           : 1;

    static const int dotxaxpyf = axpyf;
};

#else

template <typename T>
struct FuseFactor
{
    typedef std::complex<double>  dcomplex;

    static const int axpyf = std::is_same<T,double>::value   ? 4
                           : std::is_same<T,dcomplex>::value ? 4
                           : 1;

    static const int acxpyf = axpyf;

    static const int dotuxf = std::is_same<T,double>::value   ? 4
                            : std::is_same<T,dcomplex>::value ? 4
                            : 1;

    static const int dotxaxpyf = axpyf;
};


#endif



} // namespace ulmBLAS

#endif // ULMBLAS_CONFIG_BLOCKSIZE_H 1
