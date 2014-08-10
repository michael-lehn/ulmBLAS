#ifndef ULMBLAS_SRC_CONFIG_BLOCKSIZE_H
#define ULMBLAS_SRC_CONFIG_BLOCKSIZE_H 1

#include <complex>
#include <type_traits>

namespace ulmBLAS {

#if defined(__SSE3__)

template <typename T>
struct BlockSize
{
    static const int MC = std::is_same<float, T>::value                ? 768
                        : std::is_same<double, T>::value               ? 384
                        : std::is_same<std::complex<float>, T>::value  ? 384
                        : std::is_same<std::complex<double>, T>::value ? 192
                        : -1;

    static const int KC = std::is_same<float, T>::value                ? 384
                        : std::is_same<double, T>::value               ? 384
                        : std::is_same<std::complex<float>, T>::value  ? 384
                        : std::is_same<std::complex<double>, T>::value ? 192
                        : -1;

    static const int NC = std::is_same<float, T>::value                ? 4096
                        : std::is_same<double, T>::value               ? 4096
                        : std::is_same<std::complex<float>, T>::value  ? 4096
                        : std::is_same<std::complex<double>, T>::value ? 4096
                        : -1;


    static const int MR = std::is_same<float, T>::value                ? 8
                        : std::is_same<double, T>::value               ? 4
                        : std::is_same<std::complex<float>, T>::value  ? 4
                        : std::is_same<std::complex<double>, T>::value ? 2
                        : -1;

    static const int NR = std::is_same<float, T>::value                ? 4
                        : std::is_same<double, T>::value               ? 4
                        : std::is_same<std::complex<float>, T>::value  ? 2
                        : std::is_same<std::complex<double>, T>::value ? 2
                        : -1;
};

#else

template <typename T>
struct BlockSize
{
    static const int MC = std::is_same<float, T>::value                ? 768
                        : std::is_same<double, T>::value               ? 384
                        : std::is_same<std::complex<float>, T>::value  ? 384
                        : std::is_same<std::complex<double>, T>::value ? 192
                        : -1;

    static const int KC = std::is_same<float, T>::value                ? 384
                        : std::is_same<double, T>::value               ? 384
                        : std::is_same<std::complex<float>, T>::value  ? 384
                        : std::is_same<std::complex<double>, T>::value ? 192
                        : -1;

    static const int NC = std::is_same<float, T>::value                ? 4096
                        : std::is_same<double, T>::value               ? 4096
                        : std::is_same<std::complex<float>, T>::value  ? 4096
                        : std::is_same<std::complex<double>, T>::value ? 4096
                        : -1;


    static const int MR = std::is_same<float, T>::value                ? 8
                        : std::is_same<double, T>::value               ? 4
                        : std::is_same<std::complex<float>, T>::value  ? 4
                        : std::is_same<std::complex<double>, T>::value ? 2
                        : -1;

    static const int NR = std::is_same<float, T>::value                ? 4
                        : std::is_same<double, T>::value               ? 4
                        : std::is_same<std::complex<float>, T>::value  ? 2
                        : std::is_same<std::complex<double>, T>::value ? 2
                        : -1;
};

#endif



} // namespace ulmBLAS

#endif // ULMBLAS_SRC_CONFIG_BLOCKSIZE_H 1
