#ifndef ULMBLAS_AUXILIARY_ABS1_TCC
#define ULMBLAS_AUXILIARY_ABS1_TCC 1

#include <cmath>
#include <ulmblas/auxiliary/abs1.h>

namespace ulmBLAS {

template <typename T>
T
abs1(const T &x)
{
    return std::abs(x);
}

template <typename T>
T
abs1(const std::complex<T> &x)
{
    return std::abs(std::real(x)) + std::abs(std::imag(x));
}

} // namespace ulmBLAS

#endif // ULMBLAS_AUXILIARY_ABS1_TCC
