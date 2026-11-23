#ifndef ULMBLAS_AUXILIARY_CONJ_TCC
#define ULMBLAS_AUXILIARY_CONJ_TCC 1

#include <ulmblas/auxiliary/conjugate.h>

namespace ulmBLAS {

template <typename T>
T
conjugate(const T &x)
{
    return x;
}

template <typename T>
T
conjugate(const T &x, bool apply)
{
    (void)apply;
    return x;
}

template <typename T>
std::complex<T>
conjugate(const std::complex<T> &z)
{
    return std::conj(z);
}

template <typename T>
std::complex<T>
conjugate(const std::complex<T> &z, bool apply)
{
    return apply ? std::conj(z) : z;
}

} // namespace ulmBLAS

#endif // ULMBLAS_AUXILIARY_CONJ_TCC
