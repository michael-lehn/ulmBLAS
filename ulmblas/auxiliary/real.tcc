#ifndef ULMBLAS_AUXILIARY_REAL_TCC
#define ULMBLAS_AUXILIARY_REAL_TCC 1

#include <ulmblas/auxiliary/real.h>
#include <complex>

namespace ulmBLAS {

template <typename T>
typename std::enable_if<std::is_fundamental<T>::value,
         const T &>::type
real(const T &x)
{
    return x;
}

template <typename T>
typename std::enable_if<! std::is_fundamental<T>::value,
         const T>::type
real(const T &x)
{
    return std::real(x);
}

} // namespace ulmBLAS

#endif // ULMBLAS_AUXILIARY_REAL_TCC
