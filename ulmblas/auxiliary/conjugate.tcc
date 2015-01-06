#ifndef ULMBLAS_AUXILIARY_CONJ_TCC
#define ULMBLAS_AUXILIARY_CONJ_TCC 1

#include <complex>
#include <ulmblas/auxiliary/conjugate.h>

namespace ulmBLAS {

template <typename T>
typename std::enable_if<std::is_fundamental<T>::value,
         const T &>::type
conjugate(const T &x)
{
    return x;
}

template <typename T>
typename std::enable_if<std::is_fundamental<T>::value,
         const T &>::type
conjugate(const T &x, bool)
{
    return x;
}

template <typename T>
typename std::enable_if<! std::is_fundamental<T>::value,
         const T>::type
conjugate(const T &x)
{
    return std::conj(x);
}

template <typename T>
typename std::enable_if<! std::is_fundamental<T>::value,
         const T>::type
conjugate(const T &x, bool apply)
{
    return (apply) ? std::conj(x) : x;
}

} // namespace ulmBLAS

#endif // ULMBLAS_AUXILIARY_CONJ_TCC
