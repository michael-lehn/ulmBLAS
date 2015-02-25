#ifndef ULMBLAS_AUXILIARY_REAL_H
#define ULMBLAS_AUXILIARY_REAL_H 1

#include <type_traits>

namespace ulmBLAS {

template <typename T>
    typename std::enable_if<std::is_fundamental<T>::value,
             const T &>::type
    real(const T &x);

template <typename T>
    typename std::enable_if<! std::is_fundamental<T>::value,
             const T>::type
    real(const T &x);

} // namespace ulmBLAS

#endif // ULMBLAS_AUXILIARY_REAL_H

#include <ulmblas/auxiliary/real.tcc>
