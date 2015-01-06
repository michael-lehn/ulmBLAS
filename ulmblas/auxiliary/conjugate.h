#ifndef ULMBLAS_AUXILIARY_CONJUGATE_H
#define ULMBLAS_AUXILIARY_CONJUGATE_H 1

#include <type_traits>

namespace ulmBLAS {

template <typename T>
    typename std::enable_if<std::is_fundamental<T>::value,
             const T &>::type
    conjugate(const T &x);

template <typename T>
    typename std::enable_if<std::is_fundamental<T>::value,
             const T &>::type
    conjugate(const T &x, bool);

template <typename T>
    typename std::enable_if<! std::is_fundamental<T>::value,
             const T>::type
    conjugate(const T &x);

template <typename T>
    typename std::enable_if<! std::is_fundamental<T>::value,
             const T>::type
    conjugate(const T &x, bool apply);

} // namespace ulmBLAS

#endif // ULMBLAS_AUXILIARY_CONJUGATE_H

#include <ulmblas/auxiliary/conjugate.tcc>
