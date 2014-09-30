#ifndef ULMBLAS_AUXILIARY_CONJ_H
#define ULMBLAS_AUXILIARY_CONJ_H 1

#include <type_traits>

namespace ulmBLAS {

template <typename T>
    typename std::enable_if<std::is_fundamental<T>::value,
             const T &>::type
    conj(const T &x, bool);

template <typename T>
    typename std::enable_if<! std::is_fundamental<T>::value,
             const T>::type
    conj(const T &x, bool apply);

} // namespace ulmBLAS

#endif // ULMBLAS_AUXILIARY_CONJ_H

#include <ulmblas/auxiliary/conj.tcc>
