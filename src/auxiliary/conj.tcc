#ifndef SRC_AUXILIARY_CONJ_TCC
#define SRC_AUXILIARY_CONJ_TCC 1

#include <complex>
#include <src/auxiliary/conj.tcc>

namespace ulmBLAS {

template <typename T>
typename std::enable_if<std::is_fundamental<T>::value,
         const T &>::type
conj(const T &x, bool)
{
    return x;
}

template <typename T>
typename std::enable_if<! std::is_fundamental<T>::value,
         const T>::type
conj(const T &x, bool apply)
{
    return (apply) ? std::conj(x) : x;
}

} // namespace ulmBLAS

#endif // SRC_AUXILIARY_CONJ_TCC
