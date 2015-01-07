#ifndef ULMBLAS_AUXILIARY_ABS1_H
#define ULMBLAS_AUXILIARY_ABS1_H 1

#include <complex>

namespace ulmBLAS {

template <typename T>
    T
    abs1(const T &x);

template <typename T>
    T
    abs1(const std::complex<T> &x);

} // namespace ulmBLAS

#endif // ULMBLAS_AUXILIARY_ABS1_H

#include <ulmblas/auxiliary/abs1.tcc>
