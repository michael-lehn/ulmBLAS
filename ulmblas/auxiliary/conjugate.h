#ifndef ULMBLAS_AUXILIARY_CONJUGATE_H
#define ULMBLAS_AUXILIARY_CONJUGATE_H 1

#include <complex>

namespace ulmBLAS {

// The complex conjugate (x for a real x), and the conjugate only if apply is
// true.  Returned by value, so that the result never refers to the argument.
template <typename T>
T
conjugate(const T &x);

template <typename T>
T
conjugate(const T &x, bool apply);

template <typename T>
std::complex<T>
conjugate(const std::complex<T> &z);

template <typename T>
std::complex<T>
conjugate(const std::complex<T> &z, bool apply);

} // namespace ulmBLAS

#endif // ULMBLAS_AUXILIARY_CONJUGATE_H

#include <ulmblas/auxiliary/conjugate.tcc>
