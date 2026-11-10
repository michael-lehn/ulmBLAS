#ifndef ULMBLAS_LAPACK_ISLAPACKTYPE_H
#define ULMBLAS_LAPACK_ISLAPACKTYPE_H 1

#include <complex>
#include <type_traits>

namespace ulmBLAS {

//
//  Element types of the LAPACK functions (getrf, potrf, ...): floating-
//  point numbers and complex numbers of them.  The algorithms divide (and
//  take square roots) and choose pivots by size, so they need a field with
//  rounding.  For integers they would silently compute nonsense (integer
//  division truncates), so they do not compile:
//
//      static_assert(IsLapackType<T>::value, ...)
//
//  Other floating-point types (e.g. a double-double type) are enabled by a
//  specialization:
//
//      template <> struct IsLapackType<MyFloat> : std::true_type {};
//
template <typename T>
struct IsLapackType
    : std::integral_constant<bool, std::is_floating_point<T>::value>
{
};

template <typename T> struct IsLapackType<std::complex<T>> : IsLapackType<T>
{
};

} // namespace ulmBLAS

#define ULMBLAS_LAPACK_REQUIRE_FLOATING_POINT(T)                               \
    static_assert(::ulmBLAS::IsLapackType<T>::value,                           \
                  "ulmBLAS LAPACK functions need a floating-point (or "        \
                  "complex floating-point) element type, see "                 \
                  "ulmblas/lapack/islapacktype.h")

#endif // ULMBLAS_LAPACK_ISLAPACKTYPE_H
