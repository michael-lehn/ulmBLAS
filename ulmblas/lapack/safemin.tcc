#ifndef ULMBLAS_LAPACK_SAFEMIN_TCC
#define ULMBLAS_LAPACK_SAFEMIN_TCC 1

#include <limits>
#include <ulmblas/lapack/safemin.h>

namespace ulmBLAS {

template <typename T>
T
safeMin()
{
    const T eps   = std::numeric_limits<T>::epsilon() * 0.5;
    const T small = T(1) / std::numeric_limits<T>::max();

    T sMin  = std::numeric_limits<T>::min();

    if (small>=sMin) {
//
//      Use SMALL plus a bit, to avoid the possibility of rounding
//      causing overflow when computing  1/sfmin.
//
        sMin = small*(1.0+eps);
    }
    return sMin;
}

} // namespace ulmBLAS

#endif // ULMBLAS_LAPACK_SAFEMIN_TCC
