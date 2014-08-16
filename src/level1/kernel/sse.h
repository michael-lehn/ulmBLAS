#ifndef ULMBLAS_SRC_LEVEL1_KERNEL_SSE_H
#define ULMBLAS_SRC_LEVEL1_KERNEL_SSE_H 1

#include <type_traits>

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename VX, typename VY>
    void
    axpy(IndexType      n,
         const Alpha    &alpha,
         const VX       *x,
         IndexType      incX,
         VY             *y,
         IndexType      incY);

template <typename IndexType, typename Result>
static typename std::enable_if<std::is_convertible<double,Result>::value,
    void>::type
    dotu(IndexType      n,
         const double   *x,
         IndexType      incX,
         const double   *y,
         IndexType      incY,
         Result         &result);

} // namespace ulmBLAS

#endif // ULMBLAS_SRC_LEVEL1_KERNEL_SSE_H

#include <src/level1/kernel/sse.tcc>
