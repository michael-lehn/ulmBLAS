#ifndef ULMBLAS_LEVEL3_KERNEL_SSE_UGEMM_H
#define ULMBLAS_LEVEL3_KERNEL_SSE_UGEMM_H 1

#include <type_traits>
#include <ulmblas/level3/kernel/ref/ugemm.h>

namespace ulmBLAS { namespace sse {

template <typename T>
    int
    ugemm_mr();

template <typename T>
    int
    ugemm_nr();

using ref::ugemm;

template <typename IndexType>
static typename std::enable_if<std::is_convertible<IndexType,long>::value,
    void>::type
    ugemm(IndexType      _kc,
          const double   &alpha,
          const double   *A,
          const double   *B,
          const double   &beta,
          double         *C,
          IndexType      _incRowC,
          IndexType      _incColC,
          const double   *nextA,
          const double   *nextB);

} } // namespace sse, ulmBLAS

#endif // ULMBLAS_LEVEL3_KERNEL_SSE_AXPY_H 1

#include <ulmblas/level3/kernel/sse/ugemm.tcc>
