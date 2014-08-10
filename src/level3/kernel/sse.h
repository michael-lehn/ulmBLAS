#ifndef ULMBLAS_SRC_LEVEL3_KERNEL_SSE_H
#define ULMBLAS_SRC_LEVEL3_KERNEL_SSE_H 1

#include <type_traits>

namespace ulmBLAS {

template <typename IndexType>
static typename std::enable_if<std::is_convertible<IndexType,long>::value
                            && BlockSize<double>::MR==4
                            && BlockSize<double>::NR==4,
    void>::type
    gemm_micro_kernel(IndexType _kc,
                      const double &alpha,
                      const double *A, const double *B,
                      const double &beta,
                      double *C, IndexType _incRowC, IndexType _incColC,
                      const double *nextA, const double *nextB);

} // namespace ulmBLAS

#endif // ULMBLAS_SRC_LEVEL3_KERNEL_SSE_H
