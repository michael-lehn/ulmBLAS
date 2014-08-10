#ifndef ULMBLAS_SRC_LEVEL3_DGEMM_H
#define ULMBLAS_SRC_LEVEL3_DGEMM_H 1

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename MA, typename MB,
          typename Beta, typename MC>
    void
    gemm(IndexType    m,
         IndexType    n,
         IndexType    k,
         Alpha        alpha,
         const MA     *A,
         IndexType    incRowA,
         IndexType    incColA,
         const MA     *B,
         IndexType    incRowB,
         IndexType    incColB,
         Beta         beta,
         MA           *C,
         IndexType    incRowC,
         IndexType    incColC);

} // namespace ulmBLAS

#endif // ULMBLAS_SRC_LEVEL3_DGEMM_H
