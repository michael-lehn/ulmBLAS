#ifndef ULMBLAS_LEVEL2_GER_H
#define ULMBLAS_LEVEL2_GER_H 1

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename TX, typename TY,
          typename TA>
    void
    ger(IndexType    m,
        IndexType    n,
        const Alpha  &alpha,
        const TX     *x,
        IndexType    incX,
        const TY     *y,
        IndexType    incY,
        TA           *A,
        IndexType    incRowA,
        IndexType    incColA);

template <typename IndexType, typename Alpha, typename TX, typename TY,
          typename TA>
    void
    gerc(IndexType    m,
         IndexType    n,
         const Alpha  &alpha,
         bool         conj,
         const TX     *x,
         IndexType    incX,
         const TY     *y,
         IndexType    incY,
         TA           *A,
         IndexType    incRowA,
         IndexType    incColA);

template <typename IndexType, typename Alpha, typename TX, typename TY,
          typename TA>
    void
    gerc(IndexType    m,
         IndexType    n,
         const Alpha  &alpha,
         const TX     *x,
         IndexType    incX,
         const TY     *y,
         IndexType    incY,
         TA           *A,
         IndexType    incRowA,
         IndexType    incColA);

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_GER_H

#include <ulmblas/level2/ger.tcc>
