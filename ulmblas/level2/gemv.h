#ifndef ULMBLAS_LEVEL2_GEMV_H
#define ULMBLAS_LEVEL2_GEMV_H 1

namespace ulmBLAS {

//  on one thread (for callers that run gemv in small pieces, e.g. trsv)
template <typename IndexType, typename Alpha, typename TA, typename TX,
          typename Beta, typename TY>
void
gemvSerial(IndexType m, IndexType n, const Alpha &alpha, bool conjA,
           const TA *A, IndexType incRowA, IndexType incColA, const TX *x,
           IndexType incX, const Beta &beta, TY *y, IndexType incY);

template <typename IndexType, typename Alpha, typename TA, typename TX,
          typename Beta, typename TY>
void
gemv(IndexType m, IndexType n, const Alpha &alpha, bool conjA, const TA *A,
     IndexType incRowA, IndexType incColA, const TX *x, IndexType incX,
     const Beta &beta, TY *y, IndexType incY);

template <typename IndexType, typename Alpha, typename TA, typename TX,
          typename Beta, typename TY>
void
gemv(IndexType m, IndexType n, const Alpha &alpha, const TA *A,
     IndexType incRowA, IndexType incColA, const TX *x, IndexType incX,
     const Beta &beta, TY *y, IndexType incY);

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_GEMV_H

#include <ulmblas/level2/gemv.tcc>
