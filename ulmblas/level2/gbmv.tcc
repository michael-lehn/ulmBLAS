#ifndef ULMBLAS_LEVEL2_GBMV_TCC
#define ULMBLAS_LEVEL2_GBMV_TCC 1

#include <ulmblas/level1/axpy.h>
#include <ulmblas/level1/scal.h>
#include <ulmblas/level2/gbmv.h>

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename TA, typename TX,
          typename Beta, typename TY>
void
gbmv(IndexType    m,
     IndexType    n,
     IndexType    kl,
     IndexType    ku,
     const Alpha  &alpha,
     bool         conjA,
     const TA     *A,
     IndexType    ldA,
     const TX     *x,
     IndexType    incX,
     const Beta   &beta,
     TY           *y,
     IndexType    incY)
{
    if (m==0 || n==0 || (alpha==Alpha(0) && beta==Beta(1))) {
        return;
    }

    scal(m, beta, y, incY);

    if (alpha==Alpha(0)) {
        return;
    }

    if (!conjA) {
        for (IndexType j=0; j<n; ++j) {
            IndexType i0  = std::max(IndexType(0), ku-j);
            IndexType i1  = ku+1+kl - std::max(IndexType(0), (j+1+kl)-m);
            IndexType len = std::max(IndexType(0), i1-i0);

            IndexType iY  = std::max(IndexType(0), j-ku);

            axpy(len, alpha*x[j*incX],
                 &A[i0], IndexType(1),
                 &y[iY*incY], incY);
            A += ldA;
        }
    } else {
        for (IndexType j=0; j<n; ++j) {
            IndexType i0  = std::max(IndexType(0), ku-j);
            IndexType i1  = ku+1+kl - std::max(IndexType(0), (j+1+kl)-m);
            IndexType len = std::max(IndexType(0), i1-i0);

            IndexType iY  = std::max(IndexType(0), j-ku);

            acxpy(len, alpha*x[j*incX],
                  &A[i0], IndexType(1),
                  &y[iY*incY], incY);
            A += ldA;
        }
    }
}

template <typename IndexType, typename Alpha, typename TA, typename TX,
          typename Beta, typename TY>
void
gbmv(IndexType    m,
     IndexType    n,
     IndexType    kl,
     IndexType    ku,
     const Alpha  &alpha,
     const TA     *A,
     IndexType    ldA,
     const TX     *x,
     IndexType    incX,
     const Beta   &beta,
     TY           *y,
     IndexType    incY)
{
    gbmv(m, n, kl, ku, alpha, false,  A, ldA, x, incX, beta, y, incY);
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_GBMV_TCC
