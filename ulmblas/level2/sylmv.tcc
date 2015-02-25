#ifndef ULMBLAS_LEVEL2_SYLMV_TCC
#define ULMBLAS_LEVEL2_SYLMV_TCC 1

#include <ulmblas/config/fusefactor.h>
#include <ulmblas/level1/scal.h>
#include <ulmblas/level1/axpy.h>
#include <ulmblas/level1extensions/dotaxpy.h>
#include <ulmblas/level1extensions/dotxaxpyf.h>
#include <ulmblas/level2/sylmv.h>

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename TA, typename TX,
          typename Beta, typename TY>
void
sylmv(IndexType    n,
      const Alpha  &alpha,
      const TA     *A,
      IndexType    incRowA,
      IndexType    incColA,
      const TX     *x,
      IndexType    incX,
      const Beta   &beta,
      TY           *y,
      IndexType    incY)
{
    typedef decltype(Alpha(0)*TA(0)*TX(0)+Beta(0)*TY(0))  T;

    const IndexType    UnitStride(1);
    static const bool  homogeneousTypes = std::is_same<T,Alpha>::value
                                       && std::is_same<T,TA>::value
                                       && std::is_same<T,TX>::value
                                       && std::is_same<T,TY>::value;

    scal(n, beta, y, incY);

    if (homogeneousTypes && incRowA==UnitStride) {
        const IndexType bf = FuseFactor<T>::dotxaxpyf;
        const IndexType nb = (n/bf)*bf;

        T rho, rho_[bf];

        for (IndexType j=0; j<nb; j+=bf) {

            dotxaxpyf(n-j-bf, false, false, false,
                      alpha, &x[j*incX], incX,
                      &A[(j+bf)*incRowA+j*incColA], incRowA, incColA,
                      &x[(j+bf)*incX], incX,
                      &y[(j+bf)*incY], incY,
                      rho_, 1);

            for (IndexType l=0; l<bf; ++l) {
                dotaxpy(bf-1-l, false, false, false,
                        alpha*x[(j+l)*incX],
                        &A[(j+l+1)*incRowA+(j+l)*incColA], incRowA,
                        &x[(j+l+1)*incX], incX,
                        &y[(j+l+1)*incY], incY,
                        rho);
                y[(j+l)*incY] += alpha*(rho+rho_[l]);
                y[(j+l)*incY] += alpha*A[(j+l)*incRowA+(j+l)*incColA]
                                      *x[(j+l)*incX];
            }

        }
        for (IndexType j=nb; j<n; ++j) {
            dotaxpy(n-1-j, false, false, false,
                    alpha*x[j*incX],
                    &A[(j+1)*incRowA+j*incColA], incRowA,
                    &x[(j+1)*incX], incX,
                    &y[(j+1)*incY], incY,
                    rho);
            y[j*incY] += alpha*(A[j*incRowA+j*incColA]*x[j*incX]+rho);
        }
    } else if (homogeneousTypes && incColA==UnitStride) {
        const IndexType bf = FuseFactor<T>::dotxaxpyf;
        const IndexType nb = (n/bf)*bf;

        T rho, rho_[bf];

        for (IndexType i=0; i<nb; i+=bf) {

            dotxaxpyf(i, false, false, false,
                      alpha, &x[i*incX], incX,
                      &A[i*incRowA], incColA, incRowA,
                      &x[0*incX], incX,
                      &y[0*incY], incY,
                      rho_, 1);

            for (IndexType l=0; l<bf; ++l) {
                dotaxpy(l, false, false, false,
                        alpha*x[(i+l)*incX],
                        &A[(i+l)*incRowA+i*incColA], incColA,
                        &x[i*incX], incX,
                        &y[i*incY], incY,
                        rho);
                y[(i+l)*incY] += alpha*(rho+rho_[l]);
                y[(i+l)*incY] += alpha*A[(i+l)*incRowA+(i+l)*incColA]
                                      *x[(i+l)*incX];
            }
        }
        for (IndexType i=nb; i<n; ++i) {
            dotaxpy(i, false, false, false,
                    alpha*x[i*incX],
                    &A[i*incRowA], incColA,
                    &x[0*incX], incX,
                    &y[0*incY], incY,
                    rho);
            y[i*incY] += alpha*(A[i*incRowA+i*incColA]*x[i*incX]+rho);
        }
    } else {
//
//      Otherwise we just use a variant with axpy as reference implementation.
//      TODO: packing, switching between dot/axpy variant depending on
//            abs(incRowA) and abs(incColA)
//
        for (IndexType i=0; i<n; ++i) {
            axpy(i, alpha*x[i*incX], &A[i*incRowA], incColA, &y[0*incY], incY);
            y[i*incY] += alpha*A[i*incRowA+i*incColA]*x[i*incX];
            axpy(n-1-i, alpha*x[i*incX], &A[(i+1)*incRowA+i*incColA], incRowA,
                 &y[(i+1)*incY], incY);
        }
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_SYLMV_TCC
