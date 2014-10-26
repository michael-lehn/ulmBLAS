#ifndef ULMBLAS_LAPACK_GETRF_TCC
#define ULMBLAS_LAPACK_GETRF_TCC 1

#include <algorithm>
#include <ulmblas/lapack/getrf.h>
#include <ulmblas/lapack/getf2.h>
#include <ulmblas/lapack/laenv.h>
#include <ulmblas/lapack/laswp.h>
#include <ulmblas/level3/gemm.h>
#include <ulmblas/level3/trlsm.h>

namespace ulmBLAS {

template <typename IndexType, typename T>
IndexType
getrf(IndexType    m,
      IndexType    n,
      T            *A,
      IndexType    incRowA,
      IndexType    incColA,
      IndexType    *piv,
      IndexType    incPiv)
{
    const T One(1);

    if (m==0 || n==0) {
        return 0;
    }

    const IndexType nb = laenv<T>(1, "GETRF", "", m, n);

    IndexType info = 0;

    if (nb<=1 || nb>=std::min(m,n)) {
        info = getf2(m, n, A, incRowA, incColA, piv, incPiv);
    } else {
        for (IndexType j=0; j<std::min(m,n); j+=nb) {
            const IndexType jb = std::min(std::min(m,n)-j, nb);

            IndexType info_ = getf2(m-j, jb,
                                    &A[j*(incRowA+incColA)], incRowA, incColA,
                                    &piv[j*incPiv], incPiv);

            if (info==0 && info_>0) {
                info = info_ + j;
            }

            for (IndexType i=j; i<std::min(m,j+jb); ++i) {
                piv[i*incPiv] += j;
            }

            laswp(j, A, incRowA, incColA, j, j+jb, piv, incPiv);

            if (j+jb<n) {
                laswp(n-j-jb,
                      &A[(j+jb)*incColA], incRowA, incColA,
                      j, j+jb,
                      piv, incPiv);

                trlsm(jb, n-j-jb, One, true,
                      &A[j*(incRowA+incColA)], incRowA, incColA,
                      &A[j*incRowA+(j+jb)*incColA], incRowA, incColA);

                if (j+jb<m) {
                    gemm(m-j-jb, n-j-jb, jb, -One,
                         &A[(j+jb)*incRowA + j*incColA], incRowA, incColA,
                         &A[j*incRowA + (j+jb)*incColA], incRowA, incColA,
                         One,
                         &A[(j+jb)*(incRowA+incColA)], incRowA, incColA);
                }
            }
        }
    }
    return info;
}

} // namespace ulmBLAS

#endif // ULMBLAS_LAPACK_GETRF_TCC
