#ifndef ULMBLAS_LAPACK_LASWP_TCC
#define ULMBLAS_LAPACK_LASWP_TCC 1

#include <iostream>

#include <algorithm>
#include <ulmblas/lapack/laswp.h>
#include <ulmblas/level1/swap.h>

namespace ulmBLAS {

template <typename IndexType, typename T>
void
laswp(IndexType    n,
      T            *A,
      IndexType    incRowA,
      IndexType    incColA,
      IndexType    k1,
      IndexType    k2,
      IndexType    *piv,
      IndexType    incPiv)
{
    const IndexType inc = (incPiv > 0) ? 1    : -1;
    const IndexType i1  = (incPiv > 0) ? k1   : k2-1;
    const IndexType i2  = (incPiv > 0) ? k2-1 : k1;

    if (incPiv==0 || n==0) {
        return;
    }

    const IndexType n32 = (n/32)*32;

    if (n32!=0) {
        for (IndexType j=0; j<n32; j+=32) {
            for (IndexType i=i1; i<=i2; i+=inc) {
                IndexType ip = piv[i*incPiv];
                if (ip!=i) {
                    swap(32,
                         &A[ip*incRowA+j*incColA], incColA,
                         &A[i *incRowA+j*incColA], incColA);
                }
            }
        }
    }
    if (n32!=n) {
        for (IndexType i=i1; i<=i2; i+=inc) {
            IndexType ip = piv[i*incPiv];
            if (ip!=i) {
                swap(n-n32,
                     &A[ip*incRowA+n32*incColA], incColA,
                     &A[i *incRowA+n32*incColA], incColA);
            }
        }
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_LAPACK_LASWP_TCC
