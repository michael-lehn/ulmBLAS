#ifndef ULMBLAS_LAPACK_POTRF2_TCC
#define ULMBLAS_LAPACK_POTRF2_TCC 1

#include <complex>
#include <type_traits>
#include <ulmblas/lapack/potf2.h>
#include <ulmblas/lapack/potrf2.h>
#include <ulmblas/lapack/unitstride.h>
#include <ulmblas/level3/helrk.h>
#include <ulmblas/level3/sylrk.h>
#include <ulmblas/level3/trlsm.h>

//  base case size (compile time parameter, tuned for double)
#ifndef ULMBLAS_POTRF2_BASE
#define ULMBLAS_POTRF2_BASE 64
#endif

namespace ulmBLAS {

namespace potrf2_impl {

//  C := C - A A^H, lower triangle (helrk for complex, sylrk for real)
template <typename IndexType, typename T>
typename std::enable_if<std::is_fundamental<T>::value>::type
herkLower(IndexType n, IndexType k, const T *A, IndexType rs, IndexType cs,
          T *C, IndexType crs, IndexType ccs)
{
    sylrk(n, k, T(-1), A, rs, cs, T(1), C, crs, ccs);
}

template <typename IndexType, typename T>
typename std::enable_if<!std::is_fundamental<T>::value>::type
herkLower(IndexType n, IndexType k, const T *A, IndexType rs, IndexType cs,
          T *C, IndexType crs, IndexType ccs)
{
    typedef decltype(std::real(T())) Real;
    helrk(n, k, Real(-1), A, rs, cs, Real(1), C, crs, ccs);
}

//  B := B L^{-H}, L lower triangular (m x m), B n x m
template <typename IndexType, typename T>
void
trsmRightLowerConjTrans(IndexType m, IndexType n,
                        const T *L, IndexType rs, IndexType cs,
                        T *B, IndexType brs, IndexType bcs)
{
    //  B L^H = B0  <=>  conj(L) B^T = B0^T
    trlsm(m, n, T(1), true, false, L, rs, cs, B, bcs, brs);
}

template <typename IndexType, typename T>
IndexType
rec(IndexType n, T *A, IndexType rs, IndexType cs)
{
    if (n<=ULMBLAS_POTRF2_BASE) {
        return potf2(n, A, rs, cs);
    }
    const IndexType n1 = n/2;
    const IndexType n2 = n-n1;

    IndexType info = rec(n1, A, rs, cs);
    if (info) {
        return info;
    }
    T *A21 = &A[n1*rs];
    T *A22 = &A[n1*(rs+cs)];

    trsmRightLowerConjTrans(n1, n2, A, rs, cs, A21, rs, cs);
    herkLower(n2, n1, A21, rs, cs, A22, rs, cs);

    info = rec(n2, A22, rs, cs);
    return info ? info+n1 : 0;
}

} // namespace potrf2_impl

template <typename IndexType, typename T>
IndexType
potrf2(IndexType    n,
       T            *A,
       IndexType    incRowA,
       IndexType    incColA)
{
    if (n<=0) {
        return 0;
    }
    return withUnitStride(n, n, A, incRowA, incColA, true,
                          [n](T *B, IndexType rs, IndexType cs) {
                              return potrf2_impl::rec(n, B, rs, cs);
                          });
}

} // namespace ulmBLAS

#endif // ULMBLAS_LAPACK_POTRF2_TCC
