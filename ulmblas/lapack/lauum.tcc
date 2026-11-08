#ifndef ULMBLAS_LAPACK_LAUUM_TCC
#define ULMBLAS_LAPACK_LAUUM_TCC 1

#include <complex>
#include <type_traits>
#include <ulmblas/auxiliary/conjugate.h>
#include <ulmblas/lapack/islapacktype.h>
#include <ulmblas/lapack/lauum.h>
#include <ulmblas/lapack/threads.h>
#include <ulmblas/lapack/tasks/inversetasks.h>
#include <ulmblas/lapack/tasks/param.h>
#include <ulmblas/level3/heurk.h>
#include <ulmblas/level3/syurk.h>
#include <ulmblas/level3/trumm.h>

namespace ulmBLAS {

namespace lauum_impl {

//  C := C + A A^H, upper triangle (heurk for complex, syurk for real)
template <typename IndexType, typename T>
typename std::enable_if<std::is_fundamental<T>::value>::type
herkUpper(IndexType n, IndexType k, const T *A, IndexType rs, IndexType cs,
          T *C, IndexType crs, IndexType ccs)
{
    syurk(n, k, T(1), A, rs, cs, T(1), C, crs, ccs);
}

template <typename IndexType, typename T>
typename std::enable_if<!std::is_fundamental<T>::value>::type
herkUpper(IndexType n, IndexType k, const T *A, IndexType rs, IndexType cs,
          T *C, IndexType crs, IndexType ccs)
{
    typedef decltype(std::real(T())) Real;
    heurk(n, k, Real(1), A, rs, cs, Real(1), C, crs, ccs);
}

//  upper: A := U U^H, unblocked (as LAPACK's ?lauu2)
template <typename IndexType, typename T>
void
lauu2(IndexType n, T *A, IndexType rs, IndexType cs)
{
    typedef decltype(std::real(T())) Real;

    for (IndexType i=0; i<n; ++i) {
        const Real aii = std::real(A[i*(rs+cs)]);
        if (i<n-1) {
            Real d = aii*aii;
            for (IndexType k=i+1; k<n; ++k) {
                d += std::norm(A[i*rs+k*cs]);
            }
            A[i*(rs+cs)] = d;
            for (IndexType r=0; r<i; ++r) {
                T s = A[r*rs+i*cs]*aii;
                for (IndexType k=i+1; k<n; ++k) {
                    s += A[r*rs+k*cs]*conjugate(A[i*rs+k*cs]);
                }
                A[r*rs+i*cs] = s;
            }
        } else {
            for (IndexType r=0; r<=i; ++r) {
                A[r*rs+i*cs] *= aii;
            }
        }
    }
}

#ifndef ULMBLAS_LAUUM_BASE
#define ULMBLAS_LAUUM_BASE 32
#endif

//  upper: [U11 U12; 0 U22] U U^H = [U11 U11^H + U12 U12^H, U12 U22^H; .,
//  U22 U22^H]
template <typename IndexType, typename T>
void
rec(IndexType n, T *A, IndexType rs, IndexType cs)
{
    if (n<=ULMBLAS_LAUUM_BASE) {
        lauu2(n, A, rs, cs);
        return;
    }
    const IndexType n1 = n/2, n2 = n-n1;
    T *A12 = &A[n1*cs];
    T *A22 = &A[n1*(rs+cs)];

    rec(n1, A, rs, cs);
    herkUpper(n1, n2, A12, rs, cs, A, rs, cs);
    //  A12 := A12 U22^H, i.e. A12^T := conj(U22) A12^T
    trumm(n2, n1, T(1), true, false, A22, rs, cs, A12, cs, rs);
    rec(n2, A22, rs, cs);
}

} // namespace lauum_impl

template <typename IndexType, typename T>
void
lauum(bool         lower,
      IndexType    n,
      T            *A,
      IndexType    incRowA,
      IndexType    incColA)
{
    ULMBLAS_LAPACK_REQUIRE_FLOATING_POINT(T);

    if (n<=0) {
        return;
    }
    //  lower: with the transposed view U = L^T, element (i,j) of U U^H is
    //  sum_l L(l,i) conj(L(l,j)), stored at (j,i): element (j,i) of L^H L
    const IndexType rs = lower ? incColA : incRowA;
    const IndexType cs = lower ? incRowA : incColA;
#ifdef _OPENMP
    if (!omp_in_parallel()) {
        const tasks::Param p = tasks::inverseParam(long(n), numThreads());
        if (n>IndexType(p.serialN) && p.threads>1) {
            tasks::lauumUpperTasks(n, A, rs, cs, IndexType(p.nb), p.threads);
            return;
        }
    }
#endif
    lauum_impl::rec(n, A, rs, cs);
}

} // namespace ulmBLAS

#endif // ULMBLAS_LAPACK_LAUUM_TCC
