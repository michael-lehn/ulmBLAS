#ifndef ULMBLAS_LAPACK_TRTRS_TCC
#define ULMBLAS_LAPACK_TRTRS_TCC 1

#include <ulmblas/lapack/islapacktype.h>
#include <ulmblas/lapack/trtrs.h>
#include <ulmblas/lapack/tasks/solvetasks.h>

namespace ulmBLAS {

template <typename IndexType, typename T>
IndexType
trtrs(bool         lower,
      bool         trans,
      bool         conj,
      bool         unitDiag,
      IndexType    n,
      IndexType    nrhs,
      const T      *A,
      IndexType    incRowA,
      IndexType    incColA,
      T            *B,
      IndexType    incRowB,
      IndexType    incColB)
{
    ULMBLAS_LAPACK_REQUIRE_FLOATING_POINT(T);

    if (n<=0) {
        return 0;
    }
    if (!unitDiag) {
        for (IndexType i=0; i<n; ++i) {
            if (A[i*(incRowA+incColA)]==T(0)) {
                return i+1;
            }
        }
    }
    if (nrhs<=0) {
        return 0;
    }
    typedef tasks::SolveStep<IndexType, T> Step;
    Step s;
    s.kind     = Step::Solve;
    s.piv      = 0;
    s.A        = A;
    s.conj     = conj;
    s.unitDiag = unitDiag;
    s.upper    = trans ? lower : !lower;
    s.incRowA  = trans ? incColA : incRowA;
    s.incColA  = trans ? incRowA : incColA;
    tasks::solve(n, nrhs, &s, 1, B, incRowB, incColB);
    return 0;
}

} // namespace ulmBLAS

#endif // ULMBLAS_LAPACK_TRTRS_TCC
