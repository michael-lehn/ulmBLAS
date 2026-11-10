#ifndef ULMBLAS_LAPACK_POTRS_TCC
#define ULMBLAS_LAPACK_POTRS_TCC 1

#include <ulmblas/lapack/islapacktype.h>
#include <ulmblas/lapack/potrs.h>
#include <ulmblas/lapack/tasks/solvetasks.h>

namespace ulmBLAS {

template <typename IndexType, typename T>
void
potrs(bool lower, IndexType n, IndexType nrhs, const T *A, IndexType incRowA,
      IndexType incColA, T *B, IndexType incRowB, IndexType incColB)
{
    ULMBLAS_LAPACK_REQUIRE_FLOATING_POINT(T);

    if (n <= 0 || nrhs <= 0) {
	return;
    }
    //  the lower factor L (lower: as stored; upper: L = U^H, the transposed
    //  view, conjugated): L Y = B, then L^H X = Y
    const IndexType rs = lower ? incRowA : incColA;
    const IndexType cs = lower ? incColA : incRowA;
    const bool conj = !lower;

    typedef tasks::SolveStep<IndexType, T> Step;
    Step s[2];
    s[0].kind = Step::Solve;
    s[0].piv = 0;
    s[0].A = A;
    s[0].unitDiag = false;
    s[0].upper = false;
    s[0].conj = conj;
    s[0].incRowA = rs;
    s[0].incColA = cs;
    s[1] = s[0];
    s[1].upper = true;
    s[1].conj = !conj;
    s[1].incRowA = cs;
    s[1].incColA = rs;
    tasks::solve(n, nrhs, s, 2, B, incRowB, incColB);
}

} // namespace ulmBLAS

#endif // ULMBLAS_LAPACK_POTRS_TCC
