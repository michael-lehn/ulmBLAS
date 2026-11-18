#ifndef ULMBLAS_LAPACK_GETRS_TCC
#define ULMBLAS_LAPACK_GETRS_TCC 1

#include <ulmblas/lapack/getrs.h>
#include <ulmblas/lapack/islapacktype.h>
#include <ulmblas/lapack/tasks/solvetasks.h>

namespace ulmBLAS {

template <typename IndexType, typename T, typename PivIndex>
void
getrs(bool trans, bool conj, IndexType n, IndexType nrhs, const T *A,
      IndexType incRowA, IndexType incColA, const PivIndex *piv,
      IndexType incPiv, T *B, IndexType incRowB, IndexType incColB,
      IndexType pivBase)
{
    ULMBLAS_LAPACK_REQUIRE_FLOATING_POINT(T);

    if (n <= 0 || nrhs <= 0) {
	return;
    }
    typedef tasks::SolveStep<IndexType, T, PivIndex> Step;
    Step s[3];
    for (Step &x : s) {
	x.piv = piv;
	x.incPiv = incPiv;
	x.pivBase = pivBase;
	x.A = A;
	x.conj = false;
	x.unitDiag = false;
	x.upper = false;
	x.incRowA = incRowA;
	x.incColA = incColA;
    }
    if (!trans) {
	//  L U X = P B
	s[0].kind = Step::SwapForward;
	s[1].kind = Step::Solve;
	s[1].unitDiag = true;
	s[1].conj = conj;
	s[2].kind = Step::Solve;
	s[2].upper = true;
	s[2].conj = conj;
    } else {
	//  A^T = U^T L^T P^T: U^T Y = B, L^T Z = Y, X = P Z (transposed
	//  views: strides exchanged, upper <-> lower)
	s[0].kind = Step::Solve;
	s[0].conj = conj;
	s[0].incRowA = incColA;
	s[0].incColA = incRowA;
	s[1].kind = Step::Solve;
	s[1].upper = true;
	s[1].unitDiag = true;
	s[1].conj = conj;
	s[1].incRowA = incColA;
	s[1].incColA = incRowA;
	s[2].kind = Step::SwapBackward;
    }
    tasks::solve(n, nrhs, s, 3, B, incRowB, incColB);
}

} // namespace ulmBLAS

#endif // ULMBLAS_LAPACK_GETRS_TCC
