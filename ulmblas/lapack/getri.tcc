#ifndef ULMBLAS_LAPACK_GETRI_TCC
#define ULMBLAS_LAPACK_GETRI_TCC 1

#include <utility>
#include <ulmblas/lapack/buffer.h>
#include <ulmblas/lapack/getri.h>
#include <ulmblas/lapack/trtri.h>
#include <ulmblas/lapack/tasks/solvetasks.h>

namespace ulmBLAS {

template <typename IndexType, typename T, typename PivIndex>
IndexType
getri(IndexType n, T *A, IndexType incRowA, IndexType incColA,
      const PivIndex *piv, IndexType incPiv)
{
    const IndexType rs = incRowA, cs = incColA;

    const IndexType info = trtri(false, false, n, A, rs, cs);
    if (info || n <= 0) {
	return info;
    }
    //  X L = inv(U): L (unit lower) to a buffer, zeros below the diagonal
    //  of A, then X := X inv(L), i.e. L^T X^T = X^T
    Buffer<T> L(std::size_t(n) * std::size_t(n));
    for (IndexType j = 0; j < n; ++j) {
	for (IndexType i = 0; i < n; ++i) {
	    L[i + j * n] = (i > j) ? A[i * rs + j * cs] : T(i == j ? 1 : 0);
	    if (i > j) {
		A[i * rs + j * cs] = T(0);
	    }
	}
    }
    typedef tasks::SolveStep<IndexType, T> Step;
    Step st;
    st.kind = Step::Solve;
    st.piv = 0;
    st.upper = true;
    st.conj = false;
    st.unitDiag = true;
    st.A = L.data();
    st.incRowA = n;
    st.incColA = 1;
    tasks::solve(n, n, &st, 1, A, cs, rs);
    //  column interchanges: X P
    for (IndexType j = n - 2; j >= 0; --j) {
	const IndexType jp = IndexType(piv[j * incPiv]);
	if (jp != j) {
	    for (IndexType i = 0; i < n; ++i) {
		std::swap(A[i * rs + j * cs], A[i * rs + jp * cs]);
	    }
	}
    }
    return 0;
}

} // namespace ulmBLAS

#endif // ULMBLAS_LAPACK_GETRI_TCC
