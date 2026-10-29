#ifndef ULMBLAS_LEVEL3_GEMM_TCC
#define ULMBLAS_LEVEL3_GEMM_TCC 1

#include <type_traits>
#include <ulmblas/auxiliary/memorypool.h>
#include <ulmblas/config/blocksize.h>
#include <ulmblas/level1extensions/gescal.h>
#include <ulmblas/level3/gemmdirect.h>
#include <ulmblas/level3/mkernel/mgemm.h>
#include <ulmblas/level3/pack/gepack.h>
#include <ulmblas/level3/ukernel/ugemm.h>

//
//  With the OpenBLAS-derived kernel (ULMBLAS_BLIS_DGEMM_PANEL, see
//  ulmblas/level3/ukernel/blis/ugemm.h) DGEMM packs B interleaved with the
//  computation for the first block of A, as OpenBLAS's driver does: each
//  panel of NR columns of B is packed and immediately used while it is in
//  cache.  The packed layout is the same; -DULMBLAS_DGEMM_NO_INTERLEAVE
//  packs the whole block of B first, as for all other kernels.  Kernels
//  with ULMBLAS_BLIS_DGEMM_INTERLEAVE_PANELS (select.h) take that many
//  panels at a time (OpenBLAS's HASWELL kernel: 3, i.e. 24 columns, as
//  OpenBLAS's driver).
//
#if defined(ULMBLAS_BLIS_DGEMM_PANEL) && !defined(ULMBLAS_DGEMM_NO_INTERLEAVE)
#define ULMBLAS_DGEMM_INTERLEAVE 1
#else
#define ULMBLAS_DGEMM_INTERLEAVE 0
#endif
#if defined(ULMBLAS_BLIS_DGEMM_INTERLEAVE_PANELS)
#define ULMBLAS_DGEMM_INTERLEAVE_PANELS ULMBLAS_BLIS_DGEMM_INTERLEAVE_PANELS
#else
#define ULMBLAS_DGEMM_INTERLEAVE_PANELS 1
#endif

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename TA, typename TB,
          typename Beta, typename TC>
void
gemm(IndexType m, IndexType n, IndexType k, const Alpha &alpha, bool conjA,
     const TA *A, IndexType incRowA, IndexType incColA, bool conjB, const TB *B,
     IndexType incRowB, IndexType incColB, const Beta &beta, TC *C,
     IndexType incRowC, IndexType incColC)
{
    typedef decltype(Alpha(0) * TA(0) * TB(0)) T;

    constexpr IndexType MC = BlockSize<T>::MC;
    constexpr IndexType NC = BlockSize<T>::NC;
    constexpr IndexType KC = BlockSize<T>::KC;

    constexpr IndexType MR = BlockSizeUGemm<T>::MR;
    constexpr IndexType NR = BlockSizeUGemm<T>::NR;

    const IndexType mb = (m + MC - 1) / MC;
    const IndexType nb = (n + NC - 1) / NC;
    const IndexType kb = (k + KC - 1) / KC;

    const IndexType mc_ = m % MC;
    const IndexType nc_ = n % NC;
    const IndexType kc_ = k % KC;

    static MemoryPool<T> memoryPool;

    if (alpha == Alpha(0) || k == 0) {
	gescal(m, n, beta, C, incRowC, incColC);
	return;
    }
    //  Kernels that prefer row-major C: for column-major C compute
    //  C^T = B^T A^T (C^T is row major), as BLIS does
    if (UGemmRowPreference<T>::value && incRowC == IndexType(1) &&
        incColC != IndexType(1)) {
	gemm(n, m, k, alpha, conjB, B, incColB, incRowB, conjA, A, incColA,
	     incRowA, beta, C, incColC, incRowC);
	return;
    }
    if (!conjA && !conjB &&
        gemm_direct(m, n, k, alpha, A, incRowA, incColA, B, incRowB, incColB,
                    beta, C, incRowC, incColC)) {
	return;
    }

    T *A_ = memoryPool.allocate(MC * KC + MR);
    T *B_ = memoryPool.allocate(KC * NC + NR);

    constexpr bool interleave =
        ULMBLAS_DGEMM_INTERLEAVE && std::is_same<T, double>::value;

    for (IndexType j = 0; j < nb; ++j) {
	IndexType nc = (j != nb - 1 || nc_ == 0) ? NC : nc_;

	for (IndexType l = 0; l < kb; ++l) {
	    IndexType kc = (l != kb - 1 || kc_ == 0) ? KC : kc_;
	    Beta beta_ = (l == 0) ? beta : Beta(1);

	    if (!interleave) {
		gepack_B(kc, nc, conjB, &B[l * KC * incRowB + j * NC * incColB],
		         incRowB, incColB, B_);
	    }

	    for (IndexType i = 0; i < mb; ++i) {
		IndexType mc = (i != mb - 1 || mc_ == 0) ? MC : mc_;

		gepack_A(mc, kc, conjA, &A[i * MC * incRowA + l * KC * incColA],
		         incRowA, incColA, A_);

		if (interleave && i == 0) {
		    //  pack B panel by panel (or a few panels at a time),
		    //  each used right away
		    constexpr IndexType NI =
		        NR * ULMBLAS_DGEMM_INTERLEAVE_PANELS;
		    for (IndexType jr = 0; jr < nc; jr += NI) {
			IndexType nr = (nc - jr < NI) ? nc - jr : NI;

			gepack_B(kc, nr, conjB,
			         &B[l * KC * incRowB + (j * NC + jr) * incColB],
			         incRowB, incColB, &B_[jr * kc]);
			mgemm(mc, nr, kc, alpha, A_, &B_[jr * kc], beta_,
			      &C[(j * NC + jr) * incColC], incRowC, incColC);
		    }
		} else {
		    mgemm(mc, nc, kc, alpha, A_, B_, beta_,
		          &C[i * MC * incRowC + j * NC * incColC], incRowC,
		          incColC);
		}
	    }
	}
    }

    memoryPool.release(A_);
    memoryPool.release(B_);
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_GEMM_TCC
