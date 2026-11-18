#ifndef ULMBLAS_LEVEL3_GEMM_MACROKERNELTR_H
#define ULMBLAS_LEVEL3_GEMM_MACROKERNELTR_H 1

#include <cstddef>
#include <ulmblas/level3/gemm/macrokernel.h>

//
//  The macro kernel of gemmt: C <- beta C + alpha A_ B_ for the elements of
//  an mc x nc block of C in the lower (or upper) triangle of the whole
//  matrix; the other elements of C are not accessed.  d is the position of
//  the block: its first row minus its first column in the whole matrix.
//  Element (r, c) of the block is in the lower triangle if r + d >= c, in
//  the upper if r + d <= c.
//
//  Tile by tile, as BLIS's gemmt macro kernel (bli_gemmt_l_ker_var2): tiles
//  in the triangle with the micro kernel, the run of them in a column of
//  tiles with one panel call if the backend has one (see MacroKernel),
//  tiles outside are skipped, a tile that meets the diagonal is computed
//  into a local tile (beta = 0) and added to C in the triangle.  A block
//  wholly in the triangle is computed by Macro, the macro kernel of gemm
//  of the frame (e.g. with block calls, or the SME2 macro kernel).
//
namespace ulmBLAS {

template <typename Backend, typename Macro = MacroKernel<Backend>>
struct MacroKernelTr
{
    typedef typename Backend::T T;

    template <typename IndexType, typename Beta, typename TC>
    static void
    run(bool lower, IndexType d, IndexType mc, IndexType nc, IndexType kc,
        const T &alpha, const T *A_, const T *B_, const Beta &beta, TC *C,
        IndexType incRowC, IndexType incColC);
};

template <typename Backend, typename Macro>
template <typename IndexType, typename Beta, typename TC>
void
MacroKernelTr<Backend, Macro>::run(bool lower, IndexType d, IndexType mc,
                                   IndexType nc, IndexType kc,
                                   const typename Backend::T &alpha,
                                   const typename Backend::T *A_,
                                   const typename Backend::T *B_,
                                   const Beta &beta, TC *C, IndexType incRowC,
                                   IndexType incColC)
{
    constexpr IndexType MR = Backend::MR;
    constexpr IndexType NR = Backend::NR;

    //  the whole block in the triangle, or none of it
    if (lower ? d >= nc - 1 : d + mc - 1 <= 0) {
	Macro::run(mc, nc, kc, alpha, A_, B_, beta, C, incRowC, incColC);
	return;
    }
    if (lower ? d + mc - 1 < 0 : d > nc - 1) {
	return;
    }

    const IndexType mp = (mc + MR - 1) / MR;
    const IndexType np = (nc + NR - 1) / NR;
    const IndexType mFull = mc / MR; // full tiles in a column of tiles

    for (IndexType j = 0; j < np; ++j) {
	const IndexType c0 = j * NR;
	const IndexType nr = (c0 + NR <= nc) ? NR : nc - c0;
	const T *Bj = &B_[j * kc * NR];
	TC *Cj = &C[c0 * incColC];

	//  tiles [i0, i1) of the column in the triangle (whole tiles), the
	//  others meet the diagonal or are outside: lower, row r of the
	//  block is in the triangle for the whole column if r + d >= c0 + nr
	//  - 1, outside if r + d < c0; upper, in it if r + d <= c0, outside
	//  if r + d > c0 + nr - 1
	IndexType i0, i1, iMeet0, iMeet1;
	if (lower) {
	    //  first tile with all rows r >= c0 + nr - 1 - d
	    const IndexType rIn = c0 + nr - 1 - d;
	    i0 = rIn <= 0 ? 0 : (rIn + MR - 1) / MR;
	    i1 = mp;
	    //  tiles before i0 with a row r >= c0 - d meet the diagonal
	    const IndexType rMeet = c0 - d;
	    iMeet0 = rMeet <= 0 ? 0 : rMeet / MR;
	    iMeet1 = i0 < mp ? i0 : mp;
	} else {
	    //  last row r <= c0 - d: tiles [0, i1) with all rows in it
	    const IndexType rIn = c0 - d; // rows r <= rIn
	    i0 = 0;
	    i1 = rIn < 0 ? 0 : (rIn + 1) / MR;
	    if (i1 > mp) {
		i1 = mp;
	    }
	    //  tiles from i1 with a row r <= c0 + nr - 1 - d meet it
	    const IndexType rMeet = c0 + nr - 1 - d;
	    iMeet0 = i1;
	    iMeet1 = rMeet < 0 ? i1 : rMeet / MR + 1;
	    if (iMeet1 > mp) {
		iMeet1 = mp;
	    }
	}

	//  the tiles in the triangle: a panel call for the full ones (lower:
	//  a suffix of the column, upper: a prefix), then tile by tile
	IndexType iPanel = i0;
	if (nr == NR && i1 > i0) {
	    const IndexType full = (i1 < mFull ? i1 : mFull) - i0;
	    if (full > 0) {
		iPanel += BackendCalls<Backend>::panel(
		    full, kc, alpha, &A_[i0 * kc * MR], Bj, beta,
		    &Cj[i0 * MR * incRowC], incRowC, incColC);
	    }
	}
	for (IndexType i = iPanel; i < i1; ++i) {
	    const IndexType mr = (i * MR + MR <= mc) ? MR : mc - i * MR;
	    const T *Ai = &A_[i * kc * MR];
	    TC *Cij = &Cj[i * MR * incRowC];
	    if (mr == MR && nr == NR) {
		Backend::ugemm(kc, alpha, Ai, Bj, beta, Cij, incRowC, incColC,
		               Ai, Bj);
	    } else {
		Backend::ugemm(mr, nr, kc, alpha, Ai, Bj, beta, Cij, incRowC,
		               incColC, Ai, Bj);
	    }
	}

	//  the tiles that meet the diagonal: into a local tile, then the
	//  elements in the triangle
	for (IndexType i = iMeet0; i < iMeet1; ++i) {
	    const IndexType r0 = i * MR;
	    const IndexType mr = (r0 + MR <= mc) ? MR : mc - r0;
	    alignas(64) T C_[MR * NR];
	    Backend::ugemm(kc, alpha, &A_[i * kc * MR], Bj, T(0), C_,
	                   IndexType(1), MR, &A_[i * kc * MR], Bj);
	    TC *Cij = &Cj[r0 * incRowC];
	    for (IndexType c = 0; c < nr; ++c) {
		for (IndexType r = 0; r < mr; ++r) {
		    const IndexType diff = r0 + r + d - (c0 + c);
		    if (lower ? diff < 0 : diff > 0) {
			continue;
		    }
		    TC &x = Cij[r * incRowC + c * incColC];
		    x = beta == Beta(0) ? TC(C_[r + c * MR])
		                        : TC(beta * x + C_[r + c * MR]);
		}
	    }
	}
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_GEMM_MACROKERNELTR_H
