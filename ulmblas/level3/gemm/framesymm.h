#ifndef ULMBLAS_LEVEL3_GEMM_FRAMESYMM_H
#define ULMBLAS_LEVEL3_GEMM_FRAMESYMM_H 1

#include <algorithm>
#include <complex>
#include <cstddef>
#include <ulmblas/auxiliary/conjugate.h>
#include <ulmblas/level3/gemm/frame.h>
#include <ulmblas/level3/gemm/pack.h>
#include <ulmblas/level3/gemm/plan.h>
#include <ulmblas/level3/gemm/threads.h>

//
//  The frame algorithm of symm and hemm: C <- beta C + alpha S B (S on the
//  left, m x m) or alpha B S (on the right, n x n) for a symmetric
//  (Hermitian, Herm) matrix S of which only the lower (upper) triangle is
//  stored, with the backend, block sizes and macro kernel of a gemm frame
//  Frame.
//
//  As OpenBLAS (driver/level3/symm_k.c: the gemm driver with other copy
//  routines) and BLIS (the gemm control tree, packm for structured
//  matrices): the loops of gemm (GemmFrame::runLoops, also on one thread),
//  only the packing of S differs (SymmRect).  A panel of a block of S
//
//      - in the stored triangle is packed by the packing of the backend,
//      - in the other triangle as well, with swapped strides (S(r, c) =
//        S(c, r), conjugated for hemm),
//      - that crosses the diagonal element by element (hemm: the real part
//        of the diagonal; its imaginary part is not referenced).
//
//  Kernels that prefer row-major C (column-major C for the others) get the
//  transposed problem C^T = B^T S^T, S on the other side; S^T is S with
//  swapped strides and the other triangle stored (hemm: S^T = conj(S) is
//  Hermitian, exactly the transposed view, no conjugation).
//
namespace ulmBLAS {

//
//  Element (r, c) of S from its stored triangle.
//
template <bool Herm, typename T, typename IndexType, typename TS>
T
symmElement(bool lower, const TS *S, IndexType incRowS, IndexType incColS,
            IndexType r, IndexType c)
{
    if (r == c) {
	return Herm ? T(std::real(S[r * (incRowS + incColS)]))
	            : T(S[r * (incRowS + incColS)]);
    }
    if (lower ? r > c : r < c) {
	return T(S[r * incRowS + c * incColS]);
    }
    return Herm ? T(conjugate(S[c * incRowS + r * incColS]))
                : T(S[c * incRowS + r * incColS]);
}

//
//  The part of C for runLoops: all of C (GemmRect), the packing of the
//  symmetric operand (A if left, otherwise B) from its stored triangle.
//
template <typename Macro, typename IndexType, bool Herm> struct SymmRect
{
    GemmRect<Macro, IndexType> rect;
    bool left, lower;

    void
    columns(std::ptrdiff_t jc, IndexType &j0, IndexType &j1) const
    {
	rect.columns(jc, j0, j1);
    }

    void
    rows(std::ptrdiff_t u, IndexType j, IndexType nc, IndexType &i0,
         IndexType &i1) const
    {
	rect.rows(u, j, nc, i0, i1);
    }

    IndexType
    maxRows() const
    {
	return rect.maxRows();
    }

    template <typename T, typename Beta, typename TC>
    void
    macro(IndexType i, IndexType j, IndexType mc, IndexType nc, IndexType kc,
          const T &alpha, const T *A_, const T *B_, const Beta &beta, TC *C,
          IndexType incRowC, IndexType incColC) const
    {
	rect.macro(i, j, mc, nc, kc, alpha, A_, B_, beta, C, incRowC, incColC);
    }

    //  rows i, ..., i + mc - 1, columns l, ..., l + kc - 1 of A: panels of
    //  MR rows; a panel is in the stored triangle, in the other one or
    //  crosses the diagonal (class 0, 1, 2); consecutive panels of class 0
    //  and 1 with one call of PackA
    template <typename Backend, typename TA>
    void
    packA(IndexType i, IndexType l, IndexType mc, IndexType kc, bool conj,
          const TA *A, IndexType incRowA, IndexType incColA,
          typename Backend::T *buffer) const
    {
	typedef typename Backend::T T;
	if (!left) {
	    GemmPack::packA<Backend>(i, l, mc, kc, conj, A, incRowA, incColA,
	                             buffer);
	    return;
	}
	const IndexType MR = Backend::MR;
	const IndexType mp = (mc + MR - 1) / MR;
	auto cls = [&](IndexType p) -> int {
	    const IndexType r0 = i + p * MR;
	    const IndexType r1 = std::min(i + mc, r0 + MR) - 1; // last row
	    //  stored: all r > all c (lower), all r < all c (upper); the
	    //  diagonal element by element (hemm: its real part only)
	    if (lower ? r0 > l + kc - 1 : r1 < l) {
		return 0;
	    }
	    if (lower ? r1 < l : r0 > l + kc - 1) {
		return 1;
	    }
	    return 2;
	};
	for (IndexType p0 = 0; p0 < mp;) {
	    const int c0 = cls(p0);
	    IndexType p1 = p0 + 1;
	    while (c0 != 2 && p1 < mp && cls(p1) == c0) {
		++p1;
	    }
	    const IndexType r0 = i + p0 * MR;
	    const IndexType rows = std::min(i + mc, i + p1 * MR) - r0;
	    T *buf = buffer + p0 * MR * kc;
	    if (c0 == 0) {
		PackA<Backend>::run(rows, kc, false,
		                    &A[r0 * incRowA + l * incColA], incRowA,
		                    incColA, buf);
	    } else if (c0 == 1) {
		PackA<Backend>::run(rows, kc, Herm,
		                    &A[l * incRowA + r0 * incColA], incColA,
		                    incRowA, buf);
	    } else {
		for (IndexType q = 0; q < kc; ++q) {
		    for (IndexType r = 0; r < MR; ++r) {
			buf[q * MR + r] =
			    r < rows
			        ? symmElement<Herm, T>(lower, A, incRowA,
			                               incColA, r0 + r, l + q)
			        : T(0);
		    }
		}
	    }
	    p0 = p1;
	}
    }

    //  rows l, ..., l + kc - 1, columns j, ..., j + nc - 1 of B: the same
    //  with panels of NR columns
    template <typename Backend, typename TB>
    void
    packB(IndexType l, IndexType j, IndexType kc, IndexType nc, bool conj,
          const TB *B, IndexType incRowB, IndexType incColB,
          typename Backend::T *buffer) const
    {
	typedef typename Backend::T T;
	if (left) {
	    GemmPack::packB<Backend>(l, j, kc, nc, conj, B, incRowB, incColB,
	                             buffer);
	    return;
	}
	const IndexType NR = Backend::NR;
	const IndexType np = (nc + NR - 1) / NR;
	auto cls = [&](IndexType q) -> int {
	    const IndexType c0 = j + q * NR;
	    const IndexType c1 = std::min(j + nc, c0 + NR) - 1; // last column
	    //  stored: all r > all c (lower), all r < all c (upper)
	    if (lower ? l > c1 : l + kc - 1 < c0) {
		return 0;
	    }
	    if (lower ? l + kc - 1 < c0 : l > c1) {
		return 1;
	    }
	    return 2;
	};
	for (IndexType q0 = 0; q0 < np;) {
	    const int k0 = cls(q0);
	    IndexType q1 = q0 + 1;
	    while (k0 != 2 && q1 < np && cls(q1) == k0) {
		++q1;
	    }
	    const IndexType c0 = j + q0 * NR;
	    const IndexType cols = std::min(j + nc, j + q1 * NR) - c0;
	    T *buf = buffer + q0 * kc * NR;
	    if (k0 == 0) {
		PackB<Backend>::run(kc, cols, false,
		                    &B[l * incRowB + c0 * incColB], incRowB,
		                    incColB, buf);
	    } else if (k0 == 1) {
		PackB<Backend>::run(kc, cols, Herm,
		                    &B[c0 * incRowB + l * incColB], incColB,
		                    incRowB, buf);
	    } else {
		for (IndexType r = 0; r < kc; ++r) {
		    for (IndexType c = 0; c < NR; ++c) {
			buf[r * NR + c] =
			    c < cols
			        ? symmElement<Herm, T>(lower, B, incRowB,
			                               incColB, l + r, c0 + c)
			        : T(0);
		    }
		}
	    }
	    q0 = q1;
	}
    }
};

template <typename Frame, bool Herm> struct SymmFrame
{
    typedef typename Frame::BackendType Backend;
    typedef typename Frame::MacroType Macro;
    typedef typename Backend::T T;

    //  C <- beta C + alpha S B (left) or alpha B S (right), C m x n; S
    //  stored in its lower (upper) triangle
    template <typename IndexType, typename Alpha, typename TA, typename TB,
              typename Beta, typename TC>
    static void
    run(bool left, bool lower, IndexType m, IndexType n, const Alpha &alpha,
        const TA *S, IndexType incRowS, IndexType incColS, const TB *B,
        IndexType incRowB, IndexType incColB, const Beta &beta, TC *C,
        IndexType incRowC, IndexType incColC);
};

template <typename Frame, bool Herm>
template <typename IndexType, typename Alpha, typename TA, typename TB,
          typename Beta, typename TC>
void
SymmFrame<Frame, Herm>::run(bool left, bool lower, IndexType m, IndexType n,
                            const Alpha &alpha, const TA *S, IndexType incRowS,
                            IndexType incColS, const TB *B, IndexType incRowB,
                            IndexType incColB, const Beta &beta, TC *C,
                            IndexType incRowC, IndexType incColC)
{
    //  the orientation the kernels prefer: C^T = B^T S^T otherwise
    const bool rowMajorC = incColC == IndexType(1) && incRowC != IndexType(1);
    const bool colMajorC = incRowC == IndexType(1) && incColC != IndexType(1);
    if (Backend::rowPreference ? colMajorC : rowMajorC) {
	run(!left, !lower, n, m, alpha, S, incColS, incRowS, B, incColB,
	    incRowB, beta, C, incColC, incRowC);
	return;
    }
    //  the product as gemm: A (m x k) times B (k x n), S is A or B
    const IndexType k = left ? m : n;

    std::ptrdiff_t nt = GemmThreads::forProblem<T>(m, n, k);
    std::ptrdiff_t jcWays = 1, P = 1;
    if (nt > 1) {
	Frame::partitionThreads(nt, std::ptrdiff_t(m), std::ptrdiff_t(n),
	                        jcWays, P);
    }
    const GemmRect<Macro, IndexType> rect = {
        m, n, IndexType(Backend::MR), IndexType(Backend::NR), jcWays, P};
    const SymmRect<Macro, IndexType, Herm> shape = {rect, left, lower};
    //  on one thread (also when the pool runs fewer threads than planned):
    //  the loops of runLoops with one group of one thread
    const GemmRect<Macro, IndexType> rect1 = {
        m, n, IndexType(Backend::MR), IndexType(Backend::NR), 1, 1};
    const SymmRect<Macro, IndexType, Herm> shape1 = {rect1, left, lower};
    auto none = [] {};
    auto serial = [&] {
	if (left) {
	    Frame::runLoops(1, 1, shape1, none, k, alpha, false, S, incRowS,
	                    incColS, false, B, incRowB, incColB, beta, C,
	                    incRowC, incColC);
	} else {
	    Frame::runLoops(1, 1, shape1, none, k, alpha, false, B, incRowB,
	                    incColB, false, S, incRowS, incColS, beta, C,
	                    incRowC, incColC);
	}
    };
    if (left) {
	Frame::runLoops(jcWays, P, shape, serial, k, alpha, false, S, incRowS,
	                incColS, false, B, incRowB, incColB, beta, C, incRowC,
	                incColC);
    } else {
	Frame::runLoops(jcWays, P, shape, serial, k, alpha, false, B, incRowB,
	                incColB, false, S, incRowS, incColS, beta, C, incRowC,
	                incColC);
    }
}

//
//  symm (hemm, Herm) in the selection tables (gemm/plan.h), looked up as an
//  m x n x k product (k = m on the left, n on the right).
//
template <typename IndexType, typename Alpha, typename TA, typename TB,
          typename Beta, typename TC>
struct SymmPlan
{
    typedef void (*Run)(bool left, bool lower, IndexType m, IndexType n,
                        const Alpha &alpha, const TA *S, IndexType incRowS,
                        IndexType incColS, const TB *B, IndexType incRowB,
                        IndexType incColB, const Beta &beta, TC *C,
                        IndexType incRowC, IndexType incColC);

    Run run;
    const char *name;
};

template <bool Herm, typename IndexType, typename Alpha, typename TA,
          typename TB, typename Beta, typename TC>
struct SymmOp
{
    typedef SymmPlan<IndexType, Alpha, TA, TB, Beta, TC> Plan;

    static constexpr std::size_t sizeC = sizeof(TC);
    static constexpr std::size_t sizeA = sizeof(TA);

    template <typename F>
    static constexpr Plan
    plan(const char *name)
    {
	typedef SymmFrame<typename GemmFrameOf<F>::type, Herm> Frame;
	return Plan{&Frame::template run<IndexType, Alpha, TA, TB, Beta, TC>,
	            name};
    }
};

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_GEMM_FRAMESYMM_H
