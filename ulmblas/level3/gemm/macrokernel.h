#ifndef ULMBLAS_LEVEL3_GEMM_MACROKERNEL_H
#define ULMBLAS_LEVEL3_GEMM_MACROKERNEL_H 1

#include <complex>
#include <ulmblas/level3/ukernel/backend.h>
#include <ulmblas/level3/ukernel/kernelbackend.h>

//
//  The macro kernel of gemm: C <- beta C + alpha A_ B_ for a block of mc x
//  nc elements of C from the packed blocks A_ (mc x kc) and B_ (kc x nc),
//  tile by tile with the micro kernel of the backend
//  (ulmblas/level3/ukernel/backend.h); partial tiles at the end of the
//  block with the partial micro kernel.
//
//  Packed blocks (ulmblas/level3/gemm/pack.h):
//
//      A_   horizontal panels of MR rows and kc columns, one after the
//           other; each panel column major (the MR elements of a column
//           are contiguous).  Panel i starts at A_ + i*MR*kc.
//      B_   vertical panels of kc rows and NR columns, one after the
//           other; each panel row major (the NR elements of a row are
//           contiguous).  Panel j starts at B_ + j*kc*NR.
//
//  The last panels are padded with zeros to MR rows and NR columns.
//
//  Tile (i, j) of C, MR x NR elements, gets one call of the micro kernel:
//
//      C_tile <- beta C_tile + alpha A_panel(i) B_panel(j)
//
//  Partial tiles (mr x nr with mr < MR or nr < NR, at the bottom and the
//  right of the block) use the partial micro kernel: it computes the full
//  MR x NR product and updates only the mr x nr elements of C.
//
namespace ulmBLAS {

//
//  Kernels that compute a whole column of tiles of C per call: panel j of
//  B_ with the panels of A_ gives the column of tiles (0, j), ..., (mp-1, j)
//  of the block.  The OpenBLAS-derived kernels (ULMBLAS_BLIS_DGEMM_PANEL, see
//  ulmblas/level3/ukernel/blis/ugemm.h) can do that: MgemmPanel::run
//  computes the mp full tiles of the column and returns their number.  The
//  default computes nothing and returns 0.
//
template <typename IndexType, typename T, typename Beta, typename TC>
struct MgemmPanel
{
    static IndexType
    run(IndexType mp, IndexType kc, const T &alpha, const T *A, const T *B,
        const Beta &beta, TC *C, IndexType incRowC, IndexType incColC)
    {
	return 0;
    }
};

#if defined(ULMBLAS_BLIS_DGEMM_PANEL)
template <typename IndexType>
struct MgemmPanel<IndexType, double, double, double>
{
    static IndexType
    run(IndexType mp, IndexType kc, const double &alpha, const double *A,
        const double *B, const double &beta, double *C, IndexType incRowC,
        IndexType incColC)
    {
	if (mp == 0 || incRowC != IndexType(1)) {
	    return 0;
	}
	blisk::ugemm_panel(mp, kc, alpha, A, B, beta, C, incColC);
	return mp;
    }
};
#endif

//
//  Kernels that compute all full tiles of the block with one call
//  (ULMBLAS_BLIS_DGEMM_BLOCK, see ulmblas/level3/ukernel/blis/ugemm.h):
//  MgemmBlock::run computes the mp x np full tiles and returns np, the
//  number of columns of tiles it computed.  The default computes
//  nothing and returns 0.
//
template <typename IndexType, typename T, typename Beta, typename TC>
struct MgemmBlock
{
    static IndexType
    run(IndexType mp, IndexType np, IndexType kc, const T &alpha, const T *A,
        const T *B, const Beta &beta, TC *C, IndexType incRowC,
        IndexType incColC)
    {
	return 0;
    }
};

#if defined(ULMBLAS_BLIS_DGEMM_BLOCK)
template <typename IndexType>
struct MgemmBlock<IndexType, double, double, double>
{
    static IndexType
    run(IndexType mp, IndexType np, IndexType kc, const double &alpha,
        const double *A, const double *B, const double &beta, double *C,
        IndexType incRowC, IndexType incColC)
    {
	if (mp == 0 || np == 0 || incRowC != IndexType(1)) {
	    return 0;
	}
	return blisk::ugemm_block(mp, np, kc, alpha, A, B, beta, C, incColC)
	           ? np
		   : 0;
    }
};
#endif

//
//  The same for the vendored kernels of the other types with block calls
//  (blisk::Kernel<T>::blockcall, ulmblas/level3/ukernel/blis/xugemm.h);
//  -DULMBLAS_SGEMM_NO_PANEL, -DULMBLAS_ZGEMM_NO_PANEL switch them off (one
//  call per tile).
//
#if defined(USE_BLIS_XGEMM)
template <typename IndexType, typename T> struct MgemmBlockKernel
{
    static IndexType
    run(IndexType mp, IndexType np, IndexType kc, const T &alpha, const T *A,
        const T *B, const T &beta, T *C, IndexType incRowC, IndexType incColC)
    {
	if (mp == 0 || np == 0 || incRowC != IndexType(1)) {
	    return 0;
	}
	return blisk::xugemm_block(mp, np, kc, alpha, A, B, beta, C, incColC)
	           ? np
		   : 0;
    }
};

#if defined(USE_BLIS_ZGEMM) && ULMBLAS_BLIS_ZGEMM_BLOCKCALL &&                 \
    !defined(ULMBLAS_ZGEMM_NO_PANEL)
template <typename IndexType>
struct MgemmBlock<IndexType, std::complex<double>, std::complex<double>,
                  std::complex<double>>
    : MgemmBlockKernel<IndexType, std::complex<double>>
{
};
#endif

#if defined(USE_BLIS_SGEMM) && ULMBLAS_BLIS_SGEMM_BLOCKCALL &&                 \
    !defined(ULMBLAS_SGEMM_NO_PANEL)
template <typename IndexType>
struct MgemmBlock<IndexType, float, float, float>
    : MgemmBlockKernel<IndexType, float>
{
};
#endif

#if defined(USE_BLIS_CGEMM) && ULMBLAS_BLIS_CGEMM_BLOCKCALL &&                 \
    !defined(ULMBLAS_CGEMM_NO_PANEL)
template <typename IndexType>
struct MgemmBlock<IndexType, std::complex<float>, std::complex<float>,
                  std::complex<float>>
    : MgemmBlockKernel<IndexType, std::complex<float>>
{
};
#endif
#endif // USE_BLIS_XGEMM

//
//  The calls of a backend for more than one tile (OpenBLAS kernels): for
//  the default backend by the type (MgemmBlock, MgemmPanel above), for a
//  backend of a vendored kernel (KernelBackend<K>) by its descriptor.
//
template <typename Backend> struct BackendCalls
{
    template <typename IndexType, typename T, typename Beta, typename TC>
    static IndexType
    block(IndexType mp, IndexType np, IndexType kc, const T &alpha, const T *A,
          const T *B, const Beta &beta, TC *C, IndexType incRowC,
          IndexType incColC)
    {
	return MgemmBlock<IndexType, T, Beta, TC>::run(
	    mp, np, kc, alpha, A, B, beta, C, incRowC, incColC);
    }

    template <typename IndexType, typename T, typename Beta, typename TC>
    static IndexType
    panel(IndexType mp, IndexType kc, const T &alpha, const T *A, const T *B,
          const Beta &beta, TC *C, IndexType incRowC, IndexType incColC)
    {
	return MgemmPanel<IndexType, T, Beta, TC>::run(
	    mp, kc, alpha, A, B, beta, C, incRowC, incColC);
    }
};

template <typename K> struct BackendCalls<KernelBackend<K>>
{
    template <typename IndexType, typename T, typename Beta, typename TC>
    static IndexType
    block(IndexType mp, IndexType np, IndexType kc, const T &alpha, const T *A,
          const T *B, const Beta &beta, TC *C, IndexType incRowC,
          IndexType incColC)
    {
	return KernelBackend<K>::block(mp, np, kc, alpha, A, B, beta, C,
	                               incRowC, incColC);
    }

    template <typename IndexType, typename T, typename Beta, typename TC>
    static IndexType
    panel(IndexType mp, IndexType kc, const T &alpha, const T *A, const T *B,
          const Beta &beta, TC *C, IndexType incRowC, IndexType incColC)
    {
	return KernelBackend<K>::panel(mp, kc, alpha, A, B, beta, C, incRowC,
	                               incColC);
    }
};

template <typename Backend> struct MacroKernel
{
    typedef typename Backend::T T;

    template <typename IndexType, typename Beta, typename TC>
    static void
    run(IndexType mc, IndexType nc, IndexType kc, const T &alpha, const T *A_,
        const T *B_, const Beta &beta, TC *C, IndexType incRowC,
        IndexType incColC);
};

template <typename Backend>
template <typename IndexType, typename Beta, typename TC>
void
MacroKernel<Backend>::run(IndexType mc, IndexType nc, IndexType kc,
                          const typename Backend::T &alpha,
                          const typename Backend::T *A_,
                          const typename Backend::T *B_, const Beta &beta,
                          TC *C, IndexType incRowC, IndexType incColC)
{
    constexpr IndexType MR = Backend::MR;
    constexpr IndexType NR = Backend::NR;

    const IndexType mp = (mc + MR - 1) / MR;
    const IndexType np = (nc + NR - 1) / NR;

    const IndexType mr_ = mc % MR;
    const IndexType nr_ = nc % NR;

    IndexType mr, nr;

    const T *nextA;
    const T *nextB;

    //  all full tiles with one call (OpenBLAS kernels that support it)
    const IndexType jb = BackendCalls<Backend>::block(
        mc / MR, nc / NR, kc, alpha, A_, B_, beta, C, incRowC, incColC);

    for (IndexType j = 0; j < np; ++j) {
	nr = (j != np - 1 || nr_ == 0) ? NR : nr_;
	nextB = &B_[j * kc * NR];

	IndexType i0 = 0;
	if (j < jb) {
	    i0 = mc / MR;
	} else if (nr == NR) {
	    i0 = BackendCalls<Backend>::panel(
	        mc / MR, kc, alpha, A_, &B_[j * kc * NR], beta,
	        &C[j * NR * incColC], incRowC, incColC);
	}

	for (IndexType i = i0; i < mp; ++i) {
	    mr = (i != mp - 1 || mr_ == 0) ? MR : mr_;
	    nextA = &A_[(i + 1) * kc * MR];

	    if (i == mp - 1) {
		nextA = A_;
		nextB = &B_[(j + 1) * kc * NR];
		if (j == np - 1) {
		    nextB = B_;
		}
	    }

	    if (mr == MR && nr == NR) {
		Backend::ugemm(kc, alpha, &A_[i * kc * MR], &B_[j * kc * NR],
		               beta, &C[i * MR * incRowC + j * NR * incColC],
		               incRowC, incColC, nextA, nextB);
	    } else {
		// Call the buffered micro kernel
		Backend::ugemm(mr, nr, kc, alpha, &A_[i * kc * MR],
		               &B_[j * kc * NR], beta,
		               &C[i * MR * incRowC + j * NR * incColC], incRowC,
		               incColC, nextA, nextB);
	    }
	}
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_GEMM_MACROKERNEL_H
