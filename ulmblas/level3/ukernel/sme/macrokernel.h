#ifndef ULMBLAS_LEVEL3_UKERNEL_SME_MACROKERNEL_H
#define ULMBLAS_LEVEL3_UKERNEL_SME_MACROKERNEL_H 1

#include <cstddef>
#include <ulmblas/level3/gemm/macrokernel.h>
#include <ulmblas/level3/ukernel/sme/fpsr.h>
#include <ulmblas/config/inline.h>
#include <ulmblas/level3/ukernel/sme/ugemm.h>

//
//  Macro kernel for double with the SME2 micro kernel (32 x 16 tiles, see
//  sme/ugemm.tcc) in one streaming section per block, instead of one per
//  tile as with the portable macro kernel (ulmblas/level3/gemm/macrokernel.h,
//  through ugemm_32x16).  The packed blocks and the order of the tiles are
//  the same:
//
//      C_tile <- beta C_tile + alpha A_panel(i) B_panel(j)
//
//  for the tiles (i, j) of the mc x nc block, column of tiles by column of
//  tiles.
//
//  PrefetchC: before a tile is computed, the tile below it is prefetched
//  (as ugemm_32x16 does in ordinary mode).  In streaming mode PRFM is
//  expensive (48 per tile, about 45 to 85 ns); it pays off only if C comes
//  from memory and kc is not small, otherwise it costs.  gemmSelect
//  (ulmblas/level3/gemm/select.h) chooses the variant for the problem.
//
//  C with unit row stride only; for other C and other types the portable
//  macro kernel is used.
//
namespace ulmBLAS {
namespace sme {

//
//  The tile kernels in streaming mode with ZA enabled, per type: double 32 x
//  16 (ugemm_32x16_za), float 32 x 32 (sgemm_32x32_za).
//
struct TileZa
{
    static constexpr std::ptrdiff_t MR = 32;

    static ULMBLAS_ALWAYS_INLINE void
    run(std::ptrdiff_t kc, double alpha, const double *A_, const double *B_,
        double beta, double *C, std::ptrdiff_t ldC, std::ptrdiff_t mr,
        std::ptrdiff_t nr) __arm_streaming __arm_inout("za")
    {
	ugemm_32x16_za(kc, alpha, A_, B_, beta, C, ldC, mr, nr);
    }
    static ULMBLAS_ALWAYS_INLINE void
    run(std::ptrdiff_t kc, float alpha, const float *A_, const float *B_,
        float beta, float *C, std::ptrdiff_t ldC, std::ptrdiff_t mr,
        std::ptrdiff_t nr) __arm_streaming __arm_inout("za")
    {
	sgemm_32x32_za(kc, alpha, A_, B_, beta, C, ldC, mr, nr);
    }
};

//  NR of the tile kernel for T
template <typename T> struct TileNr
{
    static constexpr std::ptrdiff_t value = sizeof(T) == 8 ? 16 : 32;
};

//
//  The mc x nc block of C in streaming mode, one tile after the other.
//  PrefetchC: the 32 rows below the tile (the next tile of the column of
//  tiles) in all its columns.
//
template <bool PrefetchC, typename T>
__arm_new("za") __arm_locally_streaming void macroBlockStreaming(
    std::ptrdiff_t mc, std::ptrdiff_t nc, std::ptrdiff_t kc, T alpha,
    const T *A_, const T *B_, T beta, T *C, std::ptrdiff_t ldC)
{
    constexpr std::ptrdiff_t MR = TileZa::MR;
    constexpr std::ptrdiff_t NR = TileNr<T>::value;
    //  the 32 rows below in cache lines of 128 bytes
    constexpr std::ptrdiff_t L = 128 / sizeof(T);

    for (std::ptrdiff_t j = 0; j < nc; j += NR) {
	const std::ptrdiff_t nr = (nc - j < NR) ? nc - j : NR;
	for (std::ptrdiff_t i = 0; i < mc; i += MR) {
	    const std::ptrdiff_t mr = (mc - i < MR) ? mc - i : MR;
	    T *Cij = C + i + j * ldC;
	    if (PrefetchC) {
		for (std::ptrdiff_t c = 0; c < NR; ++c) {
		    for (std::ptrdiff_t r = MR; r < 2 * MR; r += L) {
			__builtin_prefetch(Cij + c * ldC + r, 1, 2);
		    }
		    __builtin_prefetch(Cij + c * ldC + 2 * MR - 1, 1, 2);
		}
	    }
	    TileZa::run(kc, alpha, A_ + i * kc, B_ + j * kc, beta, Cij, ldC, mr,
	                nr);
	}
    }
}

//
//  The block with macroBlockStreaming if possible (double, unit row stride
//  of C); false otherwise, then the caller uses the portable macro kernel.
//
template <bool PrefetchC, typename IndexType, typename T, typename Beta,
          typename TC>
inline bool
macroBlock(IndexType mc, IndexType nc, IndexType kc, const T &alpha,
           const T *A_, const T *B_, const Beta &beta, TC *C, IndexType incRowC,
           IndexType incColC)
{
    return false;
}

template <bool PrefetchC, typename IndexType>
inline bool
macroBlock(IndexType mc, IndexType nc, IndexType kc, const double &alpha,
           const double *A_, const double *B_, const double &beta, double *C,
           IndexType incRowC, IndexType incColC)
{
    if (incRowC != IndexType(1)) {
	return false;
    }
    const FpsrRestore fpsr; // see sme/fpsr.h
    macroBlockStreaming<PrefetchC>(mc, nc, kc, alpha, A_, B_, beta, C, incColC);
    return true;
}

template <bool PrefetchC, typename IndexType>
inline bool
macroBlock(IndexType mc, IndexType nc, IndexType kc, const float &alpha,
           const float *A_, const float *B_, const float &beta, float *C,
           IndexType incRowC, IndexType incColC)
{
    if (incRowC != IndexType(1)) {
	return false;
    }
    const FpsrRestore fpsr; // see sme/fpsr.h
    macroBlockStreaming<PrefetchC>(mc, nc, kc, alpha, A_, B_, beta, C, incColC);
    return true;
}

template <typename Backend, bool PrefetchC> struct MacroKernel
{
    typedef typename Backend::T T;

    template <typename IndexType, typename Beta, typename TC>
    static void
    run(IndexType mc, IndexType nc, IndexType kc, const T &alpha, const T *A_,
        const T *B_, const Beta &beta, TC *C, IndexType incRowC,
        IndexType incColC);
};

template <typename Backend, bool PrefetchC>
template <typename IndexType, typename Beta, typename TC>
void
MacroKernel<Backend, PrefetchC>::run(IndexType mc, IndexType nc, IndexType kc,
                                     const typename Backend::T &alpha,
                                     const typename Backend::T *A_,
                                     const typename Backend::T *B_,
                                     const Beta &beta, TC *C, IndexType incRowC,
                                     IndexType incColC)
{
    if (!macroBlock<PrefetchC>(mc, nc, kc, alpha, A_, B_, beta, C, incRowC,
                               incColC)) {
	ulmBLAS::MacroKernel<Backend>::run(mc, nc, kc, alpha, A_, B_, beta, C,
	                                   incRowC, incColC);
    }
}

} // namespace sme
} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_UKERNEL_SME_MACROKERNEL_H
