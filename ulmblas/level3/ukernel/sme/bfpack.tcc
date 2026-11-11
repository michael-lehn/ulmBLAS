#ifndef ULMBLAS_LEVEL3_UKERNEL_SME_BFPACK_TCC
#define ULMBLAS_LEVEL3_UKERNEL_SME_BFPACK_TCC 1

//
//  ulmBLAS: packing of single panels of bfloat16 matrices for the SME2
//  micro kernel for float (MR = NR = 32, sgemm.tcc), in streaming mode: the
//  layout of spack.tcc (per k 32 consecutive floats), the elements
//  converted to float.  A bfloat16 is the upper half of a float, so the
//  conversion interleaves zeros (ZIP1, ZIP2 of 16-bit elements): 32
//  bfloat16 in one vector become 32 floats in two.
//
//  - Panels whose 32 elements per k are contiguous (A with unit row stride,
//    B with unit column stride): one load (LD1H) and one store of 2 vectors
//    (ST1W) per k.
//  - The transposed panels: the tile ZA0.H (32 x 32 elements of 16 bits at
//    SVL = 512 bits) as transpose buffer: per 32 values of k the 32 rows
//    (A) or columns (B) are loaded into its horizontal slices, and each
//    vertical slice (the 32 elements of one k) is read, converted and
//    stored.
//
//  Other strides use the portable routines (which convert element by
//  element).
//
#include <arm_sme.h>
#include <cstdint>
#include <ulmblas/auxiliary/bfloat16.h>
#include <ulmblas/level3/ukernel/sme/fpsr.h>

namespace ulmBLAS {
namespace sme {

//  32 bfloat16 (16-bit elements of x) as 2 x 16 floats
inline svfloat32x2_t
bf16_to_f32(svuint16_t x) __arm_streaming
{
    const svuint16_t zero = svdup_u16(0);
    return svcreate2(svreinterpret_f32(svzip1(zero, x)),
                     svreinterpret_f32(svzip2(zero, x)));
}

//
//  For each k: the mn <= 32 bfloat16 at X (then X + incX) as 32 floats in
//  the buffer, the floats mn to 31 zero.
//
__arm_locally_streaming inline void
bfpack_copy(long mn, long k, const std::uint16_t *X, long incX, float *buffer)
{
    const svcount_t pc = svptrue_c32();
    const svbool_t pm = svwhilelt_b16_s64(int64_t(0), int64_t(mn));

    for (long l = 0; l < k; ++l) {
	svst1_f32_x2(pc, buffer, bf16_to_f32(svld1_u16(pm, X)));
	buffer += 32;
	X += incX;
    }
}

//
//  The rows c < mn <= 32 of X (row c: k elements at X + c*incX) transposed
//  into the buffer (per k 32 floats, the elements mn to 31 zero).  Row c is
//  loaded into the horizontal slice c of ZA0.H; the slices of the rows mn
//  to 31 stay zero.  The last k%32 columns with the loads predicated to the
//  columns j to k-1 (inactive elements of a slice are set to zero).
//
__arm_new("za") __arm_locally_streaming inline void bfpack_transpose(
    long mn, long k, const std::uint16_t *X, long incX, float *buffer)
{
    const svbool_t p = svptrue_b16();
    const svcount_t pc = svptrue_c32();
    const svuint16_t zero = svdup_u16(0);

    svzero_za();
    for (long j = 0; j < k; j += 32) {
	const long n = (k - j < 32) ? k - j : 32;
	const svbool_t pk = svwhilelt_b16_s64(int64_t(0), int64_t(n));

	for (long c = 0; c < mn; ++c) {
	    svld1_hor_za16(0, c, pk, X + c * incX + j);
	}
	for (long r = 0; r < n; ++r) {
	    svuint16_t x = svread_ver_za16_u16_m(zero, p, 0, r);
	    svst1_f32_x2(pc, buffer + 32 * r, bf16_to_f32(x));
	}
	buffer += 32 * n;
    }
}

inline const std::uint16_t *
bits(const bfloat16 *x)
{
    return reinterpret_cast<const std::uint16_t *>(x);
}

} // namespace sme

//
//  P consecutive panels of MR = 32 rows of A, panel by panel.
//
template <int P, typename IndexType>
static void
pack_MRxk(IndexType k, bool, const bfloat16 *A, IndexType incRowA,
          IndexType incColA, float *buffer)
{
    const sme::FpsrRestore fpsr; // see sme/fpsr.h

    constexpr IndexType MR = sme::BlockSizeUGemm<float>::MR;

    static_assert(MR == 32, "SME packing of A requires MR = 32");

    if (incRowA != IndexType(1) && incColA != IndexType(1)) {
	pack_MRxk<P, IndexType, bfloat16, float>(k, false, A, incRowA, incColA,
	                                         buffer);
	return;
    }
    for (IndexType p = 0; p < P; ++p) {
	if (incRowA == IndexType(1)) {
	    sme::bfpack_copy(MR, k, sme::bits(A + p * MR), incColA,
	                     buffer + p * k * MR);
	} else {
	    sme::bfpack_transpose(MR, k, sme::bits(A + p * MR * incRowA),
	                          incRowA, buffer + p * k * MR);
	}
    }
}

//
//  Panel of NR = 32 columns of B.
//
template <typename IndexType>
static void
pack_kxNR(IndexType k, bool, const bfloat16 *B, IndexType incRowB,
          IndexType incColB, float *buffer)
{
    const sme::FpsrRestore fpsr; // see sme/fpsr.h

    constexpr IndexType NR = sme::BlockSizeUGemm<float>::NR;

    static_assert(NR == 32, "SME packing of B requires NR = 32");

    if (incColB == IndexType(1)) {
	sme::bfpack_copy(NR, k, sme::bits(B), incRowB, buffer);
    } else if (incRowB == IndexType(1)) {
	sme::bfpack_transpose(NR, k, sme::bits(B), incColB, buffer);
    } else {
	pack_kxNR<IndexType, bfloat16, float>(k, false, B, incRowB, incColB,
	                                      buffer);
    }
}

//
//  The last panel of A with mr < 32 rows (see gepack_A).
//
template <typename IndexType>
static bool
pack_MRxk_partial(IndexType mr, IndexType k, bool, const bfloat16 *A,
                  IndexType incRowA, IndexType incColA, float *buffer)
{
    const sme::FpsrRestore fpsr; // see sme/fpsr.h

    if (incRowA == IndexType(1)) {
	sme::bfpack_copy(mr, k, sme::bits(A), incColA, buffer);
	return true;
    }
    if (incColA == IndexType(1)) {
	sme::bfpack_transpose(mr, k, sme::bits(A), incRowA, buffer);
	return true;
    }
    return false;
}

//
//  The last panel of B with nr < 32 columns (see gepack_B).
//
template <typename IndexType>
static bool
pack_kxNR_partial(IndexType nr, IndexType k, bool, const bfloat16 *B,
                  IndexType incRowB, IndexType incColB, float *buffer)
{
    const sme::FpsrRestore fpsr; // see sme/fpsr.h

    if (incColB == IndexType(1)) {
	sme::bfpack_copy(nr, k, sme::bits(B), incRowB, buffer);
	return true;
    }
    if (incRowB == IndexType(1)) {
	sme::bfpack_transpose(nr, k, sme::bits(B), incColB, buffer);
	return true;
    }
    return false;
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_UKERNEL_SME_BFPACK_TCC
