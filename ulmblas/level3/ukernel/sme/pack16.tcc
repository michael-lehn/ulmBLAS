#ifndef ULMBLAS_LEVEL3_UKERNEL_SME_PACK16_TCC
#define ULMBLAS_LEVEL3_UKERNEL_SME_PACK16_TCC 1

//
//  ulmBLAS: packing of single panels of bfloat16 and float16 matrices for
//  the SME2 micro kernel for float (MR = NR = 32, sgemm.tcc), in streaming
//  mode: the layout of spack.tcc (per k 32 consecutive floats), the
//  elements converted to float.  32 elements of 16 bits in one vector
//  become 32 floats in two:
//
//  - bfloat16 is the upper half of a float: zeros are interleaved (ZIP1,
//    ZIP2 of 16-bit elements).
//  - float16: each element is duplicated (ZIP1, ZIP2 with itself), so that
//    it is the lower half of a 32-bit element, and converted (FCVT).
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
#include <ulmblas/auxiliary/float16.h>
#include <ulmblas/level3/ukernel/sme/fpsr.h>

namespace ulmBLAS {
namespace sme {

//  32 elements of 16 bits (x) as 2 x 16 floats
inline svfloat32x2_t
to_f32(svuint16_t x, const bfloat16 *) __arm_streaming
{
    const svuint16_t zero = svdup_u16(0);
    return svcreate2(svreinterpret_f32(svzip1(zero, x)),
                     svreinterpret_f32(svzip2(zero, x)));
}

inline svfloat32x2_t
to_f32(svuint16_t x, const float16 *) __arm_streaming
{
    const svbool_t p = svptrue_b32();
    const svfloat16_t h = svreinterpret_f16(x);
    return svcreate2(svcvt_f32_f16_x(p, svzip1(h, h)),
                     svcvt_f32_f16_x(p, svzip2(h, h)));
}

//
//  For each k: the mn <= 32 elements at X (then X + incX) as 32 floats in
//  the buffer, the floats mn to 31 zero.
//
template <typename T16>
__arm_locally_streaming inline void
pack16_copy(long mn, long k, const T16 *X, long incX, float *buffer)
{
    const svcount_t pc = svptrue_c32();
    const svbool_t pm = svwhilelt_b16_s64(int64_t(0), int64_t(mn));

    for (long l = 0; l < k; ++l) {
	const svuint16_t x =
	    svld1_u16(pm, reinterpret_cast<const std::uint16_t *>(X));
	svst1_f32_x2(pc, buffer, to_f32(x, X));
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
template <typename T16>
__arm_new("za") __arm_locally_streaming inline void pack16_transpose(
    long mn, long k, const T16 *X, long incX, float *buffer)
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
	    svst1_f32_x2(pc, buffer + 32 * r, to_f32(x, X));
	}
	buffer += 32 * n;
    }
}

//
//  P consecutive panels of MR = 32 rows of A, panel by panel (unit row or
//  column stride).
//
template <int P, typename IndexType, typename T16>
void
pack16_MRxk(IndexType k, const T16 *A, IndexType incRowA, IndexType incColA,
            float *buffer)
{
    const FpsrRestore fpsr; // see sme/fpsr.h

    constexpr IndexType MR = BlockSizeUGemm<float>::MR;

    static_assert(MR == 32, "SME packing of A requires MR = 32");

    for (IndexType p = 0; p < P; ++p) {
	if (incRowA == IndexType(1)) {
	    pack16_copy(MR, k, A + p * MR, incColA, buffer + p * k * MR);
	} else {
	    pack16_transpose(MR, k, A + p * MR * incRowA, incRowA,
	                     buffer + p * k * MR);
	}
    }
}

//
//  Panel of NR = 32 columns of B (unit row or column stride).
//
template <typename IndexType, typename T16>
void
pack16_kxNR(IndexType k, const T16 *B, IndexType incRowB, IndexType incColB,
            float *buffer)
{
    const FpsrRestore fpsr; // see sme/fpsr.h

    constexpr IndexType NR = BlockSizeUGemm<float>::NR;

    static_assert(NR == 32, "SME packing of B requires NR = 32");

    if (incColB == IndexType(1)) {
	pack16_copy(NR, k, B, incRowB, buffer);
    } else {
	pack16_transpose(NR, k, B, incColB, buffer);
    }
}

//
//  Partial panel: mn < 32 rows of A or columns of B (incMN: stride between
//  them, incK: stride in k); false for strides that are not supported here
//  (the portable routine packs them).
//
template <typename IndexType, typename T16>
bool
pack16_partial(IndexType mn, IndexType k, const T16 *X, IndexType incMN,
               IndexType incK, float *buffer)
{
    const FpsrRestore fpsr; // see sme/fpsr.h

    if (incMN == IndexType(1)) {
	pack16_copy(mn, k, X, incK, buffer);
	return true;
    }
    if (incK == IndexType(1)) {
	pack16_transpose(mn, k, X, incMN, buffer);
	return true;
    }
    return false;
}

} // namespace sme

//
//  The overloads called by gepack_A and gepack_B (see sme/pack.h).
//
#define ULMBLAS_SME_PACK16(T16)                                                \
    template <int P, typename IndexType>                                       \
    static void pack_MRxk(IndexType k, bool, const T16 *A, IndexType incRowA,  \
                          IndexType incColA, float *buffer)                    \
    {                                                                          \
	if (incRowA != IndexType(1) && incColA != IndexType(1)) {              \
	    pack_MRxk<P, IndexType, T16, float>(k, false, A, incRowA, incColA, \
	                                        buffer);                       \
	    return;                                                            \
	}                                                                      \
	sme::pack16_MRxk<P>(k, A, incRowA, incColA, buffer);                   \
    }                                                                          \
                                                                               \
    template <typename IndexType>                                              \
    static void pack_kxNR(IndexType k, bool, const T16 *B, IndexType incRowB,  \
                          IndexType incColB, float *buffer)                    \
    {                                                                          \
	if (incRowB != IndexType(1) && incColB != IndexType(1)) {              \
	    pack_kxNR<IndexType, T16, float>(k, false, B, incRowB, incColB,    \
	                                     buffer);                          \
	    return;                                                            \
	}                                                                      \
	sme::pack16_kxNR(k, B, incRowB, incColB, buffer);                      \
    }                                                                          \
                                                                               \
    template <typename IndexType>                                              \
    static bool pack_MRxk_partial(IndexType mr, IndexType k, bool,             \
                                  const T16 *A, IndexType incRowA,             \
                                  IndexType incColA, float *buffer)            \
    {                                                                          \
	return sme::pack16_partial(mr, k, A, incRowA, incColA, buffer);        \
    }                                                                          \
                                                                               \
    template <typename IndexType>                                              \
    static bool pack_kxNR_partial(IndexType nr, IndexType k, bool,             \
                                  const T16 *B, IndexType incRowB,             \
                                  IndexType incColB, float *buffer)            \
    {                                                                          \
	return sme::pack16_partial(nr, k, B, incColB, incRowB, buffer);        \
    }

ULMBLAS_SME_PACK16(bfloat16)
ULMBLAS_SME_PACK16(float16)

#undef ULMBLAS_SME_PACK16

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_UKERNEL_SME_PACK16_TCC
