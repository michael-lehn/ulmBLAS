#ifndef ULMBLAS_LEVEL3_UKERNEL_SME_SPACK_TCC
#define ULMBLAS_LEVEL3_UKERNEL_SME_SPACK_TCC 1

//
//  ulmBLAS: packing of single panels for the SME2 micro kernel for float
//  (MR = NR = 32, sgemm.tcc), in streaming mode, as for double (pack.tcc):
//  the packing layout of the portable routines pack_MRxk and pack_kxNR (per
//  k 32 consecutive elements).
//
//  - Panels whose 32 elements per k are contiguous (A with unit row stride,
//    B with unit column stride): copied with multi-vector loads and stores
//    (LD1W, ST1W of 2 vectors).
//  - The transposed panels (A with unit column stride, B with unit row
//    stride): ZA as transpose buffer (the tiles ZA0.S and ZA1.S have 16 x
//    16 elements at SVL = 512 bits): per 16 values of k, the 32 rows (A) or
//    columns (B) are loaded into the horizontal slices of the two tiles
//    (LD1W), and the vertical slices are the 32 elements of each k (ST1W).
//
//  Other strides use the portable routines.  For float conj has no effect.
//
#include <arm_sme.h>
#include <cstdint>

namespace ulmBLAS { namespace sme {

//
//  For each k: the mn <= 32 floats at X (then X + incX) into 32 floats of
//  the buffer, the floats mn to 31 zero.
//
__arm_locally_streaming
inline void
spack_copy(long mn, long k, const float *X, long incX, float *buffer)
{
    const svcount_t pc = svptrue_c32();
    const svcount_t pm = svwhilelt_c32_s64(int64_t(0), int64_t(mn), 2);

    for (long l=0; l<k; ++l) {
        svst1_f32_x2(pc, buffer, svld1_f32_x2(pm, X));
        buffer += 32;
        X      += incX;
    }
}

//
//  The rows c < mn <= 32 of X (row c: k elements at X + c*incX) transposed
//  into the buffer (per k 32 elements, the elements mn to 31 zero).  Row
//  16*t+c is loaded into the horizontal slice c of tile t; the slices of
//  the rows mn to 31 stay zero.  The last k%16 columns with the loads
//  predicated to the columns j to k-1 (inactive elements of a slice are
//  set to zero, see pack_A_rowmajor).
//
__arm_new("za") __arm_locally_streaming
inline void
spack_transpose(long mn, long k, const float *X, long incX, float *buffer)
{
    const svbool_t p = svptrue_b32();

    svzero_za();
    for (long j=0; j<k; j+=16) {
        const long     n  = (k-j<16) ? k-j : 16;
        const svbool_t pk = svwhilelt_b32_s64(int64_t(0), int64_t(n));

        for (long c=0; c<16; ++c) {
            if (c   <mn) svld1_hor_za32(0, c, pk, X + (c   )*incX + j);
            if (c+16<mn) svld1_hor_za32(1, c, pk, X + (c+16)*incX + j);
        }
        for (long r=0; r<n; ++r) {
            svst1_ver_za32(0, r, p, buffer + 32*r);
            svst1_ver_za32(1, r, p, buffer + 32*r + 16);
        }
        buffer += 32*n;
    }
}

} // namespace sme

//
//  P consecutive panels of MR = 32 rows of A, panel by panel.
//
template <int P, typename IndexType>
static void
pack_MRxk(IndexType      k,
          bool           ,
          const float    *A,
          IndexType      incRowA,
          IndexType      incColA,
          float          *buffer)
{
    constexpr IndexType MR = sme::BlockSizeUGemm<float>::MR;

    static_assert(MR==32, "SME packing of A requires MR = 32");

    if (incRowA!=IndexType(1) && incColA!=IndexType(1)) {
        pack_MRxk<P, IndexType, float, float>(k, false, A, incRowA, incColA,
                                              buffer);
        return;
    }
    for (IndexType p=0; p<P; ++p) {
        if (incRowA==IndexType(1)) {
            sme::spack_copy(MR, k, A + p*MR, incColA, buffer + p*k*MR);
        } else {
            sme::spack_transpose(MR, k, A + p*MR*incRowA, incRowA,
                                 buffer + p*k*MR);
        }
    }
}

//
//  Panel of NR = 32 columns of B.
//
template <typename IndexType>
static void
pack_kxNR(IndexType      k,
          bool           ,
          const float    *B,
          IndexType      incRowB,
          IndexType      incColB,
          float          *buffer)
{
    constexpr IndexType NR = sme::BlockSizeUGemm<float>::NR;

    static_assert(NR==32, "SME packing of B requires NR = 32");

    if (incColB==IndexType(1)) {
        sme::spack_copy(NR, k, B, incRowB, buffer);
    } else if (incRowB==IndexType(1)) {
        sme::spack_transpose(NR, k, B, incColB, buffer);
    } else {
        pack_kxNR<IndexType, float, float>(k, false, B, incRowB, incColB,
                                           buffer);
    }
}

//
//  The last panel of A with mr < 32 rows (see gepack_A).
//
template <typename IndexType>
static bool
pack_MRxk_partial(IndexType      mr,
                  IndexType      k,
                  bool           ,
                  const float    *A,
                  IndexType      incRowA,
                  IndexType      incColA,
                  float          *buffer)
{
    if (incRowA==IndexType(1)) {
        sme::spack_copy(mr, k, A, incColA, buffer);
        return true;
    }
    if (incColA==IndexType(1)) {
        sme::spack_transpose(mr, k, A, incRowA, buffer);
        return true;
    }
    return false;
}

//
//  The last panel of B with nr < 32 columns (see gepack_B).
//
template <typename IndexType>
static bool
pack_kxNR_partial(IndexType      nr,
                  IndexType      k,
                  bool           ,
                  const float    *B,
                  IndexType      incRowB,
                  IndexType      incColB,
                  float          *buffer)
{
    if (incColB==IndexType(1)) {
        sme::spack_copy(nr, k, B, incRowB, buffer);
        return true;
    }
    if (incRowB==IndexType(1)) {
        sme::spack_transpose(nr, k, B, incColB, buffer);
        return true;
    }
    return false;
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_UKERNEL_SME_SPACK_TCC
