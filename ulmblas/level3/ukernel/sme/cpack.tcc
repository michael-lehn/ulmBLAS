#ifndef ULMBLAS_LEVEL3_UKERNEL_SME_CPACK_TCC
#define ULMBLAS_LEVEL3_UKERNEL_SME_CPACK_TCC 1

//
//  ulmBLAS: packing of single panels for the SME2 micro kernel for complex
//  float (MR = 32, NR = 16, cgemm.tcc), in streaming mode, as for complex
//  double (zpack.tcc): the packing layout of the portable routines
//  pack_MRxk and pack_kxNR (per k 32 (A) or 16 (B) consecutive complex
//  elements, real and imaginary part interleaved), conjugated if conj is
//  set.
//
//  - Panels whose elements per k are contiguous (A with unit row stride, B
//    with unit column stride): 64 (A) or 32 (B) floats per k, copied with
//    multi-vector loads and stores (LD1W, ST1W of 4 or 2 vectors).
//  - The transposed panels (A with unit column stride, B with unit row
//    stride): ZA as transpose buffer with 64-bit elements, one complex
//    float per element (the tiles ZA0.D ... ZA7.D have 8 x 8 elements at
//    SVL = 512 bits): per 8 values of k, the rows (A) or columns (B) are
//    loaded into the horizontal slices of the tiles 0 to 3 (A) or 0 and 1
//    (B) (LD1D), and the vertical slices are the elements of each k.
//
//  Other strides use the portable routines.  conj negates the imaginary
//  parts (multiplication by 1, -1 in alternating lanes, only of the active
//  elements: the padding stays +0).
//
#include <arm_sme.h>
#include <complex>
#include <cstdint>

namespace ulmBLAS { namespace sme {

//
//  For each k: the m2 <= 16*V floats at X (then X + incX) into 16*V floats
//  of the buffer (V = 2 or 4 vectors), the floats m2 to 16*V-1 zero.
//
template <int V>
__arm_locally_streaming
inline void
cpack_copy(long m2, long k, const float *X, long incX, float *buffer,
           bool conj)
{
    const svcount_t   pc = svptrue_c32();
    const svcount_t   pm = svwhilelt_c32_s64(int64_t(0), int64_t(m2), V);
    const svfloat32_t s  = svdupq_f32(1.0f, -1.0f, 1.0f, -1.0f);
    const svbool_t    p0 = svwhilelt_b32_s64(int64_t(0),  int64_t(m2));
    const svbool_t    p1 = svwhilelt_b32_s64(int64_t(16), int64_t(m2));
    const svbool_t    p2 = svwhilelt_b32_s64(int64_t(32), int64_t(m2));
    const svbool_t    p3 = svwhilelt_b32_s64(int64_t(48), int64_t(m2));

    for (long l=0; l<k; ++l) {
        if (V==4) {
            svfloat32x4_t x = svld1_f32_x4(pm, X);
            if (conj) {
                x = svcreate4(svmul_m(p0, svget4(x, 0), s),
                              svmul_m(p1, svget4(x, 1), s),
                              svmul_m(p2, svget4(x, 2), s),
                              svmul_m(p3, svget4(x, 3), s));
            }
            svst1_f32_x4(pc, buffer, x);
        } else {
            svfloat32x2_t x = svld1_f32_x2(pm, X);
            if (conj) {
                x = svcreate2(svmul_m(p0, svget2(x, 0), s),
                              svmul_m(p1, svget2(x, 1), s));
            }
            svst1_f32_x2(pc, buffer, x);
        }
        buffer += 16*V;
        X      += incX;
    }
}

//
//  The vertical slice r of tile t (64-bit elements) into the buffer: 8
//  complex elements of one k, multiplied by s in the active floats a.
//
template <unsigned t>
inline void
cpack_store_k(long r, float *buffer, svfloat32_t s, svbool_t a)
    __arm_streaming __arm_in("za")
{
    const svbool_t    p = svptrue_b64();
    const svfloat64_t z = svdup_f64(0.0);
    svfloat32_t       x = svreinterpret_f32(svread_ver_za64_f64_m(z, p, t, r));

    svst1_f32(svptrue_b32(), buffer + 16*t, svmul_m(a, x, s));
}

//
//  The rows c < mn <= 8*V of X (row c: k complex elements at X + c*incX,
//  incX in floats) transposed into the buffer (per k 8*V complex elements,
//  the elements mn to 8*V-1 zero; V = 2 or 4 tiles).  Row 8*t+c is loaded
//  into the horizontal slice c of tile t; the slices of the rows mn to
//  8*V-1 stay zero.
//
template <int V>
__arm_new("za") __arm_locally_streaming
inline void
cpack_transpose(long mn, long k, const float *X, long incX, float *buffer,
                bool conj)
{
    const svfloat32_t s = conj ? svdupq_f32(1.0f, -1.0f, 1.0f, -1.0f)
                               : svdup_f32(1.0f);
    //  the floats of the rows < mn in the vectors of a k
    const svbool_t    a0 = svwhilelt_b32_s64(int64_t(0),  int64_t(2*mn));
    const svbool_t    a1 = svwhilelt_b32_s64(int64_t(16), int64_t(2*mn));
    const svbool_t    a2 = svwhilelt_b32_s64(int64_t(32), int64_t(2*mn));
    const svbool_t    a3 = svwhilelt_b32_s64(int64_t(48), int64_t(2*mn));

    svzero_za();
    for (long j=0; j<k; j+=8) {
        //  the 64-bit elements j, ..., min(j+8, k)-1 of the rows; inactive
        //  elements are set to zero
        const long     n  = (k-j<8) ? k-j : 8;
        const svbool_t pk = svwhilelt_b64_s64(int64_t(0), int64_t(n));
        const float    *Xj = X + 2*j;

        for (long c=0; c<8; ++c) {
            if (c   <mn)         svld1_hor_za64(0, c, pk, Xj + (c   )*incX);
            if (c+ 8<mn)         svld1_hor_za64(1, c, pk, Xj + (c+ 8)*incX);
            if (V==4 && c+16<mn) svld1_hor_za64(2, c, pk, Xj + (c+16)*incX);
            if (V==4 && c+24<mn) svld1_hor_za64(3, c, pk, Xj + (c+24)*incX);
        }
        for (long r=0; r<n; ++r) {
            float *b = buffer + 16*V*r;
            cpack_store_k<0>(r, b, s, a0);
            cpack_store_k<1>(r, b, s, a1);
            if (V==4) {
                cpack_store_k<2>(r, b, s, a2);
                cpack_store_k<3>(r, b, s, a3);
            }
        }
        buffer += 16*V*n;
    }
}

} // namespace sme

//
//  P consecutive panels of MR = 32 rows of A, panel by panel.
//
template <int P, typename IndexType>
static void
pack_MRxk(IndexType                     k,
          bool                          conj,
          const std::complex<float>     *A,
          IndexType                     incRowA,
          IndexType                     incColA,
          std::complex<float>           *buffer)
{
    constexpr IndexType MR = sme::BlockSizeUGemm<std::complex<float> >::MR;

    static_assert(MR==32, "SME packing of complex A requires MR = 32");

    if (incRowA!=IndexType(1) && incColA!=IndexType(1)) {
        pack_MRxk<P, IndexType, std::complex<float>, std::complex<float> >(
            k, conj, A, incRowA, incColA, buffer);
        return;
    }
    const float *a = reinterpret_cast<const float *>(A);
    float       *b = reinterpret_cast<float *>(buffer);
    for (IndexType p=0; p<P; ++p) {
        if (incRowA==IndexType(1)) {
            sme::cpack_copy<4>(2*MR, k, a + 2*p*MR, 2*incColA, b + 2*p*k*MR,
                               conj);
        } else {
            sme::cpack_transpose<4>(MR, k, a + 2*p*MR*incRowA, 2*incRowA,
                                    b + 2*p*k*MR, conj);
        }
    }
}

//
//  Panel of NR = 16 columns of B.
//
template <typename IndexType>
static void
pack_kxNR(IndexType                     k,
          bool                          conj,
          const std::complex<float>     *B,
          IndexType                     incRowB,
          IndexType                     incColB,
          std::complex<float>           *buffer)
{
    constexpr IndexType NR = sme::BlockSizeUGemm<std::complex<float> >::NR;

    static_assert(NR==16, "SME packing of complex B requires NR = 16");

    const float *b   = reinterpret_cast<const float *>(B);
    float       *buf = reinterpret_cast<float *>(buffer);
    if (incColB==IndexType(1)) {
        sme::cpack_copy<2>(2*NR, k, b, 2*incRowB, buf, conj);
    } else if (incRowB==IndexType(1)) {
        sme::cpack_transpose<2>(NR, k, b, 2*incColB, buf, conj);
    } else {
        pack_kxNR<IndexType, std::complex<float>, std::complex<float> >(
            k, conj, B, incRowB, incColB, buffer);
    }
}

//
//  The last panel of A with mr < 32 rows (see gepack_A).
//
template <typename IndexType>
static bool
pack_MRxk_partial(IndexType                     mr,
                  IndexType                     k,
                  bool                          conj,
                  const std::complex<float>     *A,
                  IndexType                     incRowA,
                  IndexType                     incColA,
                  std::complex<float>           *buffer)
{
    const float *a = reinterpret_cast<const float *>(A);
    float       *b = reinterpret_cast<float *>(buffer);
    if (incRowA==IndexType(1)) {
        sme::cpack_copy<4>(2*mr, k, a, 2*incColA, b, conj);
        return true;
    }
    if (incColA==IndexType(1)) {
        sme::cpack_transpose<4>(mr, k, a, 2*incRowA, b, conj);
        return true;
    }
    return false;
}

//
//  The last panel of B with nr < 16 columns (see gepack_B).
//
template <typename IndexType>
static bool
pack_kxNR_partial(IndexType                     nr,
                  IndexType                     k,
                  bool                          conj,
                  const std::complex<float>     *B,
                  IndexType                     incRowB,
                  IndexType                     incColB,
                  std::complex<float>           *buffer)
{
    const float *b   = reinterpret_cast<const float *>(B);
    float       *buf = reinterpret_cast<float *>(buffer);
    if (incColB==IndexType(1)) {
        sme::cpack_copy<2>(2*nr, k, b, 2*incRowB, buf, conj);
        return true;
    }
    if (incRowB==IndexType(1)) {
        sme::cpack_transpose<2>(nr, k, b, 2*incColB, buf, conj);
        return true;
    }
    return false;
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_UKERNEL_SME_CPACK_TCC
