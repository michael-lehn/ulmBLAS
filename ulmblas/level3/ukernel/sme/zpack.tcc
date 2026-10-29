#ifndef ULMBLAS_LEVEL3_UKERNEL_SME_ZPACK_TCC
#define ULMBLAS_LEVEL3_UKERNEL_SME_ZPACK_TCC 1

//
//  ulmBLAS: packing of single panels for the SME2 micro kernel for complex
//  double (MR = NR = 16, zgemm.tcc), in streaming mode, as for double
//  (pack.tcc): the packing layout of the portable routines pack_MRxk and
//  pack_kxNR (per k 16 consecutive complex elements, real and imaginary
//  part interleaved), conjugated if conj is set.
//
//  - Panels whose 16 elements per k are contiguous (A with unit row stride,
//    B with unit column stride): 32 doubles per k, copied with
//    multi-vector loads and stores (LD1D, ST1D of 4 vectors).
//  - The transposed panels (A with unit column stride, B with unit row
//    stride): ZA as transpose buffer with 128-bit elements, one complex
//    double per element (the 16 tiles ZA0.Q ... ZA15.Q have 4 x 4 elements
//    at SVL = 512 bits): per 4 values of k, the 16 rows (A) or columns (B)
//    are loaded into the horizontal slices of the tiles 0 to 3 (LD1Q), and
//    the vertical slices are the 16 elements of each k (MOVA, ST1D).
//
//  Other strides use the portable routines.  conj negates the imaginary
//  parts (multiplication by 1, -1 in alternating lanes).
//
#include <arm_sme.h>
#include <complex>
#include <cstdint>

namespace ulmBLAS { namespace sme {

//
//  For each k: the m2 <= 32 doubles at X (then X + incX) into 32 doubles
//  of the buffer, the doubles m2 to 31 zero.
//
__arm_locally_streaming
inline void
zpack_copy(long m2, long k, const double *X, long incX, double *buffer,
           bool conj)
{
    const svcount_t   pc = svptrue_c64();
    const svcount_t   pm = svwhilelt_c64_s64(int64_t(0), int64_t(m2), 4);
    const svfloat64_t s  = svdupq_f64(1.0, -1.0);
    //  the active doubles of the 4 vectors: the padding (zero from the
    //  predicated loads) is not multiplied, so it stays +0
    const svbool_t    p0 = svwhilelt_b64_s64(int64_t(0),  int64_t(m2));
    const svbool_t    p1 = svwhilelt_b64_s64(int64_t(8),  int64_t(m2));
    const svbool_t    p2 = svwhilelt_b64_s64(int64_t(16), int64_t(m2));
    const svbool_t    p3 = svwhilelt_b64_s64(int64_t(24), int64_t(m2));

    if (!conj) {
        for (long l=0; l<k; ++l) {
            svst1_f64_x4(pc, buffer, svld1_f64_x4(pm, X));
            buffer += 32;
            X      += incX;
        }
        return;
    }
    for (long l=0; l<k; ++l) {
        svfloat64x4_t x = svld1_f64_x4(pm, X);
        svst1_f64_x4(pc, buffer,
                     svcreate4(svmul_m(p0, svget4(x, 0), s),
                               svmul_m(p1, svget4(x, 1), s),
                               svmul_m(p2, svget4(x, 2), s),
                               svmul_m(p3, svget4(x, 3), s)));
        buffer += 32;
        X      += incX;
    }
}

//
//  The vertical slices r of the tiles 0 to 3 (128-bit elements) into the
//  buffer: the 16 complex elements of one k, multiplied by s in the active
//  doubles a0 to a3 of the 4 vectors (the padding stays +0).
//
inline void
zpack_store_k(long r, double *buffer, svfloat64_t s, svbool_t a0,
              svbool_t a1, svbool_t a2, svbool_t a3)
    __arm_streaming __arm_in("za")
{
    const svbool_t    p = svptrue_b64();
    const svfloat64_t z = svdup_f64(0.0);

    svst1_f64(p, buffer,      svmul_m(a0, svread_ver_za128_f64_m(z, p, 0, r), s));
    svst1_f64(p, buffer + 8,  svmul_m(a1, svread_ver_za128_f64_m(z, p, 1, r), s));
    svst1_f64(p, buffer + 16, svmul_m(a2, svread_ver_za128_f64_m(z, p, 2, r), s));
    svst1_f64(p, buffer + 24, svmul_m(a3, svread_ver_za128_f64_m(z, p, 3, r), s));
}

//
//  The rows c < mn <= 16 of X (row c: k complex elements at X + c*incX,
//  incX in doubles) transposed into the buffer (per k 16 complex elements,
//  the elements mn to 15 zero).  Row 4*t+c is loaded into the horizontal
//  slice c of tile t; the slices of the rows mn to 15 stay zero.
//
__arm_new("za") __arm_locally_streaming
inline void
zpack_transpose(long mn, long k, const double *X, long incX, double *buffer,
                bool conj)
{
    const svfloat64_t s = conj ? svdupq_f64(1.0, -1.0) : svdup_f64(1.0);
    //  the doubles of the rows < mn in the 4 vectors of a k
    const svbool_t    a0 = svwhilelt_b64_s64(int64_t(0),  int64_t(2*mn));
    const svbool_t    a1 = svwhilelt_b64_s64(int64_t(8),  int64_t(2*mn));
    const svbool_t    a2 = svwhilelt_b64_s64(int64_t(16), int64_t(2*mn));
    const svbool_t    a3 = svwhilelt_b64_s64(int64_t(24), int64_t(2*mn));

    svzero_za();
    for (long j=0; j<k; j+=4) {
        //  the 128-bit elements j, ..., min(j+4, k)-1 of the rows (one
        //  predicate bit per 128-bit element: the one of its first
        //  doubleword); inactive elements are set to zero
        const long     n  = (k-j<4) ? k-j : 4;
        const svbool_t pk = svwhilelt_b64_s64(int64_t(0), int64_t(2*n));
        const double   *Xj = X + 2*j;

        for (long c=0; c<4; ++c) {
            if (c   <mn) svld1_hor_za128(0, c, pk, Xj + (c   )*incX);
            if (c+ 4<mn) svld1_hor_za128(1, c, pk, Xj + (c+ 4)*incX);
            if (c+ 8<mn) svld1_hor_za128(2, c, pk, Xj + (c+ 8)*incX);
            if (c+12<mn) svld1_hor_za128(3, c, pk, Xj + (c+12)*incX);
        }
        for (long r=0; r<n; ++r) {
            zpack_store_k(r, buffer + 32*r, s, a0, a1, a2, a3);
        }
        buffer += 32*n;
    }
}

} // namespace sme

//
//  P consecutive panels of MR = 16 rows of A, panel by panel.
//
template <int P, typename IndexType>
static void
pack_MRxk(IndexType                     k,
          bool                          conj,
          const std::complex<double>    *A,
          IndexType                     incRowA,
          IndexType                     incColA,
          std::complex<double>          *buffer)
{
    constexpr IndexType MR = sme::BlockSizeUGemm<std::complex<double> >::MR;

    static_assert(MR==16, "SME packing of complex A requires MR = 16");

    if (incRowA!=IndexType(1) && incColA!=IndexType(1)) {
        pack_MRxk<P, IndexType, std::complex<double>, std::complex<double> >(
            k, conj, A, incRowA, incColA, buffer);
        return;
    }
    const double *a = reinterpret_cast<const double *>(A);
    double       *b = reinterpret_cast<double *>(buffer);
    for (IndexType p=0; p<P; ++p) {
        if (incRowA==IndexType(1)) {
            sme::zpack_copy(2*MR, k, a + 2*p*MR, 2*incColA, b + 2*p*k*MR,
                            conj);
        } else {
            sme::zpack_transpose(MR, k, a + 2*p*MR*incRowA, 2*incRowA,
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
          const std::complex<double>    *B,
          IndexType                     incRowB,
          IndexType                     incColB,
          std::complex<double>          *buffer)
{
    constexpr IndexType NR = sme::BlockSizeUGemm<std::complex<double> >::NR;

    static_assert(NR==16, "SME packing of complex B requires NR = 16");

    const double *b   = reinterpret_cast<const double *>(B);
    double       *buf = reinterpret_cast<double *>(buffer);
    if (incColB==IndexType(1)) {
        sme::zpack_copy(2*NR, k, b, 2*incRowB, buf, conj);
    } else if (incRowB==IndexType(1)) {
        sme::zpack_transpose(NR, k, b, 2*incColB, buf, conj);
    } else {
        pack_kxNR<IndexType, std::complex<double>, std::complex<double> >(
            k, conj, B, incRowB, incColB, buffer);
    }
}

//
//  The last panel of A with mr < 16 rows (see gepack_A).
//
template <typename IndexType>
static bool
pack_MRxk_partial(IndexType                     mr,
                  IndexType                     k,
                  bool                          conj,
                  const std::complex<double>    *A,
                  IndexType                     incRowA,
                  IndexType                     incColA,
                  std::complex<double>          *buffer)
{
    const double *a = reinterpret_cast<const double *>(A);
    double       *b = reinterpret_cast<double *>(buffer);
    if (incRowA==IndexType(1)) {
        sme::zpack_copy(2*mr, k, a, 2*incColA, b, conj);
        return true;
    }
    if (incColA==IndexType(1)) {
        sme::zpack_transpose(mr, k, a, 2*incRowA, b, conj);
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
                  const std::complex<double>    *B,
                  IndexType                     incRowB,
                  IndexType                     incColB,
                  std::complex<double>          *buffer)
{
    const double *b   = reinterpret_cast<const double *>(B);
    double       *buf = reinterpret_cast<double *>(buffer);
    if (incColB==IndexType(1)) {
        sme::zpack_copy(2*nr, k, b, 2*incRowB, buf, conj);
        return true;
    }
    if (incRowB==IndexType(1)) {
        sme::zpack_transpose(nr, k, b, 2*incColB, buf, conj);
        return true;
    }
    return false;
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_UKERNEL_SME_ZPACK_TCC
