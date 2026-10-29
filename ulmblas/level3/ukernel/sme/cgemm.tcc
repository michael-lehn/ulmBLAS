#ifndef ULMBLAS_LEVEL3_UKERNEL_SME_CGEMM_TCC
#define ULMBLAS_LEVEL3_UKERNEL_SME_CGEMM_TCC 1

//
//  ulmBLAS: experimental SME2 micro kernel for complex float (MR = 32, NR =
//  16), written for ulmBLAS (not derived from another library).
//
//  As the kernel for complex double (zgemm.tcc) with 32-bit elements: the
//  real parts of the 32 x 16 tile of C are held in ZA0.S (rows 0 ... 15)
//  and ZA1.S (rows 16 ... 31), the imaginary parts in ZA2.S and ZA3.S; the
//  operands of the outer products are swapped, so that a horizontal slice
//  of a tile is a part of a column of C.  The packed panels are those of
//  ulmBLAS for std::complex<float> (per k 32 consecutive complex elements of
//  A and 16 of B, real and imaginary part interleaved): LD2W loads 16
//  complex elements and separates them into a vector of real parts and one
//  of imaginary parts.  Per k: 3 LD2W, and for both halves of the rows
//
//      Re += re(b) re(a)^T - im(b) im(a)^T     (FMOPA, FMOPS)
//      Im += im(b) re(a)^T + re(b) im(a)^T     (FMOPA, FMOPA)
//
//  i.e. 8 outer products into the 4 tiles.  At the end the slices are read,
//  multiplied by alpha, added to beta*C (C is not read for beta = 0) and
//  stored with ST2W.  C with unit row stride; partial tiles (mr < 32, nr <
//  16) with predicated loads and stores of the rows and the number of
//  columns.  NaN in alpha, beta, A, B or C propagates.
//
//  Requirements as for the DGEMM kernel: SME2, SVL = 512 bits (Apple M4,
//  M5), USE_SME.  The panels are packed by the routines of cpack.tcc.
//
#include <arm_sme.h>
#include <complex>
#include <cstdint>

namespace ulmBLAS { namespace sme {

//
//  The rows 0 ... mr-1 and columns 0 ... nr-1 of the 32 x 16 tile of C
//  (unit row stride, column stride ldC in complex elements), in streaming
//  mode.  A, B, C as arrays of float (real and imaginary parts).
//
__arm_new("za") __arm_locally_streaming
inline void
cgemm_32x16_streaming(long kc, float ar, float ai, const float *A,
                      const float *B, float br, float bi, float *C,
                      long ldC, long mr, long nr)
{
    const svbool_t p  = svptrue_b32();
    const svbool_t p0 = svwhilelt_b32_s64(int64_t(0), int64_t(mr));
    const svbool_t p1 = svwhilelt_b32_s64(int64_t(16), int64_t(mr));

    svzero_za();
    for (long l=0; l<kc; ++l) {
        svfloat32x2_t a0 = svld2_f32(p, A);
        svfloat32x2_t a1 = svld2_f32(p, A + 32);
        svfloat32x2_t b  = svld2_f32(p, B);

        svfloat32_t ar0 = svget2(a0, 0), ai0 = svget2(a0, 1);
        svfloat32_t ar1 = svget2(a1, 0), ai1 = svget2(a1, 1);
        svfloat32_t br0 = svget2(b, 0),  bi0 = svget2(b, 1);

        svmopa_za32_f32_m(0, p, p, br0, ar0);
        svmopa_za32_f32_m(1, p, p, br0, ar1);
        svmopa_za32_f32_m(2, p, p, bi0, ar0);
        svmopa_za32_f32_m(3, p, p, bi0, ar1);
        svmops_za32_f32_m(0, p, p, bi0, ai0);
        svmops_za32_f32_m(1, p, p, bi0, ai1);
        svmopa_za32_f32_m(2, p, p, br0, ai0);
        svmopa_za32_f32_m(3, p, p, br0, ai1);

        A += 64;
        B += 32;
    }

    const bool        beta0 = br==0.0f && bi==0.0f;
    const svfloat32_t z   = svdup_f32(0.0f);
    const svfloat32_t var = svdup_f32(ar), vai = svdup_f32(ai);
    const svfloat32_t vbr = svdup_f32(br), vbi = svdup_f32(bi);

    for (long c=0; c<nr; ++c) {
        float *Cc = C + 2*c*ldC;

        svfloat32_t re0 = svread_hor_za32_f32_m(z, p, 0, c);
        svfloat32_t re1 = svread_hor_za32_f32_m(z, p, 1, c);
        svfloat32_t im0 = svread_hor_za32_f32_m(z, p, 2, c);
        svfloat32_t im1 = svread_hor_za32_f32_m(z, p, 3, c);

        //  alpha * (re + i im)
        svfloat32_t or0 = svmls_x(p, svmul_x(p, re0, var), im0, vai);
        svfloat32_t oi0 = svmla_x(p, svmul_x(p, im0, var), re0, vai);
        svfloat32_t or1 = svmls_x(p, svmul_x(p, re1, var), im1, vai);
        svfloat32_t oi1 = svmla_x(p, svmul_x(p, im1, var), re1, vai);

        if (!beta0) {
            svfloat32x2_t c0 = svld2_f32(p0, Cc);
            svfloat32x2_t c1 = svld2_f32(p1, Cc + 32);
            svfloat32_t   cr0 = svget2(c0, 0), ci0 = svget2(c0, 1);
            svfloat32_t   cr1 = svget2(c1, 0), ci1 = svget2(c1, 1);

            or0 = svmls_x(p, svmla_x(p, or0, cr0, vbr), ci0, vbi);
            oi0 = svmla_x(p, svmla_x(p, oi0, ci0, vbr), cr0, vbi);
            or1 = svmls_x(p, svmla_x(p, or1, cr1, vbr), ci1, vbi);
            oi1 = svmla_x(p, svmla_x(p, oi1, ci1, vbr), cr1, vbi);
        }
        svst2_f32(p0, Cc, svcreate2(or0, oi0));
        svst2_f32(p1, Cc + 32, svcreate2(or1, oi1));
    }
}

//
//  The same, called from ordinary code (see ugemm_32x16 for the
//  prefetching of the tile of C below: 32 rows of complex float are 256
//  bytes, as the 32 rows of double).
//
inline void
cgemm_32x16(long kc, const std::complex<float> &alpha,
            const std::complex<float> *A, const std::complex<float> *B,
            const std::complex<float> &beta, std::complex<float> *C,
            long ldC, long mr, long nr)
{
    float *C_ = reinterpret_cast<float *>(C);
    for (long c=0; c<16; ++c) {
        __builtin_prefetch(C_ + 2*c*ldC + 64, 1, 2);
        __builtin_prefetch(C_ + 2*c*ldC + 96, 1, 2);
        __builtin_prefetch(C_ + 2*c*ldC + 127, 1, 2);
    }
    cgemm_32x16_streaming(kc, alpha.real(), alpha.imag(),
                          reinterpret_cast<const float *>(A),
                          reinterpret_cast<const float *>(B),
                          beta.real(), beta.imag(), C_, ldC, mr, nr);
}

template <typename IndexType>
static typename std::enable_if<std::is_convertible<IndexType,long>::value,
void>::type
ugemm(IndexType                     kc,
      const std::complex<float>     &alpha,
      const std::complex<float>     *A,
      const std::complex<float>     *B,
      const std::complex<float>     &beta,
      std::complex<float>           *C,
      IndexType                     incRowC,
      IndexType                     incColC,
      const std::complex<float>     *,
      const std::complex<float>     *)
{
    if (incRowC==IndexType(1)) {
        cgemm_32x16(long(kc), alpha, A, B, beta, C, long(incColC), 32, 16);
        return;
    }
    std::complex<float> C_[32*16];
    cgemm_32x16(long(kc), alpha, A, B, std::complex<float>(0), C_, 32, 32, 16);
    update_strided<32>(IndexType(32), IndexType(16), beta, C_, C, incRowC,
                       incColC);
}

} // namespace sme

//
//  Buffered variant for complex float with the SME2 kernel (see the one
//  for double in ugemm.h): partial tiles of C with unit row stride are
//  updated directly.
//
template <typename IndexType>
void
ugemm(IndexType                     mr,
      IndexType                     nr,
      IndexType                     kc,
      const std::complex<float>     &alpha,
      const std::complex<float>     *A,
      const std::complex<float>     *B,
      const std::complex<float>     &beta,
      std::complex<float>           *C,
      IndexType                     incRowC,
      IndexType                     incColC,
      const std::complex<float>     *,
      const std::complex<float>     *)
{
    if (incRowC==IndexType(1)) {
        sme::cgemm_32x16(long(kc), alpha, A, B, beta, C, long(incColC),
                         long(mr), long(nr));
        return;
    }
    std::complex<float> C_[32*16];
    sme::cgemm_32x16(long(kc), alpha, A, B, std::complex<float>(0), C_, 32,
                     32, 16);
    sme::update_strided<32>(mr, nr, beta, C_, C, incRowC, incColC);
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_UKERNEL_SME_CGEMM_TCC
