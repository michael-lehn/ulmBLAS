#ifndef ULMBLAS_LEVEL3_UKERNEL_NEON_ZGEMM_TCC
#define ULMBLAS_LEVEL3_UKERNEL_NEON_ZGEMM_TCC 1

//
//  ulmBLAS: AArch64 NEON micro kernel for complex double (MR = 4, NR = 3),
//  written for ulmBLAS (not derived from another library).
//
//  Every complex element is one 128-bit register (re, im).  For the element
//  (i, j) of the tile two accumulators collect, over k,
//
//      cr(i,j) += a(i) * re(b(j)) = [ re(a) re(b), im(a) re(b) ]
//      ci(i,j) += a(i) * im(b(j)) = [ re(a) im(b), im(a) im(b) ]
//
//  with FMLA by element (vfmaq_laneq_f64: the lanes of the loaded b(j) are
//  the scalars, no broadcasts).  At the end
//
//      a*b = [ cr[0] - ci[1], cr[1] + ci[0] ] = cr + [-1, 1] * swap(ci).
//
//  24 accumulators, 4 registers for A and 3 for B: 31 of the 32 vector
//  registers; per k 7 vector loads and 24 FMLA (the ratio of the 6 x 8 DGEMM
//  kernel).  The packed panels are those of ulmBLAS for std::complex<double>:
//  per k MR (A) or NR (B) consecutive complex elements.  C may have any
//  strides (each element is loaded and stored as one register).  beta = 0:
//  C is not read; NaN in alpha, beta, A, B or C propagates.
//
#include <arm_neon.h>
#include <complex>

namespace ulmBLAS { namespace neon {

//  [-x[1], x[0]] for x = [x0, x1]: multiplication by i
inline float64x2_t
mul_i(float64x2_t x)
{
    const float64x2_t s = { -1.0, 1.0 };
    return vmulq_f64(vextq_f64(x, x, 1), s);
}

//  a * b for complex numbers in registers
inline float64x2_t
cmul(float64x2_t a, float64x2_t b)
{
    //  [re(a) re(b), im(a) re(b)] + i*[re(a) im(b), im(a) im(b)]
    return vaddq_f64(vmulq_laneq_f64(a, b, 0),
                     mul_i(vmulq_laneq_f64(a, b, 1)));
}

inline void
zgemm_4x3(long kc, const std::complex<double> &alpha, const double *A,
          const double *B, const std::complex<double> &beta,
          std::complex<double> *C_, long incRowC, long incColC)
{
    float64x2_t r00 = vdupq_n_f64(0), r01 = r00, r02 = r00,
                r10 = r00, r11 = r00, r12 = r00,
                r20 = r00, r21 = r00, r22 = r00,
                r30 = r00, r31 = r00, r32 = r00;
    float64x2_t i00 = r00, i01 = r00, i02 = r00,
                i10 = r00, i11 = r00, i12 = r00,
                i20 = r00, i21 = r00, i22 = r00,
                i30 = r00, i31 = r00, i32 = r00;

    for (long l=0; l<kc; ++l) {
        const float64x2_t a0 = vld1q_f64(A),   a1 = vld1q_f64(A+2),
                          a2 = vld1q_f64(A+4), a3 = vld1q_f64(A+6);
        const float64x2_t b0 = vld1q_f64(B),   b1 = vld1q_f64(B+2),
                          b2 = vld1q_f64(B+4);

        r00 = vfmaq_laneq_f64(r00, a0, b0, 0); i00 = vfmaq_laneq_f64(i00, a0, b0, 1);
        r10 = vfmaq_laneq_f64(r10, a1, b0, 0); i10 = vfmaq_laneq_f64(i10, a1, b0, 1);
        r20 = vfmaq_laneq_f64(r20, a2, b0, 0); i20 = vfmaq_laneq_f64(i20, a2, b0, 1);
        r30 = vfmaq_laneq_f64(r30, a3, b0, 0); i30 = vfmaq_laneq_f64(i30, a3, b0, 1);
        r01 = vfmaq_laneq_f64(r01, a0, b1, 0); i01 = vfmaq_laneq_f64(i01, a0, b1, 1);
        r11 = vfmaq_laneq_f64(r11, a1, b1, 0); i11 = vfmaq_laneq_f64(i11, a1, b1, 1);
        r21 = vfmaq_laneq_f64(r21, a2, b1, 0); i21 = vfmaq_laneq_f64(i21, a2, b1, 1);
        r31 = vfmaq_laneq_f64(r31, a3, b1, 0); i31 = vfmaq_laneq_f64(i31, a3, b1, 1);
        r02 = vfmaq_laneq_f64(r02, a0, b2, 0); i02 = vfmaq_laneq_f64(i02, a0, b2, 1);
        r12 = vfmaq_laneq_f64(r12, a1, b2, 0); i12 = vfmaq_laneq_f64(i12, a1, b2, 1);
        r22 = vfmaq_laneq_f64(r22, a2, b2, 0); i22 = vfmaq_laneq_f64(i22, a2, b2, 1);
        r32 = vfmaq_laneq_f64(r32, a3, b2, 0); i32 = vfmaq_laneq_f64(i32, a3, b2, 1);

        A += 8;
        B += 6;
    }

    const float64x2_t va = { alpha.real(), alpha.imag() };
    const float64x2_t vb = { beta.real(), beta.imag() };
    const bool        beta0 = beta.real()==0.0 && beta.imag()==0.0;
    double            *C = reinterpret_cast<double *>(C_);
    const long        rs = 2*incRowC, cs = 2*incColC;

    //  C(i,j) <- beta*C(i,j) + alpha*(a*b)(i,j)
    auto update = [&](long i, long j, float64x2_t r, float64x2_t im) {
        float64x2_t ab = vaddq_f64(r, mul_i(im));
        float64x2_t c  = cmul(va, ab);
        double      *p = C + i*rs + j*cs;
        if (!beta0) {
            c = vaddq_f64(c, cmul(vb, vld1q_f64(p)));
        }
        vst1q_f64(p, c);
    };

    update(0, 0, r00, i00); update(1, 0, r10, i10);
    update(2, 0, r20, i20); update(3, 0, r30, i30);
    update(0, 1, r01, i01); update(1, 1, r11, i11);
    update(2, 1, r21, i21); update(3, 1, r31, i31);
    update(0, 2, r02, i02); update(1, 2, r12, i12);
    update(2, 2, r22, i22); update(3, 2, r32, i32);
}

template <typename IndexType>
static typename std::enable_if<std::is_convertible<IndexType,long>::value,
void>::type
ugemm(IndexType                     kc,
      const std::complex<double>    &alpha,
      const std::complex<double>    *A,
      const std::complex<double>    *B,
      const std::complex<double>    &beta,
      std::complex<double>          *C,
      IndexType                     incRowC,
      IndexType                     incColC,
      const std::complex<double>    *,
      const std::complex<double>    *)
{
    zgemm_4x3(long(kc), alpha, reinterpret_cast<const double *>(A),
              reinterpret_cast<const double *>(B), beta, C, long(incRowC),
              long(incColC));
}

} } // namespace neon, ulmBLAS

#endif // ULMBLAS_LEVEL3_UKERNEL_NEON_ZGEMM_TCC
