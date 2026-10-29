#ifndef ULMBLAS_LEVEL3_UKERNEL_NEON_CGEMM_TCC
#define ULMBLAS_LEVEL3_UKERNEL_NEON_CGEMM_TCC 1

//
//  ulmBLAS: AArch64 NEON micro kernel for complex float (MR = 4, NR = 6),
//  written for ulmBLAS (not derived from another library).
//
//  As the kernel for complex double (zgemm.tcc), with two complex elements
//  (re, im, re, im) per 128-bit register: for the rows 2*h, 2*h+1 and the
//  column j of the tile two accumulators collect, over k,
//
//      cr(h,j) += a(h) * re(b(j)),     ci(h,j) += a(h) * im(b(j))
//
//  with FMLA by element (vfmaq_laneq_f32, b(j) are the lanes 2*(j%2) and
//  2*(j%2)+1 of the loaded register j/2).  At the end
//
//      a*b = cr + [-1, 1, -1, 1] * rev64(ci).
//
//  24 accumulators, 2 registers for A and 3 for B: 29 of the 32 vector
//  registers; per k 5 vector loads and 24 FMLA (as the 8 x 12 SGEMM
//  kernel).  The packed panels are those of ulmBLAS for std::complex<float>.
//  C may have any strides (each complex element is loaded and stored as 64
//  bits; two consecutive ones as one register for unit row stride).  beta =
//  0: C is not read; NaN in alpha, beta, A, B or C propagates.
//
#include <arm_neon.h>
#include <complex>

namespace ulmBLAS { namespace neon {

//  [-x1, x0, -x3, x2] for x = [x0, x1, x2, x3]: multiplication by i
inline float32x4_t
mul_i(float32x4_t x)
{
    const float32x4_t s = { -1.0f, 1.0f, -1.0f, 1.0f };
    return vmulq_f32(vrev64q_f32(x), s);
}

//  (r + i*im) * x for two complex numbers x in a register
inline float32x4_t
cmul(float r, float im, float32x4_t x)
{
    return vfmaq_n_f32(vmulq_n_f32(x, r), mul_i(x), im);
}

inline void
cgemm_4x6(long kc, const std::complex<float> &alpha, const float *A,
          const float *B, const std::complex<float> &beta,
          std::complex<float> *C_, long incRowC, long incColC)
{
    float32x4_t r[6][2], im[6][2];

    for (int j=0; j<6; ++j) {
        r[j][0] = r[j][1] = im[j][0] = im[j][1] = vdupq_n_f32(0);
    }

#define ULMBLAS_NEON_CFMA(j, b, l)                                          \
    r[j][0]  = vfmaq_laneq_f32(r[j][0],  a0, b, l);                         \
    r[j][1]  = vfmaq_laneq_f32(r[j][1],  a1, b, l);                         \
    im[j][0] = vfmaq_laneq_f32(im[j][0], a0, b, l+1);                       \
    im[j][1] = vfmaq_laneq_f32(im[j][1], a1, b, l+1);

    for (long l=0; l<kc; ++l) {
        const float32x4_t a0 = vld1q_f32(A), a1 = vld1q_f32(A+4);
        const float32x4_t b0 = vld1q_f32(B), b1 = vld1q_f32(B+4),
                          b2 = vld1q_f32(B+8);

        ULMBLAS_NEON_CFMA(0, b0, 0) ULMBLAS_NEON_CFMA(1, b0, 2)
        ULMBLAS_NEON_CFMA(2, b1, 0) ULMBLAS_NEON_CFMA(3, b1, 2)
        ULMBLAS_NEON_CFMA(4, b2, 0) ULMBLAS_NEON_CFMA(5, b2, 2)

        A += 8;
        B += 12;
    }

#undef ULMBLAS_NEON_CFMA

    const float ar = alpha.real(), ai = alpha.imag();
    const float br = beta.real(),  bi = beta.imag();
    const bool  beta0 = br==0.0f && bi==0.0f;
    float       *C = reinterpret_cast<float *>(C_);
    const long  rs = 2*incRowC, cs = 2*incColC;

    for (int j=0; j<6; ++j) {
        for (int h=0; h<2; ++h) {
            float       *p0 = C + 2*h*rs + j*cs;
            float       *p1 = p0 + rs;
            float32x4_t c   = cmul(ar, ai, vaddq_f32(r[j][h], mul_i(im[j][h])));

            if (incRowC==1) {
                if (!beta0) {
                    c = vaddq_f32(c, cmul(br, bi, vld1q_f32(p0)));
                }
                vst1q_f32(p0, c);
            } else {
                if (!beta0) {
                    float32x4_t x = vcombine_f32(vld1_f32(p0), vld1_f32(p1));
                    c = vaddq_f32(c, cmul(br, bi, x));
                }
                vst1_f32(p0, vget_low_f32(c));
                vst1_f32(p1, vget_high_f32(c));
            }
        }
    }
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
    cgemm_4x6(long(kc), alpha, reinterpret_cast<const float *>(A),
              reinterpret_cast<const float *>(B), beta, C, long(incRowC),
              long(incColC));
}

} } // namespace neon, ulmBLAS

#endif // ULMBLAS_LEVEL3_UKERNEL_NEON_CGEMM_TCC
