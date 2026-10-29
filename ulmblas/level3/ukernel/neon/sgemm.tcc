#ifndef ULMBLAS_LEVEL3_UKERNEL_NEON_SGEMM_TCC
#define ULMBLAS_LEVEL3_UKERNEL_NEON_SGEMM_TCC 1

//
//  ulmBLAS: AArch64 NEON micro kernel for float (MR = 8, NR = 12), written
//  for ulmBLAS (not derived from another library).
//
//  The 8 x 12 tile of C is held in 24 registers, two per column (rows 0..3
//  and 4..7).  Per k the column of the panel of A is loaded into 2
//  registers, the row of the panel of B into 3, and each column j of the
//  tile is updated with FMLA by element (vfmaq_laneq_f32: the lanes of the
//  loaded row of B are the scalars, no broadcasts): 29 of the 32 vector
//  registers, per k 5 vector loads and 24 FMLA (the ratio of the 6 x 8
//  DGEMM kernel).  The packed panels are those of ulmBLAS for float.
//
//  C with unit row stride is loaded and stored with vector loads and
//  stores; for other strides alpha*A*B is stored into a local tile and C is
//  updated in C++.  beta = 0: C is not read; NaN in alpha, beta, A, B or C
//  propagates.
//
#include <arm_neon.h>

namespace ulmBLAS { namespace neon {

inline void
sgemm_8x12(long kc, float alpha, const float *A, const float *B, float beta,
           float *C, long incRowC, long incColC)
{
    float32x4_t c[12][2];

    for (int j=0; j<12; ++j) {
        c[j][0] = c[j][1] = vdupq_n_f32(0);
    }

#define ULMBLAS_NEON_SFMA(j, b, l)                                          \
    c[j][0] = vfmaq_laneq_f32(c[j][0], a0, b, l);                           \
    c[j][1] = vfmaq_laneq_f32(c[j][1], a1, b, l);

    for (long l=0; l<kc; ++l) {
        const float32x4_t a0 = vld1q_f32(A), a1 = vld1q_f32(A+4);
        const float32x4_t b0 = vld1q_f32(B), b1 = vld1q_f32(B+4),
                          b2 = vld1q_f32(B+8);

        ULMBLAS_NEON_SFMA( 0, b0, 0) ULMBLAS_NEON_SFMA( 1, b0, 1)
        ULMBLAS_NEON_SFMA( 2, b0, 2) ULMBLAS_NEON_SFMA( 3, b0, 3)
        ULMBLAS_NEON_SFMA( 4, b1, 0) ULMBLAS_NEON_SFMA( 5, b1, 1)
        ULMBLAS_NEON_SFMA( 6, b1, 2) ULMBLAS_NEON_SFMA( 7, b1, 3)
        ULMBLAS_NEON_SFMA( 8, b2, 0) ULMBLAS_NEON_SFMA( 9, b2, 1)
        ULMBLAS_NEON_SFMA(10, b2, 2) ULMBLAS_NEON_SFMA(11, b2, 3)

        A += 8;
        B += 12;
    }

#undef ULMBLAS_NEON_SFMA

    if (incRowC==1) {
        for (int j=0; j<12; ++j) {
            float       *p  = C + j*incColC;
            float32x4_t c0 = vmulq_n_f32(c[j][0], alpha);
            float32x4_t c1 = vmulq_n_f32(c[j][1], alpha);
            if (beta!=0.0f) {
                c0 = vfmaq_n_f32(c0, vld1q_f32(p), beta);
                c1 = vfmaq_n_f32(c1, vld1q_f32(p+4), beta);
            }
            vst1q_f32(p, c0);
            vst1q_f32(p+4, c1);
        }
        return;
    }

    float C_[8*12];
    for (int j=0; j<12; ++j) {
        vst1q_f32(C_ + 8*j,     vmulq_n_f32(c[j][0], alpha));
        vst1q_f32(C_ + 8*j + 4, vmulq_n_f32(c[j][1], alpha));
    }
    for (long j=0; j<12; ++j) {
        for (long i=0; i<8; ++i) {
            float &x = C[i*incRowC+j*incColC];
            x = (beta==0.0f) ? C_[i+8*j] : beta*x + C_[i+8*j];
        }
    }
}

template <typename IndexType>
static typename std::enable_if<std::is_convertible<IndexType,long>::value,
void>::type
ugemm(IndexType      kc,
      const float    &alpha,
      const float    *A,
      const float    *B,
      const float    &beta,
      float          *C,
      IndexType      incRowC,
      IndexType      incColC,
      const float    *,
      const float    *)
{
    sgemm_8x12(long(kc), alpha, A, B, beta, C, long(incRowC), long(incColC));
}

} } // namespace neon, ulmBLAS

#endif // ULMBLAS_LEVEL3_UKERNEL_NEON_SGEMM_TCC
