// Micro kernels for blocked::gemm: AB <- alpha A_ B_ for an MR x NR tile,
// A_ a panel of MR rows and kc columns, B_ a panel of kc rows and NR
// columns, both packed; AB is a column-major MR x NR buffer.
#ifndef KERNELS_H
#define KERNELS_H 1

#if defined(__aarch64__)
#include <arm_neon.h>
#elif defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

namespace blocked {

// 1. plain C++: kc rank-1 updates of the tile
template <long MR_, long NR_> struct RefKernel
{
    static const long MR = MR_, NR = NR_;

    static void
    run(long kc, double alpha, const double *A, const double *B, double *AB)
    {
	double ab[MR * NR] = {};
	for (long l = 0; l < kc; ++l) {
	    for (long j = 0; j < NR; ++j) {
		for (long i = 0; i < MR; ++i) {
		    ab[i + j * MR] += A[i] * B[j];
		}
	    }
	    A += MR;
	    B += NR;
	}
	for (long i = 0; i < MR * NR; ++i) {
	    AB[i] = alpha * ab[i];
	}
    }
};

// 2. the vector extensions of GCC (also in clang): vectors of two doubles,
//    the tile as 4 x 6 vectors, which the compiler keeps in registers
struct VecKernel
{
    static const long MR = 8, NR = 6;
    typedef double vec __attribute__((vector_size(16)));

    static void
    run(long kc, double alpha, const double *A, const double *B, double *AB)
    {
	vec ab[NR][MR / 2] = {};
	for (long l = 0; l < kc; ++l) {
	    vec a[MR / 2];
	    for (long i = 0; i < MR / 2; ++i) {
		a[i] = vec{A[2 * i], A[2 * i + 1]};
	    }
	    for (long j = 0; j < NR; ++j) {
		const vec b = {B[j], B[j]}; // broadcast
		for (long i = 0; i < MR / 2; ++i) {
		    ab[j][i] += a[i] * b;
		}
	    }
	    A += MR;
	    B += NR;
	}
	for (long j = 0; j < NR; ++j) {
	    for (long i = 0; i < MR / 2; ++i) {
		const vec x = ab[j][i] * alpha;
		AB[2 * i + j * MR] = x[0];
		AB[2 * i + 1 + j * MR] = x[1];
	    }
	}
    }
};

#if defined(__aarch64__)
// 3. NEON intrinsics: the 8 x 6 tile in 24 registers of two doubles; one
//    FMA multiplies a column of A (4 registers) with one element of B, taken
//    directly from a register (vfmaq_laneq_f64)
struct NeonKernel
{
    static const long MR = 8, NR = 6;

    static void
    run(long kc, double alpha, const double *A, const double *B, double *AB)
    {
	float64x2_t ab[NR][4];
	for (long j = 0; j < NR; ++j) {
	    for (long i = 0; i < 4; ++i) {
		ab[j][i] = vdupq_n_f64(0);
	    }
	}
	for (long l = 0; l < kc; ++l) {
	    const float64x2_t a0 = vld1q_f64(A), a1 = vld1q_f64(A + 2);
	    const float64x2_t a2 = vld1q_f64(A + 4), a3 = vld1q_f64(A + 6);
	    const float64x2_t b01 = vld1q_f64(B), b23 = vld1q_f64(B + 2);
	    const float64x2_t b45 = vld1q_f64(B + 4);
#define COLUMN(j, b, lane)                                                     \
    ab[j][0] = vfmaq_laneq_f64(ab[j][0], a0, b, lane);                         \
    ab[j][1] = vfmaq_laneq_f64(ab[j][1], a1, b, lane);                         \
    ab[j][2] = vfmaq_laneq_f64(ab[j][2], a2, b, lane);                         \
    ab[j][3] = vfmaq_laneq_f64(ab[j][3], a3, b, lane);
	    COLUMN(0, b01, 0)
	    COLUMN(1, b01, 1)
	    COLUMN(2, b23, 0)
	    COLUMN(3, b23, 1)
	    COLUMN(4, b45, 0)
	    COLUMN(5, b45, 1)
#undef COLUMN
	    A += MR;
	    B += NR;
	}
	for (long j = 0; j < NR; ++j) {
	    for (long i = 0; i < 4; ++i) {
		vst1q_f64(&AB[2 * i + j * MR], vmulq_n_f64(ab[j][i], alpha));
	    }
	}
    }
};
#endif

#if defined(__AVX2__) && defined(__FMA__)
// 3'. AVX2 intrinsics (x86-64): the 8 x 6 tile in 12 registers of four
//     doubles; each element of B is broadcast into a register
struct AvxKernel
{
    static const long MR = 8, NR = 6;

    static void
    run(long kc, double alpha, const double *A, const double *B, double *AB)
    {
	__m256d ab[NR][2];
	for (long j = 0; j < NR; ++j) {
	    ab[j][0] = ab[j][1] = _mm256_setzero_pd();
	}
	for (long l = 0; l < kc; ++l) {
	    const __m256d a0 = _mm256_load_pd(A), a1 = _mm256_load_pd(A + 4);
	    for (long j = 0; j < NR; ++j) {
		const __m256d b = _mm256_broadcast_sd(B + j);
		ab[j][0] = _mm256_fmadd_pd(a0, b, ab[j][0]);
		ab[j][1] = _mm256_fmadd_pd(a1, b, ab[j][1]);
	    }
	    A += MR;
	    B += NR;
	}
	const __m256d al = _mm256_set1_pd(alpha);
	for (long j = 0; j < NR; ++j) {
	    _mm256_store_pd(&AB[j * MR], _mm256_mul_pd(ab[j][0], al));
	    _mm256_store_pd(&AB[4 + j * MR], _mm256_mul_pd(ab[j][1], al));
	}
    }
};
#endif

// the kernel with intrinsics for this processor, if there is one
#if defined(__aarch64__)
typedef NeonKernel SimdKernel;
#define HAVE_SIMD_KERNEL 1
#elif defined(__AVX2__) && defined(__FMA__)
typedef AvxKernel SimdKernel;
#define HAVE_SIMD_KERNEL 1
#endif

} // namespace blocked

#endif // KERNELS_H
