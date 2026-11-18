#ifndef ULMBLAS_LEVEL3_UKERNEL_NEON_SMALL_H
#define ULMBLAS_LEVEL3_UKERNEL_NEON_SMALL_H 1

//
//  GEMM for small matrices on AArch64 with NEON (no streaming mode): the
//  algorithm of ulmblas/level3/ukernel/simd/small.h with the NEON vector
//  operations below.
//
//      float            L = 4, MV = 4, NC = 6 (16 x 6 tile, 24 of 32
//                       registers); complex MV = 4, NC = 3 (8 x 3)
//      double           L = 2, MV = 4, NC = 6 (8 x 6 tile); complex 4 x 3
//
//  NEON has no masked loads and stores: elements beyond r one by one.
//
//  neon::SmallGemm<Fallback> (all sizes) and neon::TinyGemm<Fallback> (m,
//  n <= 16: problems below the grid of make tune, where entering the
//  streaming mode of SME costs more than the kernel gains; on an Apple M5
//  relative to OpenBLAS, dgemm 8 x 8 x 64 and 16 x 16 x 16, all four
//  layouts: 0.93 to 1.4, sme small kernel 0.6 to 0.8 for A row major).
//
//  Defined on AArch64 (ULMBLAS_NEON_SMALL is then defined).
//

#include <cstddef>
#include <ulmblas/config/inline.h>
#include <ulmblas/level3/ukernel/simd/small.h>

#if defined(__aarch64__)

#include <arm_neon.h>

#define ULMBLAS_NEON_SMALL 1

namespace ulmBLAS {
namespace neon {

//  the instruction set tag
struct Isa
{
};

} // namespace neon

namespace simd {

using neon::Isa;

template <> struct Vec<Isa, float>
{
    typedef float32x4_t reg;
    static constexpr std::ptrdiff_t L = 4, MV = 4, NC = 6, DR = 4, DC = 4;

    static ULMBLAS_ALWAYS_INLINE reg
    zero()
    {
	return vdupq_n_f32(0.0f);
    }
    //  the first r (1 <= r <= L) elements at x, the others zero
    static ULMBLAS_ALWAYS_INLINE reg
    load(const float *x, std::ptrdiff_t r)
    {
	if (r == L) {
	    return vld1q_f32(x);
	}
	reg v = vdupq_n_f32(0.0f);
	v = vld1q_lane_f32(x, v, 0);
	if (r > 1) {
	    v = vld1q_lane_f32(x + 1, v, 1);
	}
	if (r > 2) {
	    v = vld1q_lane_f32(x + 2, v, 2);
	}
	return v;
    }
    static ULMBLAS_ALWAYS_INLINE void
    store(float *x, reg v, std::ptrdiff_t r)
    {
	if (r == L) {
	    vst1q_f32(x, v);
	    return;
	}
	vst1q_lane_f32(x, v, 0);
	if (r > 1) {
	    vst1q_lane_f32(x + 1, v, 1);
	}
	if (r > 2) {
	    vst1q_lane_f32(x + 2, v, 2);
	}
    }
    static ULMBLAS_ALWAYS_INLINE reg
    broadcast(float x)
    {
	return vdupq_n_f32(x);
    }
    static ULMBLAS_ALWAYS_INLINE reg
    fma(reg a, reg b, reg c)
    {
	return vfmaq_f32(c, a, b);
    }
    static ULMBLAS_ALWAYS_INLINE reg
    mul(reg a, reg b)
    {
	return vmulq_f32(a, b);
    }
    static ULMBLAS_ALWAYS_INLINE reg
    add(reg a, reg b)
    {
	return vaddq_f32(a, b);
    }
    static ULMBLAS_ALWAYS_INLINE float
    reduce(reg v)
    {
	return vaddvq_f32(v);
    }
};

template <> struct Vec<Isa, double>
{
    typedef float64x2_t reg;
    static constexpr std::ptrdiff_t L = 2, MV = 4, NC = 6, DR = 4, DC = 4;

    static ULMBLAS_ALWAYS_INLINE reg
    zero()
    {
	return vdupq_n_f64(0.0);
    }
    static ULMBLAS_ALWAYS_INLINE reg
    load(const double *x, std::ptrdiff_t r)
    {
	return r == L ? vld1q_f64(x) : vld1q_lane_f64(x, vdupq_n_f64(0.0), 0);
    }
    static ULMBLAS_ALWAYS_INLINE void
    store(double *x, reg v, std::ptrdiff_t r)
    {
	if (r == L) {
	    vst1q_f64(x, v);
	} else {
	    vst1q_lane_f64(x, v, 0);
	}
    }
    static ULMBLAS_ALWAYS_INLINE reg
    broadcast(double x)
    {
	return vdupq_n_f64(x);
    }
    static ULMBLAS_ALWAYS_INLINE reg
    fma(reg a, reg b, reg c)
    {
	return vfmaq_f64(c, a, b);
    }
    static ULMBLAS_ALWAYS_INLINE reg
    mul(reg a, reg b)
    {
	return vmulq_f64(a, b);
    }
    static ULMBLAS_ALWAYS_INLINE reg
    add(reg a, reg b)
    {
	return vaddq_f64(a, b);
    }
    static ULMBLAS_ALWAYS_INLINE double
    reduce(reg v)
    {
	return vaddvq_f64(v);
    }
};

template <> struct CVec<Isa, float>
{
    typedef float32x4_t reg;
    static constexpr std::ptrdiff_t MV = 4, NC = 3;

    //  (re, im) -> (im, re) per complex element
    static ULMBLAS_ALWAYS_INLINE reg
    swap(reg v)
    {
	return vrev64q_f32(v);
    }
    //  (s0, s1) in every complex element
    static ULMBLAS_ALWAYS_INLINE reg
    pair(float s0, float s1)
    {
	const float s[4] = {s0, s1, s0, s1};
	return vld1q_f32(s);
    }
};

template <> struct CVec<Isa, double>
{
    typedef float64x2_t reg;
    static constexpr std::ptrdiff_t MV = 4, NC = 3;

    static ULMBLAS_ALWAYS_INLINE reg
    swap(reg v)
    {
	return vextq_f64(v, v, 1);
    }
    static ULMBLAS_ALWAYS_INLINE reg
    pair(double s0, double s1)
    {
	const double s[2] = {s0, s1};
	return vld1q_f64(s);
    }
};

} // namespace simd

namespace neon {

template <typename Fallback> using SmallGemm = simd::SmallGemm<Isa, Fallback>;

template <typename Fallback>
using TinyGemm = simd::SmallGemm<Isa, Fallback, 16, 16>;

} // namespace neon
} // namespace ulmBLAS

#endif // __aarch64__

#endif // ULMBLAS_LEVEL3_UKERNEL_NEON_SMALL_H
