#ifndef ULMBLAS_LEVEL3_UKERNEL_AVX_SMALL_H
#define ULMBLAS_LEVEL3_UKERNEL_AVX_SMALL_H 1

//
//  GEMM for small matrices on x86-64 with AVX2/FMA or AVX-512F: the
//  algorithm of ulmblas/level3/ukernel/simd/small.h with the vector
//  operations below (masked loads and stores for the rows beyond r).
//
//                  AVX-512F (32 registers)      AVX2/FMA (16 registers)
//      float       L = 16, 32 x 8 tile          L = 8, 16 x 6 tile
//      double      L = 8, 24 x 8 tile           L = 4, 8 x 6 tile
//      complex     2 vectors x 6 columns        2 vectors x 2 columns
//      dot tile    4 x 4                        2 x 4
//
//  avx::SmallGemm<Fallback> (all sizes); the selection tables bound the
//  sizes (ulmblas/level3/gemm/select.h).
//
//  Defined if the build has FMA and AVX2 (ULMBLAS_AVX_SMALL is then
//  defined).
//

#include <cstddef>
#include <ulmblas/config/inline.h>
#include <ulmblas/level3/ukernel/simd/small.h>

#if defined(__x86_64__) && defined(__AVX2__) && defined(__FMA__)

#include <immintrin.h>

#define ULMBLAS_AVX_SMALL 1

namespace ulmBLAS {
namespace avx {

//  the instruction set tag
struct Isa
{
};

} // namespace avx

namespace simd {

#if defined(__AVX512F__)

template <> struct Vec<avx::Isa, float>
{
    typedef __m512 reg;
    static constexpr std::ptrdiff_t L = 16, MV = 2, NC = 8, DR = 4, DC = 4;

    static ULMBLAS_ALWAYS_INLINE __mmask16
    mask(std::ptrdiff_t r)
    {
	return __mmask16((1u << r) - 1);
    }
    static ULMBLAS_ALWAYS_INLINE reg
    zero()
    {
	return _mm512_setzero_ps();
    }
    //  the first r (1 <= r <= L) elements at x, the others zero
    static ULMBLAS_ALWAYS_INLINE reg
    load(const float *x, std::ptrdiff_t r)
    {
	return r == L ? _mm512_loadu_ps(x) : _mm512_maskz_loadu_ps(mask(r), x);
    }
    static ULMBLAS_ALWAYS_INLINE void
    store(float *x, reg v, std::ptrdiff_t r)
    {
	if (r == L) {
	    _mm512_storeu_ps(x, v);
	} else {
	    _mm512_mask_storeu_ps(x, mask(r), v);
	}
    }
    static ULMBLAS_ALWAYS_INLINE reg
    broadcast(float x)
    {
	return _mm512_set1_ps(x);
    }
    static ULMBLAS_ALWAYS_INLINE reg
    fma(reg a, reg b, reg c)
    {
	return _mm512_fmadd_ps(a, b, c);
    }
    static ULMBLAS_ALWAYS_INLINE reg
    mul(reg a, reg b)
    {
	return _mm512_mul_ps(a, b);
    }
    static ULMBLAS_ALWAYS_INLINE reg
    add(reg a, reg b)
    {
	return _mm512_add_ps(a, b);
    }
    static ULMBLAS_ALWAYS_INLINE float
    reduce(reg v)
    {
	return _mm512_reduce_add_ps(v);
    }
};

template <> struct Vec<avx::Isa, double>
{
    typedef __m512d reg;
    static constexpr std::ptrdiff_t L = 8, MV = 3, NC = 8, DR = 4, DC = 4;

    static ULMBLAS_ALWAYS_INLINE __mmask8
    mask(std::ptrdiff_t r)
    {
	return __mmask8((1u << r) - 1);
    }
    static ULMBLAS_ALWAYS_INLINE reg
    zero()
    {
	return _mm512_setzero_pd();
    }
    static ULMBLAS_ALWAYS_INLINE reg
    load(const double *x, std::ptrdiff_t r)
    {
	return r == L ? _mm512_loadu_pd(x) : _mm512_maskz_loadu_pd(mask(r), x);
    }
    static ULMBLAS_ALWAYS_INLINE void
    store(double *x, reg v, std::ptrdiff_t r)
    {
	if (r == L) {
	    _mm512_storeu_pd(x, v);
	} else {
	    _mm512_mask_storeu_pd(x, mask(r), v);
	}
    }
    static ULMBLAS_ALWAYS_INLINE reg
    broadcast(double x)
    {
	return _mm512_set1_pd(x);
    }
    static ULMBLAS_ALWAYS_INLINE reg
    fma(reg a, reg b, reg c)
    {
	return _mm512_fmadd_pd(a, b, c);
    }
    static ULMBLAS_ALWAYS_INLINE reg
    mul(reg a, reg b)
    {
	return _mm512_mul_pd(a, b);
    }
    static ULMBLAS_ALWAYS_INLINE reg
    add(reg a, reg b)
    {
	return _mm512_add_pd(a, b);
    }
    static ULMBLAS_ALWAYS_INLINE double
    reduce(reg v)
    {
	return _mm512_reduce_add_pd(v);
    }
};

template <> struct CVec<avx::Isa, float>
{
    typedef __m512 reg;
    static constexpr std::ptrdiff_t MV = 2, NC = 6;

    //  (re, im) -> (im, re) per complex element
    static ULMBLAS_ALWAYS_INLINE reg
    swap(reg v)
    {
	return _mm512_permute_ps(v, 0xb1);
    }
    //  (s0, s1) in every complex element
    static ULMBLAS_ALWAYS_INLINE reg
    pair(float s0, float s1)
    {
	return _mm512_setr_ps(s0, s1, s0, s1, s0, s1, s0, s1, s0, s1, s0, s1,
	                      s0, s1, s0, s1);
    }
};

template <> struct CVec<avx::Isa, double>
{
    typedef __m512d reg;
    static constexpr std::ptrdiff_t MV = 2, NC = 6;

    static ULMBLAS_ALWAYS_INLINE reg
    swap(reg v)
    {
	return _mm512_permute_pd(v, 0x55);
    }
    static ULMBLAS_ALWAYS_INLINE reg
    pair(double s0, double s1)
    {
	return _mm512_setr_pd(s0, s1, s0, s1, s0, s1, s0, s1);
    }
};

#else // AVX2

template <> struct Vec<avx::Isa, float>
{
    typedef __m256 reg;
    static constexpr std::ptrdiff_t L = 8, MV = 2, NC = 6, DR = 2, DC = 4;

    static ULMBLAS_ALWAYS_INLINE __m256i
    mask(std::ptrdiff_t r)
    {
	static const std::int32_t m[16] = {-1, -1, -1, -1, -1, -1, -1, -1,
	                                   0,  0,  0,  0,  0,  0,  0,  0};
	return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(m + 8 - r));
    }
    static ULMBLAS_ALWAYS_INLINE reg
    zero()
    {
	return _mm256_setzero_ps();
    }
    static ULMBLAS_ALWAYS_INLINE reg
    load(const float *x, std::ptrdiff_t r)
    {
	return r == L ? _mm256_loadu_ps(x) : _mm256_maskload_ps(x, mask(r));
    }
    static ULMBLAS_ALWAYS_INLINE void
    store(float *x, reg v, std::ptrdiff_t r)
    {
	if (r == L) {
	    _mm256_storeu_ps(x, v);
	} else {
	    _mm256_maskstore_ps(x, mask(r), v);
	}
    }
    static ULMBLAS_ALWAYS_INLINE reg
    broadcast(float x)
    {
	return _mm256_set1_ps(x);
    }
    static ULMBLAS_ALWAYS_INLINE reg
    fma(reg a, reg b, reg c)
    {
	return _mm256_fmadd_ps(a, b, c);
    }
    static ULMBLAS_ALWAYS_INLINE reg
    mul(reg a, reg b)
    {
	return _mm256_mul_ps(a, b);
    }
    static ULMBLAS_ALWAYS_INLINE reg
    add(reg a, reg b)
    {
	return _mm256_add_ps(a, b);
    }
    static ULMBLAS_ALWAYS_INLINE float
    reduce(reg v)
    {
	__m128 s =
	    _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
	s = _mm_add_ps(s, _mm_movehl_ps(s, s));
	s = _mm_add_ss(s, _mm_movehdup_ps(s));
	return _mm_cvtss_f32(s);
    }
};

template <> struct Vec<avx::Isa, double>
{
    typedef __m256d reg;
    static constexpr std::ptrdiff_t L = 4, MV = 2, NC = 6, DR = 2, DC = 4;

    static ULMBLAS_ALWAYS_INLINE __m256i
    mask(std::ptrdiff_t r)
    {
	static const std::int64_t m[8] = {-1, -1, -1, -1, 0, 0, 0, 0};
	return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(m + 4 - r));
    }
    static ULMBLAS_ALWAYS_INLINE reg
    zero()
    {
	return _mm256_setzero_pd();
    }
    static ULMBLAS_ALWAYS_INLINE reg
    load(const double *x, std::ptrdiff_t r)
    {
	return r == L ? _mm256_loadu_pd(x) : _mm256_maskload_pd(x, mask(r));
    }
    static ULMBLAS_ALWAYS_INLINE void
    store(double *x, reg v, std::ptrdiff_t r)
    {
	if (r == L) {
	    _mm256_storeu_pd(x, v);
	} else {
	    _mm256_maskstore_pd(x, mask(r), v);
	}
    }
    static ULMBLAS_ALWAYS_INLINE reg
    broadcast(double x)
    {
	return _mm256_set1_pd(x);
    }
    static ULMBLAS_ALWAYS_INLINE reg
    fma(reg a, reg b, reg c)
    {
	return _mm256_fmadd_pd(a, b, c);
    }
    static ULMBLAS_ALWAYS_INLINE reg
    mul(reg a, reg b)
    {
	return _mm256_mul_pd(a, b);
    }
    static ULMBLAS_ALWAYS_INLINE reg
    add(reg a, reg b)
    {
	return _mm256_add_pd(a, b);
    }
    static ULMBLAS_ALWAYS_INLINE double
    reduce(reg v)
    {
	__m128d s =
	    _mm_add_pd(_mm256_castpd256_pd128(v), _mm256_extractf128_pd(v, 1));
	s = _mm_add_sd(s, _mm_unpackhi_pd(s, s));
	return _mm_cvtsd_f64(s);
    }
};

template <> struct CVec<avx::Isa, float>
{
    typedef __m256 reg;
    static constexpr std::ptrdiff_t MV = 2, NC = 2;

    static ULMBLAS_ALWAYS_INLINE reg
    swap(reg v)
    {
	return _mm256_permute_ps(v, 0xb1);
    }
    static ULMBLAS_ALWAYS_INLINE reg
    pair(float s0, float s1)
    {
	return _mm256_setr_ps(s0, s1, s0, s1, s0, s1, s0, s1);
    }
};

template <> struct CVec<avx::Isa, double>
{
    typedef __m256d reg;
    static constexpr std::ptrdiff_t MV = 2, NC = 2;

    static ULMBLAS_ALWAYS_INLINE reg
    swap(reg v)
    {
	return _mm256_permute_pd(v, 0x5);
    }
    static ULMBLAS_ALWAYS_INLINE reg
    pair(double s0, double s1)
    {
	return _mm256_setr_pd(s0, s1, s0, s1);
    }
};

#endif // AVX2

} // namespace simd

namespace avx {

template <typename Fallback> using SmallGemm = simd::SmallGemm<Isa, Fallback>;

} // namespace avx
} // namespace ulmBLAS

#endif // __x86_64__ && __AVX2__ && __FMA__

#endif // ULMBLAS_LEVEL3_UKERNEL_AVX_SMALL_H
