#ifndef ULMBLAS_LEVEL3_UKERNEL_AVX_I8GEMM_TCC
#define ULMBLAS_LEVEL3_UKERNEL_AVX_I8GEMM_TCC 1

//
//  ulmBLAS: micro kernel of the 8-bit integer GEMM (level3/i8gemm/driver.h)
//  for AVX2: an 8 x 6 tile of C (int32).  The panels hold the values as
//  int16 (sign or zero extended when packed), in groups of 4 values of k,
//  so one kernel serves all four combinations of signed and unsigned A and
//  B, and the products are exact (VPMADDWD: int16 x int16, pairs summed
//  into int32; |a*b| <= 255*255).  Per group of 4 values of k: two loads of
//  A (8 rows x 4 values), per column of B a broadcast of its 4 values and
//  two VPMADDWD into the two accumulators of the column (int32 lane t of an
//  accumulator: row t/2, values 2*(t%2) and 2*(t%2)+1 of the group).  At
//  the end the pairs are added (VPHADDD), which gives the 8 rows of the
//  column in order after a permutation.
//
//  With AVX-VNNI (VPDPWSSD: the multiply and the addition in one
//  instruction, e.g. Intel Alder Lake) the same kernel with the fused
//  instruction, selected at run time (hasAvxVnni; target attribute, so the
//  avx2 backend needs no extra flags).
//
//  Not used for the mixed signs with VPDPBUSD (unsigned x signed bytes,
//  twice the products per instruction) yet.
//
#include <cstdint>
#include <immintrin.h>
#include <type_traits>

#if defined(__GNUC__)
#include <cpuid.h>
#endif

#define ULMBLAS_AVX_VNNI __attribute__((target("avx2,avxvnni")))

namespace ulmBLAS {
namespace avx {

//  -DULMBLAS_I8GEMM_NO_VNNI: the kernel without AVX-VNNI also on CPUs with
//  it (tests)
inline bool
hasAvxVnni()
{
#if defined(ULMBLAS_I8GEMM_NO_VNNI)
    return false;
#elif defined(__GNUC__)
    static const bool has = [] {
	unsigned a, b, c, d;
	if (!__get_cpuid_count(7, 1, &a, &b, &c, &d)) {
	    return false;
	}
	return (a & (1u << 4)) != 0; // CPUID.(7,1):EAX[4] AVX-VNNI
    }();
    return has;
#else
    return false;
#endif
}

struct MaddPlain
{
    static inline __m256i
    madd(__m256i acc, __m256i a, __m256i b)
    {
	return _mm256_add_epi32(acc, _mm256_madd_epi16(a, b));
    }
};

struct MaddVnni
{
    ULMBLAS_AVX_VNNI static inline __m256i
    madd(__m256i acc, __m256i a, __m256i b)
    {
	return _mm256_dpwssd_avx_epi32(acc, a, b);
    }
};

//  the 8 rows of a column from the two accumulators (pairs of lanes)
inline __m256i
i8column(__m256i lo, __m256i hi)
{
    //  hadd: lo pairs 0..3 | hi pairs 0..3 in the low 128 bits, lo pairs 4..7
    //  | hi pairs 4..7 in the high 128 bits: rows 0 1 4 5 | 2 3 6 7
    const __m256i h = _mm256_hadd_epi32(lo, hi);
    return _mm256_permute4x64_epi64(h, 0xD8); // rows 0 1 2 3 | 4 5 6 7
}

inline void
i8store8(__m256i acc, std::int32_t alpha, std::int32_t beta, std::int32_t *C)
{
    if (alpha != 1) {
	acc = _mm256_mullo_epi32(acc, _mm256_set1_epi32(alpha));
    }
    if (beta == 1) {
	acc = _mm256_add_epi32(acc, _mm256_loadu_si256((const __m256i *)C));
    } else if (beta != 0) {
	acc = _mm256_add_epi32(
	    acc, _mm256_mullo_epi32(_mm256_loadu_si256((const __m256i *)C),
	                            _mm256_set1_epi32(beta)));
    }
    _mm256_storeu_si256((__m256i *)C, acc);
}

#define ULMBLAS_I8_COL(j)                                                      \
    {                                                                          \
	const __m256i b = _mm256_set1_epi64x(*(const long long *)(B + 4 * j)); \
	c##j##0 = M::madd(c##j##0, a0, b);                                     \
	c##j##1 = M::madd(c##j##1, a1, b);                                     \
    }

#define ULMBLAS_I8_KERNEL(NAME, TARGET)                                        \
    template <typename M>                                                      \
    TARGET void NAME(long kg, std::int32_t alpha, const std::int16_t *A,       \
                     const std::int16_t *B, std::int32_t beta,                 \
                     std::int32_t *C, long ldC)                                \
    {                                                                          \
	__m256i c00 = _mm256_setzero_si256(), c01 = c00, c10 = c00, c11 = c00, \
	        c20 = c00, c21 = c00, c30 = c00, c31 = c00, c40 = c00,         \
	        c41 = c00, c50 = c00, c51 = c00;                               \
	for (long g = 0; g < kg; ++g) {                                        \
	    const __m256i a0 = _mm256_loadu_si256((const __m256i *)A);         \
	    const __m256i a1 = _mm256_loadu_si256((const __m256i *)(A + 16));  \
	    ULMBLAS_I8_COL(0)                                                  \
	    ULMBLAS_I8_COL(1)                                                  \
	    ULMBLAS_I8_COL(2)                                                  \
	    ULMBLAS_I8_COL(3)                                                  \
	    ULMBLAS_I8_COL(4)                                                  \
	    ULMBLAS_I8_COL(5)                                                  \
	    A += 32;                                                           \
	    B += 24;                                                           \
	}                                                                      \
	i8store8(i8column(c00, c01), alpha, beta, C);                          \
	i8store8(i8column(c10, c11), alpha, beta, C + ldC);                    \
	i8store8(i8column(c20, c21), alpha, beta, C + 2 * ldC);                \
	i8store8(i8column(c30, c31), alpha, beta, C + 3 * ldC);                \
	i8store8(i8column(c40, c41), alpha, beta, C + 4 * ldC);                \
	i8store8(i8column(c50, c51), alpha, beta, C + 5 * ldC);                \
    }

ULMBLAS_I8_KERNEL(i8kernel_8x6, )
ULMBLAS_I8_KERNEL(i8kernel_8x6_vnni, ULMBLAS_AVX_VNNI)

#undef ULMBLAS_I8_KERNEL
#undef ULMBLAS_I8_COL

//  for the driver of level3/i8gemm/driver.h; Vnni: VPDPWSSD
template <bool Vnni> struct I8Kernel
{
    typedef std::int16_t P;
    static constexpr long MR = 8, NR = 6, KG = 4;
    static constexpr long MC = 192, KC = 1024, NC = 3072;
    static constexpr bool partial = false;

    template <bool UA, bool UB>
    static void
    kernel(long kg, std::int32_t alpha, const P *A, const P *B,
           std::int32_t beta, std::int32_t *C, long ldC, long, long)
    {
	run(std::integral_constant<bool, Vnni>(), kg, alpha, A, B, beta, C,
	    ldC);
    }

    static void
    run(std::false_type, long kg, std::int32_t alpha, const P *A, const P *B,
        std::int32_t beta, std::int32_t *C, long ldC)
    {
	i8kernel_8x6<MaddPlain>(kg, alpha, A, B, beta, C, ldC);
    }

    static void
    run(std::true_type, long kg, std::int32_t alpha, const P *A, const P *B,
        std::int32_t beta, std::int32_t *C, long ldC)
    {
	i8kernel_8x6_vnni<MaddVnni>(kg, alpha, A, B, beta, C, ldC);
    }
};

} // namespace avx
} // namespace ulmBLAS

#undef ULMBLAS_AVX_VNNI

#endif // ULMBLAS_LEVEL3_UKERNEL_AVX_I8GEMM_TCC
