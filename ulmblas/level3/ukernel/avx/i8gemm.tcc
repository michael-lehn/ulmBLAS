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
//  The loop over k is inline assembly (GCC, clang).  CPUs with VNNI use the
//  kernels of i8gemm_busd.tcc.
//
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <type_traits>
#include <ulmblas/level3/i8gemm/driver.h>

namespace ulmBLAS {
namespace avx {

#define ULMBLAS_AVX_INLINE inline __attribute__((always_inline))

//  the 8 rows of a column from the two accumulators (pairs of lanes)
ULMBLAS_AVX_INLINE __m256i
i8column(__m256i lo, __m256i hi)
{
    //  hadd: lo pairs 0..3 | hi pairs 0..3 in the low 128 bits, lo pairs 4..7
    //  | hi pairs 4..7 in the high 128 bits: rows 0 1 4 5 | 2 3 6 7
    const __m256i h = _mm256_hadd_epi32(lo, hi);
    return _mm256_permute4x64_epi64(h, 0xD8); // rows 0 1 2 3 | 4 5 6 7
}

ULMBLAS_AVX_INLINE void
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

//
//  The loop over the groups in inline assembly: compilers (GCC 12) stored
//  the 12 accumulators to the stack in every iteration.  12 accumulators,
//  the two registers of A, the broadcast of B and a temporary: all 16
//  registers.
//
#define ULMBLAS_I8_DECLARE                                                     \
    __m256i c00 = _mm256_setzero_si256(), c01 = c00, c10 = c00, c11 = c00,     \
            c20 = c00, c21 = c00, c30 = c00, c31 = c00, c40 = c00, c41 = c00,  \
            c50 = c00, c51 = c00;                                              \
    __m256i a0, a1, b;

#define ULMBLAS_I8_FINISH                                                      \
    i8store8(i8column(c00, c01), alpha, beta, C);                              \
    i8store8(i8column(c10, c11), alpha, beta, C + ldC);                        \
    i8store8(i8column(c20, c21), alpha, beta, C + 2 * ldC);                    \
    i8store8(i8column(c30, c31), alpha, beta, C + 3 * ldC);                    \
    i8store8(i8column(c40, c41), alpha, beta, C + 4 * ldC);                    \
    i8store8(i8column(c50, c51), alpha, beta, C + 5 * ldC);

inline void
i8kernel_8x6(long kg, std::int32_t alpha, const std::int16_t *A,
             const std::int16_t *B, std::int32_t beta, std::int32_t *C,
             long ldC)
{
    ULMBLAS_I8_DECLARE
    __m256i t;
    for (long g = 0; g < kg; ++g) {
	__asm__("vmovdqu (%[A]), %[a0]\n\t"
		    "vmovdqu 32(%[A]), %[a1]\n\t"
		    "vpbroadcastq 0(%[B]), %[b]\n\t"
		    "vpmaddwd %[b], %[a0], %[t]\n\t"
		    "vpaddd %[t], %[c00], %[c00]\n\t"
		    "vpmaddwd %[b], %[a1], %[t]\n\t"
		    "vpaddd %[t], %[c01], %[c01]\n\t"
		    "vpbroadcastq 8(%[B]), %[b]\n\t"
		    "vpmaddwd %[b], %[a0], %[t]\n\t"
		    "vpaddd %[t], %[c10], %[c10]\n\t"
		    "vpmaddwd %[b], %[a1], %[t]\n\t"
		    "vpaddd %[t], %[c11], %[c11]\n\t"
		    "vpbroadcastq 16(%[B]), %[b]\n\t"
		    "vpmaddwd %[b], %[a0], %[t]\n\t"
		    "vpaddd %[t], %[c20], %[c20]\n\t"
		    "vpmaddwd %[b], %[a1], %[t]\n\t"
		    "vpaddd %[t], %[c21], %[c21]\n\t"
		    "vpbroadcastq 24(%[B]), %[b]\n\t"
		    "vpmaddwd %[b], %[a0], %[t]\n\t"
		    "vpaddd %[t], %[c30], %[c30]\n\t"
		    "vpmaddwd %[b], %[a1], %[t]\n\t"
		    "vpaddd %[t], %[c31], %[c31]\n\t"
		    "vpbroadcastq 32(%[B]), %[b]\n\t"
		    "vpmaddwd %[b], %[a0], %[t]\n\t"
		    "vpaddd %[t], %[c40], %[c40]\n\t"
		    "vpmaddwd %[b], %[a1], %[t]\n\t"
		    "vpaddd %[t], %[c41], %[c41]\n\t"
		    "vpbroadcastq 40(%[B]), %[b]\n\t"
		    "vpmaddwd %[b], %[a0], %[t]\n\t"
		    "vpaddd %[t], %[c50], %[c50]\n\t"
		    "vpmaddwd %[b], %[a1], %[t]\n\t"
		    "vpaddd %[t], %[c51], %[c51]\n\t"
	        : [c00] "+x"(c00), [c01] "+x"(c01), [c10] "+x"(c10), [c11] "+x"(c11), [c20] "+x"(c20), [c21] "+x"(c21), [c30] "+x"(c30), [c31] "+x"(c31), [c40] "+x"(c40), [c41] "+x"(c41), [c50] "+x"(c50), [c51] "+x"(c51), [a0] "=&x"(a0), [a1] "=&x"(a1), [b] "=&x"(b),
	          [t] "=&x"(t)
	        : [A] "r"(A), [B] "r"(B)
	        : "memory");
	A += 32;
	B += 24;
    }
    ULMBLAS_I8_FINISH
}


#undef ULMBLAS_I8_DECLARE
#undef ULMBLAS_I8_FINISH

//  for the driver of level3/i8gemm/driver.h
struct I8Kernel : i8::KernelBase
{
    typedef std::int16_t P;
    static constexpr long MR = 8, NR = 6, KG = 4;
    static constexpr long MC = 192, KC = 1024, NC = 3072;

    template <bool UA, bool UB>
    static void
    kernel(long kg, std::int32_t alpha, const P *A, const P *B,
           std::int32_t beta, std::int32_t *C, long ldC, long, long)
    {
	i8kernel_8x6(kg, alpha, A, B, beta, C, ldC);
    }
};

} // namespace avx
} // namespace ulmBLAS

#undef ULMBLAS_AVX_INLINE

#endif // ULMBLAS_LEVEL3_UKERNEL_AVX_I8GEMM_TCC
