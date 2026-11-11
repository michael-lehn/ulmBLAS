#ifndef ULMBLAS_LEVEL3_UKERNEL_AVX_I8GEMM_BUSD_TCC
#define ULMBLAS_LEVEL3_UKERNEL_AVX_I8GEMM_BUSD_TCC 1

//
//  ulmBLAS: micro kernels of the 8-bit integer GEMM (level3/i8gemm/driver.h)
//  with VPDPBUSD (4 products of an unsigned and a signed byte summed into
//  each int32 lane, twice the products per instruction of the int16 kernel
//  of i8gemm.tcc):
//
//      AVX-VNNI (ymm, e.g. Intel Alder Lake)   16 x 6 tile, 12 accumulators
//      AVX512-VNNI (zmm, Ice Lake, Zen 4)      32 x 12 tile, 24 accumulators
//
//  The panels are bytes in groups of 4 values of k.  Per group: two loads of
//  A (the rows), per column of B a broadcast of its 4 bytes and two
//  VPDPBUSD; the lanes of an accumulator are rows of a column of C.  Signs:
//
//      A unsigned, B signed     VPDPBUSD with A unsigned, B signed
//      A signed, B unsigned     the operands swapped
//      A, B signed              A + 128 (bit 7 flipped when packed) is
//                               unsigned: (A + 128) B - 128 colsum(B)
//      A, B unsigned            B - 128 is signed: A (B - 128) + 128
//                               rowsum(A)
//
//  The column sums of a packed panel of B (row sums of a panel of A), times
//  128, are computed after packing (prepareB, prepareA) and stored behind
//  the panel;
//  a panel is used for many tiles.  Everything modulo 2^32.  The loop over
//  k is inline assembly (see i8gemm.tcc).
//
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <type_traits>
#include <ulmblas/level3/i8gemm/driver.h>

#if defined(__GNUC__)
#include <cpuid.h>
#endif

//  AVX-VNNI (the VEX form): clang 12; GCC 11 knows it, but its assembler
//  needs binutils 2.36 (RHEL 9: GCC 11.5 with binutils 2.35 rejects it), so
//  GCC 12 (with binutils 2.38 or newer in the distributions and conda-forge).
//  AVX512-VNNI (EVEX, binutils 2.30): with AVX-512F (the avx512 backend).
#if (defined(__clang__) && __clang_major__ >= 12) ||                           \
    (!defined(__clang__) && defined(__GNUC__) && __GNUC__ >= 12)
#define ULMBLAS_HAVE_AVXVNNI_BUSD 1
#endif
#if defined(__AVX512F__)
#define ULMBLAS_HAVE_AVX512VNNI_BUSD 1
#endif

namespace ulmBLAS {
namespace avx {

//  -DULMBLAS_I8GEMM_NO_VNNI: no VNNI kernels (tests of the AVX2 kernel)
inline bool
hasAvxVnniBusd()
{
#if defined(ULMBLAS_I8GEMM_NO_VNNI) || !defined(ULMBLAS_HAVE_AVXVNNI_BUSD)
    return false;
#else
    static const bool has = [] {
	unsigned a, b, c, d;
	return __get_cpuid_count(7, 1, &a, &b, &c, &d) && (a & (1u << 4));
    }();
    return has; // CPUID.(7,1):EAX[4] AVX-VNNI
#endif
}

inline bool
hasAvx512Vnni()
{
#if defined(ULMBLAS_I8GEMM_NO_VNNI) || !defined(ULMBLAS_HAVE_AVX512VNNI_BUSD)
    return false;
#else
    static const bool has = [] {
	unsigned a, b, c, d;
	return __get_cpuid_count(7, 0, &a, &b, &c, &d) && (c & (1u << 11));
    }();
    return has; // CPUID.(7,0):ECX[11] AVX512_VNNI (the backend has AVX-512F)
#endif
}

//
//  Sums behind the panels and the bias of bit 7 (see above), for panels of
//  N rows (columns) in groups of 4 bytes.
//
template <long N>
void
busdBias(long kg, std::uint8_t *panel)
{
    for (long q = 0; q < 4 * N * kg; ++q) {
	panel[q] ^= 0x80;
    }
}

template <long N, bool U>
void
busdSums(long kg, std::uint8_t *panel)
{
    std::int32_t s[N] = {0};
    for (long g = 0; g < kg; ++g) {
	const std::uint8_t *p = panel + 4 * N * g;
	for (long i = 0; i < N; ++i) {
	    for (long q = 0; q < 4; ++q) {
		s[i] += U ? std::int32_t(p[4 * i + q])
		          : std::int32_t(std::int8_t(p[4 * i + q]));
	    }
	}
    }
    //  stored times 128 (the correction), modulo 2^32
    for (long i = 0; i < N; ++i) {
	s[i] = std::int32_t(std::uint32_t(s[i]) << 7);
    }
    std::memcpy(panel + 4 * N * kg, s, sizeof(s));
}

template <long MR_, long NR_> struct BusdBase : i8::KernelBase
{
    typedef std::uint8_t P;
    static constexpr long MR = MR_, NR = NR_, KG = 4;
    static constexpr long extraA = 4 * MR, extraB = 4 * NR;

    //  signed x signed: A biased; unsigned x unsigned: row sums of A
    template <bool UA, bool UB>
    static void
    prepareA(long kg, P *panel)
    {
	if (!UA && !UB) {
	    busdBias<MR>(kg, panel);
	} else if (UA && UB) {
	    busdSums<MR, true>(kg, panel);
	}
    }

    //  signed x signed: column sums of B; unsigned x unsigned: B biased
    template <bool UA, bool UB>
    static void
    prepareB(long kg, P *panel)
    {
	if (!UA && !UB) {
	    busdSums<NR, false>(kg, panel);
	} else if (UA && UB) {
	    busdBias<NR>(kg, panel);
	}
    }
};

#if defined(ULMBLAS_HAVE_AVXVNNI_BUSD)

#define ULMBLAS_BUSD_YMM __attribute__((target("avx2,avxvnni")))

//  C (8 rows) <- alpha*acc + beta*C
ULMBLAS_BUSD_YMM inline void
busdStore8(__m256i acc, std::int32_t alpha, std::int32_t beta, std::int32_t *C)
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

//  SA: A signed and B unsigned (operands swapped); CORR: 0 none, 1 minus
//  128 colsum(B), 2 plus 128 rowsum(A)
template <bool SA, int CORR>
ULMBLAS_BUSD_YMM void
busdKernel16x6(long kg, std::int32_t alpha, const std::uint8_t *A,
               const std::uint8_t *B, std::int32_t beta, std::int32_t *C,
               long ldC)
{
    const std::uint8_t *sumA = A + 64 * kg, *sumB = B + 24 * kg;
    const __m256i z = _mm256_setzero_si256();
    __m256i c0_0 = z, c0_1 = z, c1_0 = z, c1_1 = z, c2_0 = z, c2_1 = z,
            c3_0 = z, c3_1 = z, c4_0 = z, c4_1 = z, c5_0 = z, c5_1 = z;
    __m256i a0, a1, b;
    for (long g = 0; g < kg; ++g) {
	if (SA) {
	    __asm__("vmovdqu (%[A]), %[a0]\n\t"
	            "vmovdqu 32(%[A]), %[a1]\n\t"
	            "vpbroadcastd 0(%[B]), %[b]\n\t"
	            "%{vex%} vpdpbusd %[a0], %[b], %[c0_0]\n\t"
	            "%{vex%} vpdpbusd %[a1], %[b], %[c0_1]\n\t"
	            "vpbroadcastd 4(%[B]), %[b]\n\t"
	            "%{vex%} vpdpbusd %[a0], %[b], %[c1_0]\n\t"
	            "%{vex%} vpdpbusd %[a1], %[b], %[c1_1]\n\t"
	            "vpbroadcastd 8(%[B]), %[b]\n\t"
	            "%{vex%} vpdpbusd %[a0], %[b], %[c2_0]\n\t"
	            "%{vex%} vpdpbusd %[a1], %[b], %[c2_1]\n\t"
	            "vpbroadcastd 12(%[B]), %[b]\n\t"
	            "%{vex%} vpdpbusd %[a0], %[b], %[c3_0]\n\t"
	            "%{vex%} vpdpbusd %[a1], %[b], %[c3_1]\n\t"
	            "vpbroadcastd 16(%[B]), %[b]\n\t"
	            "%{vex%} vpdpbusd %[a0], %[b], %[c4_0]\n\t"
	            "%{vex%} vpdpbusd %[a1], %[b], %[c4_1]\n\t"
	            "vpbroadcastd 20(%[B]), %[b]\n\t"
	            "%{vex%} vpdpbusd %[a0], %[b], %[c5_0]\n\t"
	            "%{vex%} vpdpbusd %[a1], %[b], %[c5_1]\n\t"
	            : [c0_0] "+x"(c0_0), [c0_1] "+x"(c0_1), [c1_0] "+x"(c1_0),
	              [c1_1] "+x"(c1_1), [c2_0] "+x"(c2_0), [c2_1] "+x"(c2_1),
	              [c3_0] "+x"(c3_0), [c3_1] "+x"(c3_1), [c4_0] "+x"(c4_0),
	              [c4_1] "+x"(c4_1), [c5_0] "+x"(c5_0), [c5_1] "+x"(c5_1),
	              [a0] "=&x"(a0), [a1] "=&x"(a1), [b] "=&x"(b)
	            : [A] "r"(A), [B] "r"(B)
	            : "memory");
	} else {
	    __asm__("vmovdqu (%[A]), %[a0]\n\t"
	            "vmovdqu 32(%[A]), %[a1]\n\t"
	            "vpbroadcastd 0(%[B]), %[b]\n\t"
	            "%{vex%} vpdpbusd %[b], %[a0], %[c0_0]\n\t"
	            "%{vex%} vpdpbusd %[b], %[a1], %[c0_1]\n\t"
	            "vpbroadcastd 4(%[B]), %[b]\n\t"
	            "%{vex%} vpdpbusd %[b], %[a0], %[c1_0]\n\t"
	            "%{vex%} vpdpbusd %[b], %[a1], %[c1_1]\n\t"
	            "vpbroadcastd 8(%[B]), %[b]\n\t"
	            "%{vex%} vpdpbusd %[b], %[a0], %[c2_0]\n\t"
	            "%{vex%} vpdpbusd %[b], %[a1], %[c2_1]\n\t"
	            "vpbroadcastd 12(%[B]), %[b]\n\t"
	            "%{vex%} vpdpbusd %[b], %[a0], %[c3_0]\n\t"
	            "%{vex%} vpdpbusd %[b], %[a1], %[c3_1]\n\t"
	            "vpbroadcastd 16(%[B]), %[b]\n\t"
	            "%{vex%} vpdpbusd %[b], %[a0], %[c4_0]\n\t"
	            "%{vex%} vpdpbusd %[b], %[a1], %[c4_1]\n\t"
	            "vpbroadcastd 20(%[B]), %[b]\n\t"
	            "%{vex%} vpdpbusd %[b], %[a0], %[c5_0]\n\t"
	            "%{vex%} vpdpbusd %[b], %[a1], %[c5_1]\n\t"
	            : [c0_0] "+x"(c0_0), [c0_1] "+x"(c0_1), [c1_0] "+x"(c1_0),
	              [c1_1] "+x"(c1_1), [c2_0] "+x"(c2_0), [c2_1] "+x"(c2_1),
	              [c3_0] "+x"(c3_0), [c3_1] "+x"(c3_1), [c4_0] "+x"(c4_0),
	              [c4_1] "+x"(c4_1), [c5_0] "+x"(c5_0), [c5_1] "+x"(c5_1),
	              [a0] "=&x"(a0), [a1] "=&x"(a1), [b] "=&x"(b)
	            : [A] "r"(A), [B] "r"(B)
	            : "memory");
	}
	A += 64;
	B += 24;
    }
    const __m256i acc[6][2] = {{c0_0, c0_1}, {c1_0, c1_1}, {c2_0, c2_1},
                               {c3_0, c3_1}, {c4_0, c4_1}, {c5_0, c5_1}};
    __m256i r0 = _mm256_setzero_si256(), r1 = r0;
    if (CORR == 2) {
	r0 = _mm256_loadu_si256((const __m256i *)sumA);
	r1 = _mm256_loadu_si256((const __m256i *)(sumA + 32));
    }
    for (int j = 0; j < 6; ++j) {
	__m256i x0 = acc[j][0], x1 = acc[j][1];
	if (CORR == 1) {
	    std::int32_t s;
	    std::memcpy(&s, sumB + 4 * j, 4);
	    const __m256i t = _mm256_set1_epi32(s);
	    x0 = _mm256_sub_epi32(x0, t);
	    x1 = _mm256_sub_epi32(x1, t);
	} else if (CORR == 2) {
	    x0 = _mm256_add_epi32(x0, r0);
	    x1 = _mm256_add_epi32(x1, r1);
	}
	busdStore8(x0, alpha, beta, C + j * ldC);
	busdStore8(x1, alpha, beta, C + j * ldC + 8);
    }
}

struct I8KernelBusd16x6 : BusdBase<16, 6>
{
    static constexpr long MC = 192, KC = 2048, NC = 3072;

    template <bool UA, bool UB>
    static void
    kernel(long kg, std::int32_t alpha, const P *A, const P *B,
           std::int32_t beta, std::int32_t *C, long ldC, long, long)
    {
	busdKernel16x6<!UA && UB, (!UA && !UB) ? 1
	                          : (UA && UB) ? 2
	                                       : 0>(kg, alpha, A, B, beta, C,
	                                            ldC);
    }
};

#undef ULMBLAS_BUSD_YMM
#endif // ULMBLAS_HAVE_AVXVNNI_BUSD

#if defined(ULMBLAS_HAVE_AVX512VNNI_BUSD)

#define ULMBLAS_BUSD_ZMM __attribute__((target("avx512f,avx512vnni")))

ULMBLAS_BUSD_ZMM inline void
busdStore16(__m512i acc, std::int32_t alpha, std::int32_t beta, std::int32_t *C)
{
    if (alpha != 1) {
	acc = _mm512_mullo_epi32(acc, _mm512_set1_epi32(alpha));
    }
    if (beta == 1) {
	acc = _mm512_add_epi32(acc, _mm512_loadu_si512(C));
    } else if (beta != 0) {
	acc =
	    _mm512_add_epi32(acc, _mm512_mullo_epi32(_mm512_loadu_si512(C),
	                                             _mm512_set1_epi32(beta)));
    }
    _mm512_storeu_si512(C, acc);
}

template <bool SA, int CORR>
ULMBLAS_BUSD_ZMM void
busdKernel32x12(long kg, std::int32_t alpha, const std::uint8_t *A,
                const std::uint8_t *B, std::int32_t beta, std::int32_t *C,
                long ldC)
{
    const std::uint8_t *sumA = A + 128 * kg, *sumB = B + 48 * kg;
    const __m512i z = _mm512_setzero_si512();
    __m512i c0_0 = z, c0_1 = z, c1_0 = z, c1_1 = z, c2_0 = z, c2_1 = z,
            c3_0 = z, c3_1 = z, c4_0 = z, c4_1 = z, c5_0 = z, c5_1 = z,
            c6_0 = z, c6_1 = z, c7_0 = z, c7_1 = z, c8_0 = z, c8_1 = z,
            c9_0 = z, c9_1 = z, c10_0 = z, c10_1 = z, c11_0 = z, c11_1 = z;
    __m512i a0, a1, b;
    for (long g = 0; g < kg; ++g) {
	if (SA) {
	    __asm__("vmovdqu32 (%[A]), %[a0]\n\t"
	            "vmovdqu32 64(%[A]), %[a1]\n\t"
	            "vpbroadcastd 0(%[B]), %[b]\n\t"
	            "vpdpbusd %[a0], %[b], %[c0_0]\n\t"
	            "vpdpbusd %[a1], %[b], %[c0_1]\n\t"
	            "vpbroadcastd 4(%[B]), %[b]\n\t"
	            "vpdpbusd %[a0], %[b], %[c1_0]\n\t"
	            "vpdpbusd %[a1], %[b], %[c1_1]\n\t"
	            "vpbroadcastd 8(%[B]), %[b]\n\t"
	            "vpdpbusd %[a0], %[b], %[c2_0]\n\t"
	            "vpdpbusd %[a1], %[b], %[c2_1]\n\t"
	            "vpbroadcastd 12(%[B]), %[b]\n\t"
	            "vpdpbusd %[a0], %[b], %[c3_0]\n\t"
	            "vpdpbusd %[a1], %[b], %[c3_1]\n\t"
	            "vpbroadcastd 16(%[B]), %[b]\n\t"
	            "vpdpbusd %[a0], %[b], %[c4_0]\n\t"
	            "vpdpbusd %[a1], %[b], %[c4_1]\n\t"
	            "vpbroadcastd 20(%[B]), %[b]\n\t"
	            "vpdpbusd %[a0], %[b], %[c5_0]\n\t"
	            "vpdpbusd %[a1], %[b], %[c5_1]\n\t"
	            : [c0_0] "+v"(c0_0), [c0_1] "+v"(c0_1), [c1_0] "+v"(c1_0),
	              [c1_1] "+v"(c1_1), [c2_0] "+v"(c2_0), [c2_1] "+v"(c2_1),
	              [c3_0] "+v"(c3_0), [c3_1] "+v"(c3_1), [c4_0] "+v"(c4_0),
	              [c4_1] "+v"(c4_1), [c5_0] "+v"(c5_0), [c5_1] "+v"(c5_1),
	              [a0] "=&v"(a0), [a1] "=&v"(a1), [b] "=&v"(b)
	            : [A] "r"(A), [B] "r"(B)
	            : "memory");
	    __asm__("vpbroadcastd 24(%[B]), %[b]\n\t"
	            "vpdpbusd %[a0], %[b], %[c6_0]\n\t"
	            "vpdpbusd %[a1], %[b], %[c6_1]\n\t"
	            "vpbroadcastd 28(%[B]), %[b]\n\t"
	            "vpdpbusd %[a0], %[b], %[c7_0]\n\t"
	            "vpdpbusd %[a1], %[b], %[c7_1]\n\t"
	            "vpbroadcastd 32(%[B]), %[b]\n\t"
	            "vpdpbusd %[a0], %[b], %[c8_0]\n\t"
	            "vpdpbusd %[a1], %[b], %[c8_1]\n\t"
	            "vpbroadcastd 36(%[B]), %[b]\n\t"
	            "vpdpbusd %[a0], %[b], %[c9_0]\n\t"
	            "vpdpbusd %[a1], %[b], %[c9_1]\n\t"
	            "vpbroadcastd 40(%[B]), %[b]\n\t"
	            "vpdpbusd %[a0], %[b], %[c10_0]\n\t"
	            "vpdpbusd %[a1], %[b], %[c10_1]\n\t"
	            "vpbroadcastd 44(%[B]), %[b]\n\t"
	            "vpdpbusd %[a0], %[b], %[c11_0]\n\t"
	            "vpdpbusd %[a1], %[b], %[c11_1]\n\t"
	            : [c6_0] "+v"(c6_0), [c6_1] "+v"(c6_1), [c7_0] "+v"(c7_0),
	              [c7_1] "+v"(c7_1), [c8_0] "+v"(c8_0), [c8_1] "+v"(c8_1),
	              [c9_0] "+v"(c9_0), [c9_1] "+v"(c9_1), [c10_0] "+v"(c10_0),
	              [c10_1] "+v"(c10_1), [c11_0] "+v"(c11_0),
	              [c11_1] "+v"(c11_1), [b] "=&v"(b)
	            : [A] "r"(A), [B] "r"(B), [a0] "v"(a0), [a1] "v"(a1)
	            : "memory");
	} else {
	    __asm__("vmovdqu32 (%[A]), %[a0]\n\t"
	            "vmovdqu32 64(%[A]), %[a1]\n\t"
	            "vpbroadcastd 0(%[B]), %[b]\n\t"
	            "vpdpbusd %[b], %[a0], %[c0_0]\n\t"
	            "vpdpbusd %[b], %[a1], %[c0_1]\n\t"
	            "vpbroadcastd 4(%[B]), %[b]\n\t"
	            "vpdpbusd %[b], %[a0], %[c1_0]\n\t"
	            "vpdpbusd %[b], %[a1], %[c1_1]\n\t"
	            "vpbroadcastd 8(%[B]), %[b]\n\t"
	            "vpdpbusd %[b], %[a0], %[c2_0]\n\t"
	            "vpdpbusd %[b], %[a1], %[c2_1]\n\t"
	            "vpbroadcastd 12(%[B]), %[b]\n\t"
	            "vpdpbusd %[b], %[a0], %[c3_0]\n\t"
	            "vpdpbusd %[b], %[a1], %[c3_1]\n\t"
	            "vpbroadcastd 16(%[B]), %[b]\n\t"
	            "vpdpbusd %[b], %[a0], %[c4_0]\n\t"
	            "vpdpbusd %[b], %[a1], %[c4_1]\n\t"
	            "vpbroadcastd 20(%[B]), %[b]\n\t"
	            "vpdpbusd %[b], %[a0], %[c5_0]\n\t"
	            "vpdpbusd %[b], %[a1], %[c5_1]\n\t"
	            : [c0_0] "+v"(c0_0), [c0_1] "+v"(c0_1), [c1_0] "+v"(c1_0),
	              [c1_1] "+v"(c1_1), [c2_0] "+v"(c2_0), [c2_1] "+v"(c2_1),
	              [c3_0] "+v"(c3_0), [c3_1] "+v"(c3_1), [c4_0] "+v"(c4_0),
	              [c4_1] "+v"(c4_1), [c5_0] "+v"(c5_0), [c5_1] "+v"(c5_1),
	              [a0] "=&v"(a0), [a1] "=&v"(a1), [b] "=&v"(b)
	            : [A] "r"(A), [B] "r"(B)
	            : "memory");
	    __asm__("vpbroadcastd 24(%[B]), %[b]\n\t"
	            "vpdpbusd %[b], %[a0], %[c6_0]\n\t"
	            "vpdpbusd %[b], %[a1], %[c6_1]\n\t"
	            "vpbroadcastd 28(%[B]), %[b]\n\t"
	            "vpdpbusd %[b], %[a0], %[c7_0]\n\t"
	            "vpdpbusd %[b], %[a1], %[c7_1]\n\t"
	            "vpbroadcastd 32(%[B]), %[b]\n\t"
	            "vpdpbusd %[b], %[a0], %[c8_0]\n\t"
	            "vpdpbusd %[b], %[a1], %[c8_1]\n\t"
	            "vpbroadcastd 36(%[B]), %[b]\n\t"
	            "vpdpbusd %[b], %[a0], %[c9_0]\n\t"
	            "vpdpbusd %[b], %[a1], %[c9_1]\n\t"
	            "vpbroadcastd 40(%[B]), %[b]\n\t"
	            "vpdpbusd %[b], %[a0], %[c10_0]\n\t"
	            "vpdpbusd %[b], %[a1], %[c10_1]\n\t"
	            "vpbroadcastd 44(%[B]), %[b]\n\t"
	            "vpdpbusd %[b], %[a0], %[c11_0]\n\t"
	            "vpdpbusd %[b], %[a1], %[c11_1]\n\t"
	            : [c6_0] "+v"(c6_0), [c6_1] "+v"(c6_1), [c7_0] "+v"(c7_0),
	              [c7_1] "+v"(c7_1), [c8_0] "+v"(c8_0), [c8_1] "+v"(c8_1),
	              [c9_0] "+v"(c9_0), [c9_1] "+v"(c9_1), [c10_0] "+v"(c10_0),
	              [c10_1] "+v"(c10_1), [c11_0] "+v"(c11_0),
	              [c11_1] "+v"(c11_1), [b] "=&v"(b)
	            : [A] "r"(A), [B] "r"(B), [a0] "v"(a0), [a1] "v"(a1)
	            : "memory");
	}
	A += 128;
	B += 48;
    }
    const __m512i acc[12][2] = {{c0_0, c0_1}, {c1_0, c1_1},   {c2_0, c2_1},
                                {c3_0, c3_1}, {c4_0, c4_1},   {c5_0, c5_1},
                                {c6_0, c6_1}, {c7_0, c7_1},   {c8_0, c8_1},
                                {c9_0, c9_1}, {c10_0, c10_1}, {c11_0, c11_1}};
    __m512i r0 = _mm512_setzero_si512(), r1 = r0;
    if (CORR == 2) {
	r0 = _mm512_loadu_si512(sumA);
	r1 = _mm512_loadu_si512(sumA + 64);
    }
    for (int j = 0; j < 12; ++j) {
	__m512i x0 = acc[j][0], x1 = acc[j][1];
	if (CORR == 1) {
	    std::int32_t s;
	    std::memcpy(&s, sumB + 4 * j, 4);
	    const __m512i t = _mm512_set1_epi32(s);
	    x0 = _mm512_sub_epi32(x0, t);
	    x1 = _mm512_sub_epi32(x1, t);
	} else if (CORR == 2) {
	    x0 = _mm512_add_epi32(x0, r0);
	    x1 = _mm512_add_epi32(x1, r1);
	}
	busdStore16(x0, alpha, beta, C + j * ldC);
	busdStore16(x1, alpha, beta, C + j * ldC + 16);
    }
}

struct I8KernelBusd32x12 : BusdBase<32, 12>
{
    static constexpr long MC = 192, KC = 2048, NC = 3072;

    template <bool UA, bool UB>
    static void
    kernel(long kg, std::int32_t alpha, const P *A, const P *B,
           std::int32_t beta, std::int32_t *C, long ldC, long, long)
    {
	busdKernel32x12<!UA && UB, (!UA && !UB) ? 1
	                           : (UA && UB) ? 2
	                                        : 0>(kg, alpha, A, B, beta, C,
	                                             ldC);
    }
};

#undef ULMBLAS_BUSD_ZMM
#endif // ULMBLAS_HAVE_AVX512VNNI_BUSD

} // namespace avx
} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_UKERNEL_AVX_I8GEMM_BUSD_TCC
