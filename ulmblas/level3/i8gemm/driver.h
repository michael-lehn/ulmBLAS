#ifndef ULMBLAS_LEVEL3_I8GEMM_DRIVER_H
#define ULMBLAS_LEVEL3_I8GEMM_DRIVER_H 1

//
//  Blocked driver of the 8-bit (and 16-bit) integer GEMM (C int32, modulo
//  2^32):
//
//      C <- alpha*A*B + beta*C
//
//  for a micro kernel K (a struct with static members):
//
//      MR, NR          size of the tile of C
//      KG              values of k per group (the kernel multiplies groups)
//      P               element type of the packed panels: std::uint8_t (the
//                      bits of the int8 or uint8 values) or std::int16_t
//                      (the values, sign or zero extended)
//      MC, KC, NC      block sizes (MC, NC multiples of MR, NR; KC of KG)
//      partial         true: the kernel updates partial tiles itself
//      extraA, extraB  elements of P after each packed panel of A (B) for
//                      the kernel (e.g. sums of the panel), filled by
//      prepareA<UA, UB>(kg, panel), prepareB<UA, UB>(kg, panel)
//                      called after a panel is packed (may also modify it)
//
//  KernelBase provides partial = false, extraA = extraB = 0 and prepare
//  functions that do nothing.
//      kernel<UA, UB>(kg, alpha, A, B, beta, C, ldC, mr, nr)
//                      C (unit row stride, column stride ldC) <- alpha*A*B
//                      + beta*C for the rows < mr and columns < nr (all of
//                      them if not partial), beta = 0: C not read; UA, UB:
//                      A, B unsigned
//
//  Panels: for the panel of MR rows of A (NR columns of B) and the group g
//
//      buffer[MR*KG*g + KG*i + q] = A(i, KG*g + q),
//
//  zero padded in i and in k.  The products wrap around modulo 2^32, and
//  alpha*(S1 + S2) = alpha*S1 + alpha*S2 modulo 2^32: the blocks of k are
//  accumulated with alpha and beta = 1 after the first block.
//
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <ulmblas/auxiliary/memorypool.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#elif defined(__SSE2__)
#include <emmintrin.h>
#if defined(__SSE4_1__)
#include <smmintrin.h>
#endif
#endif

namespace ulmBLAS {
namespace i8 {

struct KernelBase
{
    static constexpr bool partial = false;
    static constexpr long extraA = 0, extraB = 0;

    template <bool UA, bool UB, typename P>
    static void
    prepareA(long, P *)
    {
    }

    template <bool UA, bool UB, typename P>
    static void
    prepareB(long, P *)
    {
    }
};

template <typename P, bool U>
inline P
packedValue(std::uint8_t x)
{
    return std::is_same<P, std::uint8_t>::value ? P(x)
           : U                                  ? P(x)
                                                : P(std::int8_t(x));
}

//  the value of an element: 8-bit integers (stored as bytes, U: unsigned),
//  16-bit integers (signed)
template <typename P, bool U>
inline P
packedValue(std::int16_t x)
{
    return P(x);
}

//  the groups g0, g0+1, ... of a panel, element by element (TX: the type of
//  the stored elements, std::uint8_t or std::int16_t)
template <long MR, long KG, typename P, bool U, typename TX>
void
packGeneric(long mn, long k, const TX *X, long incMN, long incK, long g0,
            P *buffer)
{
    const long kg = (k + KG - 1) / KG;
    for (long g = g0; g < kg; ++g) {
	P *b = buffer + MR * KG * g;
	for (long i = 0; i < MR; ++i) {
	    for (long q = 0; q < KG; ++q) {
		const long l = KG * g + q;
		b[KG * i + q] = (i < mn && l < k)
		                    ? packedValue<P, U>(X[i * incMN + l * incK])
		                    : P(0);
	    }
	}
    }
}

//
//  Fast paths for bytes in groups of 4 (P = std::uint8_t, KG = 4) and full
//  panels: incK = 1 (a 4 x 4 transposition of 32-bit words per 4 rows and
//  16 values of k), incMN = 1 (per group 4 columns of k interleaved).
//
#if defined(__aarch64__) || defined(__SSE2__)

template <long MR>
void
pack4KContig(long k, const std::uint8_t *X, long incMN, std::uint8_t *buffer)
{
    const long k16 = k / 16;
    long r = 0;
    for (; r + 4 <= MR; r += 4) {
	const std::uint8_t *x0 = X + r * incMN;
	const std::uint8_t *x1 = x0 + incMN;
	const std::uint8_t *x2 = x1 + incMN;
	const std::uint8_t *x3 = x2 + incMN;
	for (long c = 0; c < k16; ++c) {
	    std::uint8_t *b = buffer + 16 * MR * c + 4 * r;
#if defined(__aarch64__)
	    uint32x4_t q0 = vreinterpretq_u32_u8(vld1q_u8(x0 + 16 * c));
	    uint32x4_t q1 = vreinterpretq_u32_u8(vld1q_u8(x1 + 16 * c));
	    uint32x4_t q2 = vreinterpretq_u32_u8(vld1q_u8(x2 + 16 * c));
	    uint32x4_t q3 = vreinterpretq_u32_u8(vld1q_u8(x3 + 16 * c));
	    uint64x2_t t0 = vreinterpretq_u64_u32(vtrn1q_u32(q0, q1));
	    uint64x2_t t1 = vreinterpretq_u64_u32(vtrn2q_u32(q0, q1));
	    uint64x2_t t2 = vreinterpretq_u64_u32(vtrn1q_u32(q2, q3));
	    uint64x2_t t3 = vreinterpretq_u64_u32(vtrn2q_u32(q2, q3));
	    vst1q_u8(b, vreinterpretq_u8_u64(vtrn1q_u64(t0, t2)));
	    vst1q_u8(b + 4 * MR, vreinterpretq_u8_u64(vtrn1q_u64(t1, t3)));
	    vst1q_u8(b + 8 * MR, vreinterpretq_u8_u64(vtrn2q_u64(t0, t2)));
	    vst1q_u8(b + 12 * MR, vreinterpretq_u8_u64(vtrn2q_u64(t1, t3)));
#else
	    __m128i q0 = _mm_loadu_si128((const __m128i *)(x0 + 16 * c));
	    __m128i q1 = _mm_loadu_si128((const __m128i *)(x1 + 16 * c));
	    __m128i q2 = _mm_loadu_si128((const __m128i *)(x2 + 16 * c));
	    __m128i q3 = _mm_loadu_si128((const __m128i *)(x3 + 16 * c));
	    __m128i t0 = _mm_unpacklo_epi32(q0, q1); // w0 of q0 q1, w1 of q0 q1
	    __m128i t1 = _mm_unpackhi_epi32(q0, q1); // w2 ..., w3 ...
	    __m128i t2 = _mm_unpacklo_epi32(q2, q3);
	    __m128i t3 = _mm_unpackhi_epi32(q2, q3);
	    _mm_storeu_si128((__m128i *)b, _mm_unpacklo_epi64(t0, t2));
	    _mm_storeu_si128((__m128i *)(b + 4 * MR),
	                     _mm_unpackhi_epi64(t0, t2));
	    _mm_storeu_si128((__m128i *)(b + 8 * MR),
	                     _mm_unpacklo_epi64(t1, t3));
	    _mm_storeu_si128((__m128i *)(b + 12 * MR),
	                     _mm_unpackhi_epi64(t1, t3));
#endif
	}
    }
    //  the remaining rows (MR not a multiple of 4): per group 4 bytes
    for (; MR % 4 != 0 && r < MR; ++r) {
	for (long g = 0; g < 4 * k16; ++g) {
	    std::memcpy(buffer + 4 * MR * g + 4 * r, X + r * incMN + 4 * g, 4);
	}
    }
    packGeneric<MR, 4, std::uint8_t, false, std::uint8_t>(MR, k, X, incMN, 1,
                                                          4 * k16, buffer);
}

template <long MR>
void
pack4MNContig(long k, const std::uint8_t *X, long incK, std::uint8_t *buffer)
{
    const long k4 = k / 4;
    for (long g = 0; g < k4; ++g) {
	const std::uint8_t *x = X + 4 * g * incK;
	std::uint8_t *b = buffer + 4 * MR * g;
	long i = 0;
	for (; i + 16 <= MR; i += 16) {
#if defined(__aarch64__)
	    uint8x16_t c0 = vld1q_u8(x + i);
	    uint8x16_t c1 = vld1q_u8(x + incK + i);
	    uint8x16_t c2 = vld1q_u8(x + 2 * incK + i);
	    uint8x16_t c3 = vld1q_u8(x + 3 * incK + i);
	    uint16x8_t z01l = vreinterpretq_u16_u8(vzip1q_u8(c0, c1));
	    uint16x8_t z01h = vreinterpretq_u16_u8(vzip2q_u8(c0, c1));
	    uint16x8_t z23l = vreinterpretq_u16_u8(vzip1q_u8(c2, c3));
	    uint16x8_t z23h = vreinterpretq_u16_u8(vzip2q_u8(c2, c3));
	    std::uint8_t *bi = b + 4 * i;
	    vst1q_u8(bi, vreinterpretq_u8_u16(vzip1q_u16(z01l, z23l)));
	    vst1q_u8(bi + 16, vreinterpretq_u8_u16(vzip2q_u16(z01l, z23l)));
	    vst1q_u8(bi + 32, vreinterpretq_u8_u16(vzip1q_u16(z01h, z23h)));
	    vst1q_u8(bi + 48, vreinterpretq_u8_u16(vzip2q_u16(z01h, z23h)));
#else
	    __m128i c0 = _mm_loadu_si128((const __m128i *)(x + i));
	    __m128i c1 = _mm_loadu_si128((const __m128i *)(x + incK + i));
	    __m128i c2 = _mm_loadu_si128((const __m128i *)(x + 2 * incK + i));
	    __m128i c3 = _mm_loadu_si128((const __m128i *)(x + 3 * incK + i));
	    __m128i z01l = _mm_unpacklo_epi8(c0, c1);
	    __m128i z01h = _mm_unpackhi_epi8(c0, c1);
	    __m128i z23l = _mm_unpacklo_epi8(c2, c3);
	    __m128i z23h = _mm_unpackhi_epi8(c2, c3);
	    std::uint8_t *bi = b + 4 * i;
	    _mm_storeu_si128((__m128i *)bi, _mm_unpacklo_epi16(z01l, z23l));
	    _mm_storeu_si128((__m128i *)(bi + 16),
	                     _mm_unpackhi_epi16(z01l, z23l));
	    _mm_storeu_si128((__m128i *)(bi + 32),
	                     _mm_unpacklo_epi16(z01h, z23h));
	    _mm_storeu_si128((__m128i *)(bi + 48),
	                     _mm_unpackhi_epi16(z01h, z23h));
#endif
	}
	for (; i + 8 <= MR; i += 8) {
#if defined(__aarch64__)
	    uint8x8_t c0 = vld1_u8(x + i);
	    uint8x8_t c1 = vld1_u8(x + incK + i);
	    uint8x8_t c2 = vld1_u8(x + 2 * incK + i);
	    uint8x8_t c3 = vld1_u8(x + 3 * incK + i);
	    uint16x8_t z01 = vreinterpretq_u16_u8(
	        vcombine_u8(vzip1_u8(c0, c1), vzip2_u8(c0, c1)));
	    uint16x8_t z23 = vreinterpretq_u16_u8(
	        vcombine_u8(vzip1_u8(c2, c3), vzip2_u8(c2, c3)));
	    std::uint8_t *bi = b + 4 * i;
	    vst1q_u8(bi, vreinterpretq_u8_u16(vzip1q_u16(z01, z23)));
	    vst1q_u8(bi + 16, vreinterpretq_u8_u16(vzip2q_u16(z01, z23)));
#else
	    __m128i c0 = _mm_loadl_epi64((const __m128i *)(x + i));
	    __m128i c1 = _mm_loadl_epi64((const __m128i *)(x + incK + i));
	    __m128i c2 = _mm_loadl_epi64((const __m128i *)(x + 2 * incK + i));
	    __m128i c3 = _mm_loadl_epi64((const __m128i *)(x + 3 * incK + i));
	    __m128i z01 = _mm_unpacklo_epi8(c0, c1);
	    __m128i z23 = _mm_unpacklo_epi8(c2, c3);
	    std::uint8_t *bi = b + 4 * i;
	    _mm_storeu_si128((__m128i *)bi, _mm_unpacklo_epi16(z01, z23));
	    _mm_storeu_si128((__m128i *)(bi + 16),
	                     _mm_unpackhi_epi16(z01, z23));
#endif
	}
	for (; i < MR; ++i) {
	    for (long q = 0; q < 4; ++q) {
		b[4 * i + q] = x[q * incK + i];
	    }
	}
    }
    packGeneric<MR, 4, std::uint8_t, false, std::uint8_t>(MR, k, X, 1, incK, k4,
                                                          buffer);
}

#endif

//
//  Widening of a packed byte panel (P = std::uint8_t layout, KG = 4) to
//  int16 (the same layout with 16-bit elements), n bytes.
//
template <bool U>
void
widen(long n, const std::uint8_t *b, std::int16_t *w)
{
    long i = 0;
#if defined(__SSE4_1__)
    for (; i + 16 <= n; i += 16) {
	__m128i x = _mm_loadu_si128((const __m128i *)(b + i));
	__m128i lo = U ? _mm_cvtepu8_epi16(x) : _mm_cvtepi8_epi16(x);
	__m128i hi = U ? _mm_cvtepu8_epi16(_mm_srli_si128(x, 8))
	               : _mm_cvtepi8_epi16(_mm_srli_si128(x, 8));
	_mm_storeu_si128((__m128i *)(w + i), lo);
	_mm_storeu_si128((__m128i *)(w + i + 8), hi);
    }
#endif
    for (; i < n; ++i) {
	w[i] = U ? std::int16_t(b[i]) : std::int16_t(std::int8_t(b[i]));
    }
}

template <long MR, long KG, typename P, bool U>
void
packPanel(long mn, long k, const std::uint8_t *X, long incMN, long incK,
          P *buffer)
{
#if defined(__aarch64__) || defined(__SSE2__)
    if (std::is_same<P, std::uint8_t>::value && KG == 4 && mn == MR) {
	std::uint8_t *b = reinterpret_cast<std::uint8_t *>(buffer);
	if (incK == 1) {
	    pack4KContig<MR>(k, X, incMN, b);
	    return;
	}
	if (incMN == 1) {
	    pack4MNContig<MR>(k, X, incK, b);
	    return;
	}
    }
    if (std::is_same<P, std::int16_t>::value && KG == 4 && mn == MR &&
        (incK == 1 || incMN == 1)) {
	//  bytes first (fast paths), then widened, in chunks of groups
	constexpr long G = 64;
	std::uint8_t tmp[4 * MR * G];
	std::int16_t *w = reinterpret_cast<std::int16_t *>(buffer);
	for (long l0 = 0; l0 < k; l0 += 4 * G) {
	    const long kb = (k - l0 < 4 * G) ? k - l0 : 4 * G;
	    const long gb = (kb + 3) / 4;
	    if (incK == 1) {
		pack4KContig<MR>(kb, X + l0, incMN, tmp);
	    } else {
		pack4MNContig<MR>(kb, X + l0 * incK, incK, tmp);
	    }
	    widen<U>(4 * MR * gb, tmp, w + 4 * MR * (l0 / 4));
	}
	return;
    }
#endif
    packGeneric<MR, KG, P, U, std::uint8_t>(mn, k, X, incMN, incK, 0, buffer);
}

//  16-bit integers (signed): element by element
template <long MR, long KG, typename P, bool U>
void
packPanel(long mn, long k, const std::int16_t *X, long incMN, long incK,
          P *buffer)
{
    packGeneric<MR, KG, P, U, std::int16_t>(mn, k, X, incMN, incK, 0, buffer);
}

//  TX: the type of the elements of A and B (std::uint8_t: 8-bit integers
//  as bytes, UA, UB their signedness; std::int16_t)
template <typename K, bool UA, bool UB, typename TX = std::uint8_t>
void
driver(long m, long n, long k, std::int32_t alpha, const TX *A, long incRowA,
       long incColA, const TX *B, long incRowB, long incColB, std::int32_t beta,
       std::int32_t *C, long incRowC, long incColC)
{
    typedef typename K::P P;
    constexpr long MR = K::MR, NR = K::NR, KG = K::KG;
    constexpr long MC = K::MC, KC = K::KC, NC = K::NC;

    static_assert(MC % MR == 0 && NC % NR == 0 && KC % KG == 0,
                  "invalid block sizes");

    static MemoryPool<P> memoryPool;

    constexpr long XA = K::extraA, XB = K::extraB;

    P *A_ = memoryPool.allocate(MC * KC + (MC / MR) * XA);
    P *B_ = memoryPool.allocate(KC * NC + (NC / NR) * XB);

    for (long jc = 0; jc < n; jc += NC) {
	const long nc = (n - jc < NC) ? n - jc : NC;
	for (long pc = 0; pc < k; pc += KC) {
	    const long kc = (k - pc < KC) ? k - pc : KC;
	    const long kg = (kc + KG - 1) / KG;
	    const std::int32_t beta_ = (pc == 0) ? beta : 1;

	    //  panel strides (with the extra elements)
	    const long sA = kg * KG * MR + XA, sB = kg * KG * NR + XB;

	    for (long j = 0; j < nc; j += NR) {
		const long nr = (nc - j < NR) ? nc - j : NR;
		P *Bp = B_ + (j / NR) * sB;
		packPanel<NR, KG, P, UB>(nr, kc,
		                         B + pc * incRowB + (jc + j) * incColB,
		                         incColB, incRowB, Bp);
		K::template prepareB<UA, UB>(kg, Bp);
	    }
	    for (long ic = 0; ic < m; ic += MC) {
		const long mc = (m - ic < MC) ? m - ic : MC;
		for (long i = 0; i < mc; i += MR) {
		    const long mr = (mc - i < MR) ? mc - i : MR;
		    P *Ap = A_ + (i / MR) * sA;
		    packPanel<MR, KG, P, UA>(
		        mr, kc, A + (ic + i) * incRowA + pc * incColA, incRowA,
		        incColA, Ap);
		    K::template prepareA<UA, UB>(kg, Ap);
		}
		for (long j = 0; j < nc; j += NR) {
		    const long nr = (nc - j < NR) ? nc - j : NR;
		    const P *Bp = B_ + (j / NR) * sB;
		    for (long i = 0; i < mc; i += MR) {
			const long mr = (mc - i < MR) ? mc - i : MR;
			const P *Ap = A_ + (i / MR) * sA;
			std::int32_t *Ct =
			    C + (ic + i) * incRowC + (jc + j) * incColC;
			if (incRowC == 1 &&
			    (K::partial || (mr == MR && nr == NR))) {
			    K::template kernel<UA, UB>(kg, alpha, Ap, Bp, beta_,
			                               Ct, incColC, mr, nr);
			    continue;
			}
			std::int32_t C_[MR * NR];
			K::template kernel<UA, UB>(kg, alpha, Ap, Bp, 0, C_, MR,
			                           MR, NR);
			const std::uint32_t b = std::uint32_t(beta_);
			for (long jj = 0; jj < nr; ++jj) {
			    for (long ii = 0; ii < mr; ++ii) {
				std::int32_t &c =
				    Ct[ii * incRowC + jj * incColC];
				const std::uint32_t t =
				    std::uint32_t(C_[ii + MR * jj]);
				c = std::int32_t(
				    b == 0 ? t : b * std::uint32_t(c) + t);
			    }
			}
		    }
		}
	    }
	}
    }
    memoryPool.release(A_);
    memoryPool.release(B_);
}

} // namespace i8
} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_I8GEMM_DRIVER_H
