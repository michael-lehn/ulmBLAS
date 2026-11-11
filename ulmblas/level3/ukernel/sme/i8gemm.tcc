#ifndef ULMBLAS_LEVEL3_UKERNEL_SME_I8GEMM_TCC
#define ULMBLAS_LEVEL3_UKERNEL_SME_I8GEMM_TCC 1

//
//  ulmBLAS: GEMM with A and B in 8-bit integers (signed or unsigned) and C
//  in int32 for SME2 (SVL = 512 bits, Apple M4, M5):
//
//      C <- alpha*A*B + beta*C     (modulo 2^32)
//
//  The 4-way outer products (SMOPA, UMOPA, SUMOPA, USMOPA into 32-bit
//  tiles) multiply 4 consecutive values of k per instruction, so the
//  panels are packed in groups of 4 values of k: for the panel of 32 rows
//  of A (or 32 columns of B) and the group g
//
//      buffer[128*g + 4*i + q] = A(i, 4*g + q),   0 <= i < 32, 0 <= q < 4,
//
//  zero padded in i and in k.  The 32 x 32 tile of C is held in ZA0.S to
//  ZA3.S as in the float kernel (sgemm.tcc, the operands of the outer
//  products swapped: a horizontal slice is a part of a column of C).  Per
//  group of 4 values of k: one load of 128 bytes of A, one of B, four outer
//  products.
//
//  The packed blocks are int8 (not converted to int32 as by the generic
//  driver), so this file has its own driver (i8gemm_sme).
//
#include <arm_neon.h>
#include <arm_sme.h>
#include <cstdint>
#include <cstring>
#include <ulmblas/auxiliary/memorypool.h>
#include <ulmblas/level3/fpsrscope.h>
#include <ulmblas/level3/ukernel/sme/fpsr.h>

namespace ulmBLAS {
namespace sme {

//
//  The 4-way outer product for the signedness of A (UA: unsigned) and B
//  (UB): zn is the vector of B, zm that of A.
//
template <bool UA, bool UB, unsigned T>
inline void
i8mopa(svbool_t p, svuint8_t b, svuint8_t a) __arm_streaming __arm_inout("za")
{
    if (!UA && !UB) {
	svmopa_za32_s8_m(T, p, p, svreinterpret_s8(b), svreinterpret_s8(a));
    } else if (UA && !UB) {
	svsumopa_za32_s8_m(T, p, p, svreinterpret_s8(b), a);
    } else if (!UA && UB) {
	svusmopa_za32_u8_m(T, p, p, b, svreinterpret_s8(a));
    } else {
	svmopa_za32_u8_m(T, p, p, b, a);
    }
}

//  columns 0 to n-1 (n <= 16) of the tiles T and T+1, see sgemm.tcc
template <unsigned T>
inline void
i32load_32x16(long n, const std::int32_t *C, long ldC, svbool_t p0,
              svbool_t p1) __arm_streaming __arm_inout("za")
{
    for (long c = 0; c < n; ++c) {
	svld1_hor_za32(T + 0, c, p0, C + c * ldC);
	svld1_hor_za32(T + 1, c, p1, C + c * ldC + 16);
    }
}

template <unsigned T>
inline void
i32store_32x16(long n, std::int32_t *C, long ldC, svbool_t p0,
               svbool_t p1) __arm_streaming __arm_in("za")
{
    for (long c = 0; c < n; ++c) {
	svst1_hor_za32(T + 0, c, p0, C + c * ldC);
	svst1_hor_za32(T + 1, c, p1, C + c * ldC + 16);
    }
}

//  C <- alpha*ZA + beta*C (C not read for beta = 0), modulo 2^32
template <unsigned T>
inline void
i32update_32x16(long n, std::uint32_t alpha, std::uint32_t beta,
                std::int32_t *C_, long ldC,
                svcount_t pc) __arm_streaming __arm_in("za")
{
    const svbool_t p = svptrue_b32();
    const svuint32_t zero = svdup_u32(0);
    std::uint32_t *C = reinterpret_cast<std::uint32_t *>(C_);

    for (long c = 0; c < n; ++c) {
	std::uint32_t *Cc = C + c * ldC;

	svuint32_t z0 = svmul_n_u32_x(p, svread_hor_za32_u32_m(zero, p, T, c),
	                              alpha);
	svuint32_t z1 = svmul_n_u32_x(
	    p, svread_hor_za32_u32_m(zero, p, T + 1, c), alpha);

	if (beta != 0) {
	    svuint32x2_t x = svld1_u32_x2(pc, Cc);

	    z0 = svmla_n_u32_x(p, z0, svget2(x, 0), beta);
	    z1 = svmla_n_u32_x(p, z1, svget2(x, 1), beta);
	}
	svst1_u32_x2(pc, Cc, svcreate2(z0, z1));
    }
}

//
//  The rows 0 to mr-1 and columns 0 to nr-1 of the 32 x 32 tile of C (unit
//  row stride, column stride ldC); kg groups of 4 values of k.
//
template <bool UA, bool UB>
__arm_new("za") __arm_locally_streaming inline void i8gemm_32x32_streaming(
    long kg, std::int32_t alpha, const std::uint8_t *A, const std::uint8_t *B,
    std::int32_t beta, std::int32_t *C, long ldC, long mr, long nr)
{
    const svbool_t p = svptrue_b8();
    const svcount_t pc = svptrue_c8();
    const bool direct = alpha == 1 && (beta == 0 || beta == 1);

    const int64_t m = mr;
    const svbool_t p0 = svwhilelt_b32_s64(int64_t(0), m);
    const svbool_t p1 = svwhilelt_b32_s64(int64_t(16), m);
    const svcount_t pcr = svwhilelt_c32_s64(int64_t(0), m, 2);
    const long n0 = (nr < 16) ? nr : 16;
    const long n1 = nr - n0;

    svzero_za();
    if (direct && beta == 1) {
	i32load_32x16<0>(n0, C, ldC, p0, p1);
	i32load_32x16<2>(n1, C + 16 * ldC, ldC, p0, p1);
    }

    for (long g = 0; g < kg; ++g) {
	svuint8x2_t a = svld1_u8_x2(pc, A);
	svuint8x2_t b = svld1_u8_x2(pc, B);

	i8mopa<UA, UB, 0>(p, svget2(b, 0), svget2(a, 0));
	i8mopa<UA, UB, 1>(p, svget2(b, 0), svget2(a, 1));
	i8mopa<UA, UB, 2>(p, svget2(b, 1), svget2(a, 0));
	i8mopa<UA, UB, 3>(p, svget2(b, 1), svget2(a, 1));

	A += 128;
	B += 128;
    }

    if (direct) {
	i32store_32x16<0>(n0, C, ldC, p0, p1);
	i32store_32x16<2>(n1, C + 16 * ldC, ldC, p0, p1);
    } else {
	i32update_32x16<0>(n0, std::uint32_t(alpha), std::uint32_t(beta), C,
	                   ldC, pcr);
	i32update_32x16<2>(n1, std::uint32_t(alpha), std::uint32_t(beta),
	                   C + 16 * ldC, ldC, pcr);
    }
}

//
//  Packing of one panel: mn <= 32 rows of A (or columns of B), element
//  (i, l) at X[i*incMN + l*incK], 0 <= l < k, into the layout above (kg =
//  ceil(k/4) groups).  NEON for full panels with incK = 1 (rows of A, or
//  columns of B, contiguous in k) or incMN = 1 (contiguous in i).
//
inline void
i8pack_generic(long mn, long k, const std::uint8_t *X, long incMN, long incK,
               long g0, std::uint8_t *buffer)
{
    const long kg = (k + 3) / 4;
    for (long g = g0; g < kg; ++g) {
	std::uint8_t *b = buffer + 128 * g;
	for (long i = 0; i < 32; ++i) {
	    for (long q = 0; q < 4; ++q) {
		const long l = 4 * g + q;
		b[4 * i + q] = (i < mn && l < k) ? X[i * incMN + l * incK] : 0;
	    }
	}
    }
}

//  incK = 1, mn = 32: per 4 rows and 16 values of k a 4 x 4 transposition
//  of 32-bit words
inline void
i8pack_kcontig(long k, const std::uint8_t *X, long incMN,
               std::uint8_t *buffer)
{
    const long k16 = k / 16;
    for (long r = 0; r < 32; r += 4) {
	const std::uint8_t *x0 = X + r * incMN;
	const std::uint8_t *x1 = x0 + incMN;
	const std::uint8_t *x2 = x1 + incMN;
	const std::uint8_t *x3 = x2 + incMN;
	for (long c = 0; c < k16; ++c) {
	    uint32x4_t q0 = vreinterpretq_u32_u8(vld1q_u8(x0 + 16 * c));
	    uint32x4_t q1 = vreinterpretq_u32_u8(vld1q_u8(x1 + 16 * c));
	    uint32x4_t q2 = vreinterpretq_u32_u8(vld1q_u8(x2 + 16 * c));
	    uint32x4_t q3 = vreinterpretq_u32_u8(vld1q_u8(x3 + 16 * c));
	    uint64x2_t t0 = vreinterpretq_u64_u32(vtrn1q_u32(q0, q1));
	    uint64x2_t t1 = vreinterpretq_u64_u32(vtrn2q_u32(q0, q1));
	    uint64x2_t t2 = vreinterpretq_u64_u32(vtrn1q_u32(q2, q3));
	    uint64x2_t t3 = vreinterpretq_u64_u32(vtrn2q_u32(q2, q3));
	    std::uint8_t *b = buffer + 512 * c + 4 * r;
	    vst1q_u8(b, vreinterpretq_u8_u64(vtrn1q_u64(t0, t2)));
	    vst1q_u8(b + 128, vreinterpretq_u8_u64(vtrn1q_u64(t1, t3)));
	    vst1q_u8(b + 256, vreinterpretq_u8_u64(vtrn2q_u64(t0, t2)));
	    vst1q_u8(b + 384, vreinterpretq_u8_u64(vtrn2q_u64(t1, t3)));
	}
    }
    i8pack_generic(32, k, X, incMN, 1, 4 * k16, buffer);
}

//  incMN = 1, mn = 32: per group the 4 columns of 32 bytes interleaved
//  (ZIP of bytes, then of halfwords)
inline void
i8pack_mncontig(long k, const std::uint8_t *X, long incK,
                std::uint8_t *buffer)
{
    const long k4 = k / 4;
    for (long g = 0; g < k4; ++g) {
	const std::uint8_t *x = X + 4 * g * incK;
	std::uint8_t *b = buffer + 128 * g;
	for (long h = 0; h < 2; ++h) {
	    uint8x16_t c0 = vld1q_u8(x + 16 * h);
	    uint8x16_t c1 = vld1q_u8(x + incK + 16 * h);
	    uint8x16_t c2 = vld1q_u8(x + 2 * incK + 16 * h);
	    uint8x16_t c3 = vld1q_u8(x + 3 * incK + 16 * h);
	    uint16x8_t z01l = vreinterpretq_u16_u8(vzip1q_u8(c0, c1));
	    uint16x8_t z01h = vreinterpretq_u16_u8(vzip2q_u8(c0, c1));
	    uint16x8_t z23l = vreinterpretq_u16_u8(vzip1q_u8(c2, c3));
	    uint16x8_t z23h = vreinterpretq_u16_u8(vzip2q_u8(c2, c3));
	    std::uint8_t *bh = b + 64 * h;
	    vst1q_u8(bh, vreinterpretq_u8_u16(vzip1q_u16(z01l, z23l)));
	    vst1q_u8(bh + 16, vreinterpretq_u8_u16(vzip2q_u16(z01l, z23l)));
	    vst1q_u8(bh + 32, vreinterpretq_u8_u16(vzip1q_u16(z01h, z23h)));
	    vst1q_u8(bh + 48, vreinterpretq_u8_u16(vzip2q_u16(z01h, z23h)));
	}
    }
    i8pack_generic(32, k, X, 1, incK, k4, buffer);
}

inline void
i8pack(long mn, long k, const std::uint8_t *X, long incMN, long incK,
       std::uint8_t *buffer)
{
    if (mn == 32 && incK == 1) {
	i8pack_kcontig(k, X, incMN, buffer);
    } else if (mn == 32 && incMN == 1) {
	i8pack_mncontig(k, X, incK, buffer);
    } else {
	i8pack_generic(mn, k, X, incMN, incK, 0, buffer);
    }
}

//
//  The driver: blocks of MC x KC of A and KC x NC of B (KC a multiple of 4),
//  32 x 32 tiles of C.  C with unit row stride is updated by the kernel,
//  otherwise through a local tile.  The products wrap around modulo 2^32;
//  alpha*(S1 + S2) = alpha*S1 + alpha*S2 modulo 2^32, so the blocks of k
//  are accumulated with alpha and beta = 1 after the first block.
//
template <bool UA, bool UB>
void
i8gemm_sme(long m, long n, long k, std::int32_t alpha, const std::uint8_t *A,
           long incRowA, long incColA, const std::uint8_t *B, long incRowB,
           long incColB, std::int32_t beta, std::int32_t *C, long incRowC,
           long incColC)
{
    constexpr long MC = 512, KC = 4096, NC = 1024;

    static MemoryPool<std::uint8_t> memoryPool;

    const FpsrScope fpsrScope; // see level3/fpsrscope.h

    std::uint8_t *A_ = memoryPool.allocate(MC * KC);
    std::uint8_t *B_ = memoryPool.allocate(KC * NC);

    for (long jc = 0; jc < n; jc += NC) {
	const long nc = (n - jc < NC) ? n - jc : NC;
	for (long pc = 0; pc < k; pc += KC) {
	    const long kc = (k - pc < KC) ? k - pc : KC;
	    const long kg = (kc + 3) / 4;
	    const std::int32_t beta_ = (pc == 0) ? beta : 1;

	    for (long j = 0; j < nc; j += 32) {
		const long nr = (nc - j < 32) ? nc - j : 32;
		i8pack(nr, kc, B + pc * incRowB + (jc + j) * incColB, incColB,
		       incRowB, B_ + (j / 32) * kg * 128);
	    }
	    for (long ic = 0; ic < m; ic += MC) {
		const long mc = (m - ic < MC) ? m - ic : MC;
		for (long i = 0; i < mc; i += 32) {
		    const long mr = (mc - i < 32) ? mc - i : 32;
		    i8pack(mr, kc, A + (ic + i) * incRowA + pc * incColA,
		           incRowA, incColA, A_ + (i / 32) * kg * 128);
		}
		for (long j = 0; j < nc; j += 32) {
		    const long nr = (nc - j < 32) ? nc - j : 32;
		    const std::uint8_t *Bp = B_ + (j / 32) * kg * 128;
		    for (long i = 0; i < mc; i += 32) {
			const long mr = (mc - i < 32) ? mc - i : 32;
			const std::uint8_t *Ap = A_ + (i / 32) * kg * 128;
			std::int32_t *Ct =
			    C + (ic + i) * incRowC + (jc + j) * incColC;
			if (incRowC == 1) {
			    i8gemm_32x32_streaming<UA, UB>(
			        kg, alpha, Ap, Bp, beta_, Ct, incColC, mr, nr);
			    continue;
			}
			std::int32_t C_[32 * 32];
			i8gemm_32x32_streaming<UA, UB>(kg, alpha, Ap, Bp, 0, C_,
			                               32, 32, 32);
			const std::uint32_t b = std::uint32_t(beta_);
			for (long jj = 0; jj < nr; ++jj) {
			    for (long ii = 0; ii < mr; ++ii) {
				std::int32_t &c =
				    Ct[ii * incRowC + jj * incColC];
				const std::uint32_t t =
				    std::uint32_t(C_[ii + 32 * jj]);
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

} // namespace sme
} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_UKERNEL_SME_I8GEMM_TCC
